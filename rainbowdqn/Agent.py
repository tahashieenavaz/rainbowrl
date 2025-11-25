import torch
import numpy
from typing import NewType
from types import SimpleNamespace
from atarihns import calculate_hns
from atarihelpers import process_state, make_environment
from collections import deque
from dataclasses import dataclass
from baloot import acceleration_device
from .RainbowNetwork import RainbowNetwork
from .PrioritizedReplayBuffer import PrioritizedReplayBuffer

LossValue = NewType("LossValue", float)


@dataclass
class Agent:
    environment: str
    lr: float = 0.0000625
    training_starts: int = 80_000
    training_frequency: int = 4
    embedding_dimension: int = 512
    stream_activation_function = torch.nn.GELU
    convolution_activation_function = torch.nn.GELU
    num_atoms: int = 51
    vmin: float = -10.0
    vmax: float = 10.0
    initial_beta: float = 0.4
    timesteps: int = 10_000_000
    buffer_size: int = 500_000
    batch_size: int = 32
    image_size: int = 84
    alpha: float = 0.5
    steps: int = 3
    max_priority: float = 1.0
    epsilon: float = 1e-6
    gamma: float = 0.99
    adam_epsilon: float = 1.5e-4
    adam_betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    record: bool = False
    record_every: int = 50
    swap_frequency: int = 2000

    def __init__(
        self,
    ):
        self.environment_identifier = self.environment
        self.environment = make_environment(
            self.environment_identifier,
            record=self.record,
            record_every=self.record_every,
        )
        self.action_dimension = self.environment.action_space.n
        self.delta_z = (self.vmax - self.vmin) / (self.num_atoms - 1)
        self.device = acceleration_device()

        self.network = RainbowNetwork(
            embedding_dimension=self.embedding_dimension,
            stream_activation_function=self.stream_activation_function,
            convolution_activation_function=self.convolution_activation_function,
            num_atoms=self.num_atoms,
            action_dimension=self.action_dimension,
            vmax=self.vmax,
            vmin=self.vmin,
        ).to(self.device)
        self.target = RainbowNetwork(
            embedding_dimension=self.embedding_dimension,
            stream_activation_function=self.stream_activation_function,
            convolution_activation_function=self.convolution_activation_function,
            num_atoms=self.num_atoms,
            action_dimension=self.action_dimension,
            vmax=self.vmax,
            vmin=self.vmin,
        ).to(self.device)
        self.target.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
        )
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.buffer_size,
            image_size=self.image_size,
            steps=self.steps,
            max_priority=self.max_priority,
            alpha=self.alpha,
            epsilon=self.epsilon,
            gamma=self.gamma,
            device=self.device,
        )
        self.t = 0

    @property
    def beta(self) -> float:
        initial_beta = self.initial_beta
        return min(1.0, initial_beta + self.t * (1.0 - initial_beta) / self.timesteps)

    @property
    def learning_starts(self):
        return self.training_starts

    def loop(self, verbose: bool = True):
        _episode = 0
        total_loss = []
        total_hns = []
        total_rewards = []
        feeding_states = deque(maxlen=4)

        while self.t < self.timesteps:
            _episode += 1
            done = False
            episode_reward = 0.0
            episode_loss = 0.0

            state, _ = self.environment.reset()
            state = process_state(state, self.image_size)

            for _ in range(4):
                feeding_states.append(state)

            while not done:
                current_states_numpy = numpy.array(feeding_states)
                current_states_torch = torch.from_numpy(current_states_numpy)
                current_states_torch = current_states_torch.unsqueeze(0).to(self.device)

                action = self.action(current_states_torch)
                next_state, reward, truncated, terminated, _ = self.environment.step(
                    action
                )
                episode_reward += reward
                next_state_processed = process_state(next_state, self.image_size)

                temporal_next_states = feeding_states.copy()
                temporal_next_states.append(next_state_processed)
                temporal_next_states_numpy = numpy.array(temporal_next_states)

                done = truncated or terminated
                self.buffer.add(
                    current_states_numpy,
                    temporal_next_states_numpy,
                    action,
                    reward,
                    done,
                )
                loss = self.train()
                episode_loss += loss
                self.tick()
                feeding_states.append(next_state_processed)

            feeding_states.clear()

            hns = calculate_hns(self.environment_identifier, episode_reward)
            total_hns.append(hns)
            total_rewards.append(episode_reward)
            total_loss.append(episode_loss)

            if verbose:
                print(
                    f"episode: {_episode}, t: {self.t}, loss: {episode_loss}, hns: {hns}, reward: {episode_reward}, beta: {self.beta}",
                    flush=True,
                )

        return SimpleNamespace(
            **{"loss": total_loss, "hns": total_hns, "rewards": total_rewards}
        )

    @torch.no_grad()
    def action(self, state: torch.Tensor):
        if self.t < self.training_starts:
            return self.environment.action_space.sample()

        # to add noise and explore
        self.network.reset_noise()

        logits = self.network(state)
        q_dist = torch.softmax(logits, dim=2)
        q_values = torch.sum(q_dist * self.network.support, dim=2)
        return q_values.argmax(dim=1).item()

    def train(self) -> LossValue:
        if self.t < self.training_starts:
            return 0.0

        if self.t % self.training_frequency != 0:
            return 0.0

        self.reset_noise()

        sample_result = self.buffer.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, terminations, weights, indices = (
            sample_result
        )
        self.optimizer.zero_grad()
        loss = self.loss(
            states, actions, rewards, next_states, terminations, weights, indices
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        self.update()
        return loss.item()

    def loss(
        self, states, actions, rewards, next_states, terminations, weights, indices
    ):
        target_pmfs = self.get_pmfs_target(next_states, rewards, terminations)
        logits = self.network(states)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        actions = actions.view(-1, 1, 1)
        actions = actions.expand(-1, -1, self.num_atoms)
        actions = actions.long()

        current_log_probs = log_probs.gather(1, actions).squeeze(1)
        loss_per_sample = -(target_pmfs * current_log_probs).sum(dim=1)
        new_priorities = loss_per_sample.detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)
        return (loss_per_sample * weights.view(-1)).mean()

    def tick(self):
        self.t += 1

    def reset_noise(self):
        self.network.reset_noise()
        self.target.reset_noise()

    @torch.no_grad()
    def get_pmfs_target(self, next_states, rewards, terminations):
        rewards = rewards.view(-1, 1)
        terminations = terminations.view(-1, 1).float()
        termination_mask = 1 - terminations

        batch_size = next_states.size(0)

        online_logits = self.network(next_states)
        online_probs = torch.softmax(online_logits, dim=2)
        next_q_online = (online_probs * self.network.support).sum(dim=2)
        best_actions = next_q_online.argmax(dim=1)

        target_logits = self.target(next_states)
        target_probs = torch.softmax(target_logits, dim=2)
        next_dist = target_probs[range(batch_size), best_actions]

        projected_atoms = (
            rewards
            + (self.gamma**self.steps)
            * self.network.support.view(1, -1)
            * termination_mask
        )
        projected_atoms = projected_atoms.clamp(self.vmin, self.vmax)
        b = (projected_atoms - self.vmin) / self.delta_z
        b = b.clamp(0, self.num_atoms - 1)

        lower_idx = b.floor().long().clamp(0, self.num_atoms - 1)
        upper_idx = lower_idx + 1
        upper_weight = b - lower_idx.float()
        lower_weight = 1.0 - upper_weight
        upper_idx = upper_idx.clamp(0, self.num_atoms - 1)
        target_pmfs = torch.zeros_like(next_dist)
        target_pmfs.scatter_add_(1, lower_idx, next_dist * lower_weight)
        target_pmfs.scatter_add_(1, upper_idx, next_dist * upper_weight)
        return target_pmfs

    def update(self):
        if self.t % self.swap_frequency == 0:
            self.target.load_state_dict(self.network.state_dict())
