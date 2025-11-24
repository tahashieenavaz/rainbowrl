import torch
from typing import NewType
from types import SimpleNamespace
from atarihns import calculate_hns
from atarihelpers import process_state, make_environment
from .RainbowNetwork import RainbowNetwork
from .PrioritizedReplayBuffer import PrioritizedReplayBuffer

LossValue = NewType("LossValue", float)


class Agent:
    def __init__(
        self,
        environment: str,
        lr: float = 0.0001,
        training_starts: int = 80_000,
        training_frequency: int = 4,
        embedding_dimension: int = 256,
        activation_fn=torch.nn.GELU,
        num_atoms: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        initial_beta: float = 0.4,
        timesteps: int = 10_000_000,
        buffer_size: int = 1_000_00,
        batch_size: int = 256,
        image_size: int = 84,
        alpha: float = 0.5,
        steps: int = 3,
        max_priority: float = 1.0,
        epsilon: float = 1e-6,
        gamma: float = 0.99,
        verbose: bool = True,
        tau: float = 0.005,
    ):
        self.environment_identifier = environment
        self.environment = make_environment(environment)
        self.training_starts = training_starts
        self.training_frequency = training_frequency
        self.action_dimension = self.environment.action_space.n
        self.delta_z = (vmax - vmin) / (num_atoms - 1)
        self.initial_beta = initial_beta
        self.timesteps = timesteps
        self.verbose = verbose
        self.steps = steps
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = num_atoms
        self.batch_size = batch_size
        self.image_size = image_size

        self.network = RainbowNetwork(
            embedding_dimension=embedding_dimension,
            activation_fn=activation_fn,
            num_atoms=num_atoms,
            action_dimension=self.action_dimension,
            vmax=vmax,
            vmin=vmin,
        )
        self.target = RainbowNetwork(
            embedding_dimension=embedding_dimension,
            activation_fn=activation_fn,
            num_atoms=num_atoms,
            action_dimension=self.action_dimension,
            vmax=vmax,
            vmin=vmin,
        )
        self.target.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            buffer_size, image_size, steps, max_priority, alpha, epsilon
        )

        self.t = 0

    @property
    def beta(self) -> float:
        initial_beta = self.initial_beta
        return min(
            1.0, initial_beta + self.t * (1.0 - initial_beta) / self.total_timesteps
        )

    def loop(self):
        total_loss = []
        total_hns = []
        total_rewards = []

        _episode = 0

        while self.t < self.timesteps:
            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            state, _ = self.env.reset()

            while not done:
                _episode += 1

                state = process_state(state, self.image_size)
                action = self.action(state)
                next_state, reward, truncated, terminated, _ = self.environment.step(
                    action
                )
                done = truncated or terminated
                self.buffer.add(state, next_state, action, reward, done)

                loss = self.train()
                episode_loss += loss.item()

                self.tick()

            if self.verbose:
                print(
                    f"episode: {_episode}, t: {self.t}, loss: {episode_loss}, hns: {hns}, reward: {episode_reward}"
                )

            hns = calculate_hns(self.environment_identifier, episode_reward)
            total_hns.append(hns)
            total_rewards.append(episode_reward)
            total_loss.append(episode_loss)

            return SimpleNamespace(
                **{"loss": total_loss, "hns": total_hns, "reward": total_rewards}
            )

    @torch.no_grad()
    def action(self, state: torch.Tensor):
        q_dist = self.network(state)
        q_values = torch.sum(q_dist * self.support, dim=2)
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def train(self) -> LossValue:
        if self.t < self.training_starts:
            return 0.0

        if self.t % self.training_frequency != 0:
            return 0.0

        self.reset_noise()

        states, actions, rewards, next_states, terminations, indices, weights = (
            self.buffer.sample(self.batch_size, self.beta)
        )

        self.optimizer.zero_grad()
        loss = self.loss(
            states, actions, rewards, next_states, terminations, indices, weights
        )
        loss.backward()
        self.optimizer.step()

        self.update()

        return loss.item()

    def loss(
        self, states, actions, rewards, next_states, terminations, weights, indices
    ):
        distribution = self.network(states)
        predicted_distribution = distribution.gather(
            1, actions.unsqueeze(-1).expand(-1, -1, self.num_atoms)
        ).squeeze(1)
        log_pred = torch.log(predicted_distribution.clamp(min=1e-5, max=1 - 1e-5))
        target_pmfs = self.get_pmfs_target(next_states, rewards, terminations)
        loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
        new_priorities = loss_per_sample.detach().cpu().numpy()
        self.buffer.update_priorities(indices, new_priorities)
        return (loss_per_sample * weights.squeeze()).mean()

    def tick(self):
        self.t += 1

    def reset_noise(self):
        self.network.reset_noise()
        self.target.reset_noise()

    @torch.no_grad()
    def get_pmfs_target(self, next_states, rewards, terminations):
        next_dist = self.target(next_states)
        support = self.target.support

        next_dist_online = self.network(next_states)
        next_q_online = torch.sum(next_dist_online * support, dim=2)
        best_actions = torch.argmax(next_q_online, dim=1)
        next_pmfs = next_dist[torch.arange(self.batch_size), best_actions]

        gamma_n = self.gamma**self.steps
        next_atoms = rewards + gamma_n * support * (1 - terminations.float())
        tz = next_atoms.clamp(self.vmin, self.vmax)

        delta_z = self.delta_z
        b = (tz - self.vmin) / delta_z
        l = b.floor().clamp(0, self.num_atoms - 1)
        u = b.ceil().clamp(0, self.num_atoms - 1)
        d_m_l = (u.float() + (l == b).float() - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs

        target_pmfs = torch.zeros_like(next_pmfs)
        for i in range(target_pmfs.size(0)):
            target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
            target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        return target_pmfs

    def update(self):
        tau = self.tau
        for target_param, param in zip(
            self.target.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
