import torch
from .RainbowNetwork import RainbowNetwork
from .PrioritizedReplayBuffer import PrioritizedReplayBuffer
from typing import NewType

LossValue = NewType("LossValue", float)


class Agent:
    def __init__(
        self,
        environment: str,
        lr: float = 0.0001,
        training_starts: int = 80_000,
        training_frequency: int = 4,
        embedding_dimension: int = 512,
        activation_fn=torch.nn.GELU,
        num_atoms: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        initial_beta: float = 0.4,
        total_timesteps: int = 10_000_000,
        buffer_size: int = 1_000_00,
        image_size: int = 84,
        alpha: float = 0.5,
        steps: int = 3,
        max_priority: float = 1.0,
        epsilon: float = 1e-6,
    ):
        self.environment = environment
        self.alpha = alpha
        self.training_starts = training_starts
        self.training_frequency = training_frequency
        self.action_dimension = self.environment.action_space.n
        self.delta_z = (vmax - vmin) / (num_atoms - 1)
        self.initial_beta = initial_beta
        self.total_timesteps = total_timesteps

        self.network = RainbowNetwork(
            embedding_dimension, activation_fn, num_atoms, self.action_dimension
        )
        self.target = RainbowNetwork(
            embedding_dimension, activation_fn, num_atoms, self.action_dimension
        )
        self.target.load_state_dict(self.network.load_state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            buffer_size, image_size, steps, max_priority, alpha, epsilon
        )

        self.t = 0

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

        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def loss(self):
        pass

    def tick(self):
        self.t += 1

    def beta(self) -> float:
        initial_beta = self.initial_beta
        return min(
            1.0, initial_beta + self.t * (1.0 - initial_beta) / self.total_timesteps
        )
