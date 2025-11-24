import torch
import numpy
from .RainbowNetwork import RainbowNetwork


class Agent:
    def __init__(
        self, environment: str, lr: float = 0.0001, training_starts: int = 80_000
    ):
        self.environment = environment
        self.training_starts = training_starts

        self.network = RainbowNetwork()
        self.target = RainbowNetwork()
        self.target.load_state_dict(self.network.load_state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.t = 0

    @torch.no_grad()
    def action(self, state: torch.Tensor):
        pass

    def tick(self):
        self.t += 1

    def train(self):
        if self.t < self.training_starts:
            return

        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()

    def loss(self):
        pass
