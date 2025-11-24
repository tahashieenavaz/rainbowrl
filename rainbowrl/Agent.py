import torch
import numpy
from .RainbowNetwork import RainbowNetwork
from typing import NewType

LossValue = NewType("LossValue", float)


class Agent:
    def __init__(
        self,
        environment: str,
        lr: float = 0.0001,
        training_starts: int = 80_000,
        training_frequency: int = 4,
    ):
        self.environment = environment
        self.training_starts = training_starts
        self.training_frequency = training_frequency

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
