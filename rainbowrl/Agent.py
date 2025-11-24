import torch
import numpy
from .RainbowNetwork import RainbowNetwork


class Agent:
    def __init__(self, environment: str):
        self.network = RainbowNetwork()
        self.target = RainbowNetwork()
        self.environment = environment
        self.optimizer = torch.optim.Adam(self.network.parameters())

        self.t = 0

    @torch.no_grad()
    def action(self, state: torch.Tensor):
        pass
