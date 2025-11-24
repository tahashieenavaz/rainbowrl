import torch
from .NoisyLinear import NoisyLinear


class RainbowNetwork(torch.nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        action_dimension: int,
        num_atoms: int,
        activation_fn: torch.nn.Module,
        vmin: float,
        vmax: float,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.action_dimension = action_dimension
        self.register_buffer("support", torch.linspace(vmin, vmax, num_atoms))

        self.phi = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.Flatten(),
        )
        self.value = torch.nn.Sequential(
            NoisyLinear(3136, embedding_dimension),
            activation_fn(),
            NoisyLinear(embedding_dimension, num_atoms),
        )
        self.advantage = torch.nn.Sequential(
            NoisyLinear(3136, embedding_dimension),
            activation_fn(),
            NoisyLinear(embedding_dimension, num_atoms * action_dimension),
        )

    def reset_noise(self):
        for layer in self.value:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, state):
        features = self.phi(state / 255.0)
        value = self.value(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage(features).view(
            -1, self.action_dimension, self.num_atoms
        )
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(q_values, dim=2)
