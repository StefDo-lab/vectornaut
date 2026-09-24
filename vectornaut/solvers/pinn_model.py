import torch
import torch.nn as nn


class GenericPINN(nn.Module):
    """
    A customizable, lightweight Neural Network (PINN-lite) that maps
    an independent variable x to a dependent variable y.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
