import torch
import torch.nn as nn


class GenericPINN(nn.Module):
    """
    A customizable, lightweight Neural Network (PINN-lite) that maps
    an independent variable x to a dependent variable y.
    """
    # Affine output scaling: y = output_shift + output_scale * net(x). The defaults (0, 1)
    # leave the network output unchanged (the 2D PINN relies on this); the 1D PINN sets
    # them from the boundary values. They are buffers, so cached weights carry them.
    _SCALING_DEFAULTS = {"output_shift": 0.0, "output_scale": 1.0}

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for name, default in self._SCALING_DEFAULTS.items():
            self.register_buffer(name, torch.full((1,), default))

    def set_output_scaling(self, shift: float, scale: float) -> None:
        with torch.no_grad():
            self.output_shift.fill_(float(shift))
            self.output_scale.fill_(float(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.output_scale + self.output_shift

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Models cached before the output scaling existed have no scaling buffers:
        # load them with the identity scaling they were trained with.
        for name, default in self._SCALING_DEFAULTS.items():
            if prefix + name not in state_dict:
                state_dict[prefix + name] = torch.full((1,), default)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
