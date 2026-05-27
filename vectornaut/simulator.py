import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .config import SimulatorOutput

class FlowPINN(nn.Module):
    """
    A simple Physics-Informed Neural Network (PINN-lite) that maps
    position y -> velocity u(y).
    """
    def __init__(self, hidden_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)

def run_simulation(
    slippage_coefficient: float,
    u_free: float = 1.5,
    c_pg: float = 2.0,
    epochs: int = 200,
    learning_rate: float = 0.015
) -> SimulatorOutput:
    """
    Runs the PINN training loop and compares it with the analytical Couette-Poiseuille flow solution.
    Solves u''(y) = -c_pg with BCs:
        u(0) = slippage_coefficient * u'(0)
        u(1) = u_free
    """
    torch.manual_seed(42)
    model = FlowPINN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 100 collocation points in domain [0, 1]
    y_pde = torch.linspace(0.0, 1.0, 100, requires_grad=True).view(-1, 1)
    
    # Boundary points
    y_0 = torch.tensor([[0.0]], requires_grad=True)
    y_1 = torch.tensor([[1.0]], requires_grad=True)
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE Loss: u''(y) + c_pg = 0
        u = model(y_pde)
        u_y = torch.autograd.grad(u, y_pde, torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_pde, torch.ones_like(u_y), create_graph=True)[0]
        pde_loss = torch.mean((u_yy + c_pg) ** 2)
        
        # Boundary Loss at y = 0: u(0) - lambda * u_y(0) = 0
        u_0 = model(y_0)
        u_y_0 = torch.autograd.grad(u_0, y_0, create_graph=True)[0]
        bc_0_loss = (u_0 - slippage_coefficient * u_y_0) ** 2
        
        # Boundary Loss at y = 1: u(1) - u_free = 0
        u_1 = model(y_1)
        bc_1_loss = (u_1 - u_free) ** 2
        
        total_loss = pde_loss + bc_0_loss + bc_1_loss
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(float(total_loss.item()))

    # Analytical Solution derivation
    # u(y) = -0.5 * c_pg * y^2 + A * y + B
    # u'(y) = -c_pg * y + A
    # BC 0: u(0) = B, u'(0) = A => B = lambda * A
    # BC 1: u(1) = -0.5 * c_pg + A + lambda * A = u_free
    # => A * (1 + lambda) = u_free + 0.5 * c_pg => A = (u_free + 0.5 * c_pg) / (1 + lambda)
    # => B = lambda * (u_free + 0.5 * c_pg) / (1 + lambda)
    
    lam = slippage_coefficient
    a_analytical = (u_free + 0.5 * c_pg) / (1.0 + lam)
    b_analytical = lam * a_analytical
    
    # Analytical wall shear stress at y=0 is u'(0) = A
    wall_shear_analytical = a_analytical
    
    # Smooth wall shear stress (lam = 0)
    wall_shear_smooth = u_free + 0.5 * c_pg
    
    # Performance gain (drag reduction %)
    performance_gain = (1.0 - (wall_shear_analytical / wall_shear_smooth)) * 100.0
    
    # Evaluate model and compare
    y_test_np = np.linspace(0.0, 1.0, 20)
    y_test_torch = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        u_pinn_torch = model(y_test_torch).view(-1)
        u_pinn_np = u_pinn_torch.numpy()
        
    u_analytical_np = -0.5 * c_pg * (y_test_np ** 2) + a_analytical * y_test_np + b_analytical
    
    # Calculate PINN wall shear stress at y=0 using autograd
    y_0_eval = torch.tensor([[0.0]], requires_grad=True)
    u_0_eval = model(y_0_eval)
    wall_shear_pinn = torch.autograd.grad(u_0_eval, y_0_eval)[0].item()
    
    # Relative L2-like error
    abs_diff = np.abs(u_pinn_np - u_analytical_np)
    ref_norm = np.abs(u_analytical_np)
    # Prevent division by zero
    relative_err = float(np.sum(abs_diff) / np.sum(ref_norm + 1e-8))
    
    return SimulatorOutput(
        solver_method="pinn",
        epochs_trained=epochs,
        final_loss=loss_history[-1],
        loss_history=loss_history,
        performance_gain_pct=performance_gain,
        relative_error=relative_err,
        sample_points=y_test_np.tolist(),
        solution_primary=u_pinn_np.tolist(),
        solution_reference=u_analytical_np.tolist(),
        primary_metric_value=wall_shear_pinn,
        reference_metric_value=wall_shear_analytical
    )
