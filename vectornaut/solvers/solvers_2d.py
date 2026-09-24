import re
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, convert_xor
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Any

from .pinn_model import GenericPINN


# ==========================================
# 2D PDE Helper Functions & Solvers
# ==========================================

def parse_rhs_2d(rhs_str: str, x_name: str, y_name: str, params: Dict[str, float]) -> sp.Expr:
    x = sp.Symbol(x_name)
    y = sp.Symbol(y_name)
    local_ns = {x_name: x, y_name: y}
    for p_name, p_val in params.items():
        local_ns[p_name] = p_val
    expr = sp.parse_expr(rhs_str.strip(), local_dict=local_ns, transformations=(standard_transformations + (convert_xor,)))
    return expr

def parse_bc_2d_string(bc_str: str, dep_name: str, ind_names: List[str], params: Dict[str, float]) -> Tuple[str, str, float]:
    """
    Parses a single 2D boundary condition string.
    Returns (edge, type, value) where:
      edge: 'left', 'right', 'bottom', 'top'
      type: 'dirichlet', 'neumann'
      value: float
    """
    lhs, rhs = bc_str.split("=")
    lhs = lhs.strip()
    rhs = rhs.strip()
    
    # 1. Determine value
    val_expr = sp.sympify(rhs)
    param_subs = {sp.Symbol(k): v for k, v in params.items()}
    val = float(val_expr.subs(param_subs))
    
    # 2. Determine edge
    edge = None
    x_var, y_var = ind_names[0], ind_names[1]
    lhs_clean = re.sub(r'\s+', '', lhs)
    
    if f"(0,{y_var})" in lhs_clean or f"(0.0,{y_var})" in lhs_clean or f"(0," in lhs_clean:
        edge = "left"
    elif f"(1,{y_var})" in lhs_clean or f"(1.0,{y_var})" in lhs_clean or f"(1," in lhs_clean:
        edge = "right"
    elif f"({x_var},0)" in lhs_clean or f"({x_var},0.0)" in lhs_clean or f",0)" in lhs_clean:
        edge = "bottom"
    elif f"({x_var},1)" in lhs_clean or f"({x_var},1.0)" in lhs_clean or f",1)" in lhs_clean:
        edge = "top"
    
    if edge is None:
        if "0" in lhs_clean:
            if "dy" in lhs_clean or "y" in lhs_clean:
                edge = "bottom"
            else:
                edge = "left"
        elif "1" in lhs_clean:
            if "dy" in lhs_clean or "y" in lhs_clean:
                edge = "top"
            else:
                edge = "right"
        else:
            edge = "left"
            
    # 3. Determine type (Dirichlet vs Neumann)
    is_derivative = False
    if "diff" in lhs or "d" + dep_name in lhs or "'" in lhs or "deriv" in lhs or "grad" in lhs:
        is_derivative = True
        
    bc_type = "neumann" if is_derivative else "dirichlet"
    
    return edge, bc_type, val

def solve_fdm_2d(
    rhs_expr: sp.Expr,
    bcs_parsed: Dict[str, Dict[str, Any]],
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    grid_size: int = 20,
    max_iter: int = 2000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves u_xx + u_yy = RHS on a square grid [0,1]x[0,1] using Finite Difference Method (Jacobi).
    """
    h = 1.0 / (grid_size - 1)
    x_vals = np.linspace(0.0, 1.0, grid_size)
    y_vals = np.linspace(0.0, 1.0, grid_size)
    
    # Evaluate RHS on the grid
    f_rhs = sp.lambdify((x_sym, y_sym), rhs_expr, "numpy")
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    F = f_rhs(X, Y)
    if isinstance(F, (int, float)):
        F = np.full_like(X, F)
        
    u = np.zeros((grid_size, grid_size))
    
    # Initialize boundary values
    if bcs_parsed['left']['type'] == 'dirichlet':
        u[0, :] = bcs_parsed['left']['value']
    if bcs_parsed['right']['type'] == 'dirichlet':
        u[-1, :] = bcs_parsed['right']['value']
    if bcs_parsed['bottom']['type'] == 'dirichlet':
        u[:, 0] = bcs_parsed['bottom']['value']
    if bcs_parsed['top']['type'] == 'dirichlet':
        u[:, -1] = bcs_parsed['top']['value']
        
    # Jacobi iteration
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Update interior points
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1] - h*h * F[i, j])
                
        # Enforce Neumann BCs
        if bcs_parsed['left']['type'] == 'neumann':
            val = bcs_parsed['left']['value']
            u[0, :] = u[1, :] - h * val
        if bcs_parsed['right']['type'] == 'neumann':
            val = bcs_parsed['right']['value']
            u[-1, :] = u[-2, :] + h * val
        if bcs_parsed['bottom']['type'] == 'neumann':
            val = bcs_parsed['bottom']['value']
            u[:, 0] = u[:, 1] - h * val
        if bcs_parsed['top']['type'] == 'neumann':
            val = bcs_parsed['top']['value']
            u[:, -1] = u[:, -2] + h * val
            
        # Check convergence
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            break
            
    return x_vals, y_vals, u

def solve_pytorch_pinn_2d(
    rhs_expr: sp.Expr,
    bcs_parsed: Dict[str, Dict[str, Any]],
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    epochs: int = 400
) -> Tuple[GenericPINN, List[float]]:
    """
    Trains a 2D PINN model to solve u_xx + u_yy = RHS.
    """
    torch.manual_seed(42)
    model = GenericPINN(input_dim=2, hidden_dim=32)
    
    # Initialize bias of the output layer to the average Dirichlet boundary condition value
    dirichlet_vals = [bc['value'] for bc in bcs_parsed.values() if bc['type'] == 'dirichlet']
    mean_val = sum(dirichlet_vals) / len(dirichlet_vals) if dirichlet_vals else 0.0
    with torch.no_grad():
        model.net[-1].bias.fill_(mean_val)
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Sample interior collocation points (20x20 grid)
    x_pde_vals = np.linspace(0.0, 1.0, 20)
    y_pde_vals = np.linspace(0.0, 1.0, 20)
    X, Y = np.meshgrid(x_pde_vals, y_pde_vals, indexing='ij')
    pts_pde = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy_pde = torch.tensor(pts_pde, dtype=torch.float32, requires_grad=True)
    
    f_pde_rhs = sp.lambdify((x_sym, y_sym), rhs_expr, 'torch')
    rhs_val = f_pde_rhs(xy_pde[:, 0:1], xy_pde[:, 1:2])
    if isinstance(rhs_val, (int, float)):
        rhs_val = torch.tensor(rhs_val)
        
    # Boundary points sets
    y_b = torch.linspace(0.0, 1.0, 20).view(-1, 1)
    xy_left = torch.cat([torch.zeros_like(y_b), y_b], dim=1)
    xy_left.requires_grad = True
    
    xy_right = torch.cat([torch.ones_like(y_b), y_b], dim=1)
    xy_right.requires_grad = True
    
    x_b = torch.linspace(0.0, 1.0, 20).view(-1, 1)
    xy_bottom = torch.cat([x_b, torch.zeros_like(x_b)], dim=1)
    xy_bottom.requires_grad = True
    
    xy_top = torch.cat([x_b, torch.ones_like(x_b)], dim=1)
    xy_top.requires_grad = True
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. PDE Loss: u_xx + u_yy - RHS = 0
        u = model(xy_pde)
        grads = torch.autograd.grad(u, xy_pde, torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        
        u_xx = torch.autograd.grad(u_x, xy_pde, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xy_pde, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        
        pde_loss = torch.mean((u_xx + u_yy - rhs_val) ** 2)
        
        # 2. BC Loss
        bc_loss = 0.0
        
        # Left edge
        u_l = model(xy_left)
        if bcs_parsed['left']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_l - bcs_parsed['left']['value']) ** 2)
        else: # neumann
            grads_l = torch.autograd.grad(u_l, xy_left, torch.ones_like(u_l), create_graph=True)[0]
            du_dx_l = grads_l[:, 0:1]
            bc_loss += torch.mean((du_dx_l - bcs_parsed['left']['value']) ** 2)
            
        # Right edge
        u_r = model(xy_right)
        if bcs_parsed['right']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_r - bcs_parsed['right']['value']) ** 2)
        else: # neumann
            grads_r = torch.autograd.grad(u_r, xy_right, torch.ones_like(u_r), create_graph=True)[0]
            du_dx_r = grads_r[:, 0:1]
            bc_loss += torch.mean((du_dx_r - bcs_parsed['right']['value']) ** 2)
            
        # Bottom edge
        u_bot = model(xy_bottom)
        if bcs_parsed['bottom']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_bot - bcs_parsed['bottom']['value']) ** 2)
        else: # neumann
            grads_b = torch.autograd.grad(u_bot, xy_bottom, torch.ones_like(u_bot), create_graph=True)[0]
            du_dy_b = grads_b[:, 1:2]
            bc_loss += torch.mean((du_dy_b - bcs_parsed['bottom']['value']) ** 2)
            
        # Top edge
        u_t = model(xy_top)
        if bcs_parsed['top']['type'] == 'dirichlet':
            bc_loss += torch.mean((u_t - bcs_parsed['top']['value']) ** 2)
        else: # neumann
            grads_t = torch.autograd.grad(u_t, xy_top, torch.ones_like(u_t), create_graph=True)[0]
            du_dy_t = grads_t[:, 1:2]
            bc_loss += torch.mean((du_dy_t - bcs_parsed['top']['value']) ** 2)
            
        total_loss = pde_loss + bc_loss * 5.0
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(float(total_loss.item()))
        
    return model, loss_history
