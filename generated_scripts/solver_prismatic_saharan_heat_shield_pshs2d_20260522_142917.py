import argparse
import json
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def solve_heat_equation(K, T_hot, T_cold, Nx, Ny, dx, dy):
    N = Nx * Ny
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)
    
    def idx(i, j):
        return i * Ny + j
    
    for i in range(Nx):
        for j in range(Ny):
            row = idx(i, j)
            if i == 0:
                A[row, row] = 1.0
                b[row] = T_hot
            elif i == Nx - 1:
                A[row, row] = 1.0
                b[row] = T_cold
            else:
                k_ip = 0.5 * (K[i+1, j] + K[i, j])
                k_im = 0.5 * (K[i-1, j] + K[i, j])
                
                if j == 0:
                    k_jp = 0.5 * (K[i, j+1] + K[i, j])
                    k_jm = k_jp
                    t_jp_idx = idx(i, j+1)
                elif j == Ny - 1:
                    k_jm = 0.5 * (K[i, j-1] + K[i, j])
                    k_jp = k_jm
                    t_jp_idx = idx(i, j-1)
                else:
                    k_jp = 0.5 * (K[i, j+1] + K[i, j])
                    k_jm = 0.5 * (K[i, j-1] + K[i, j])
                    t_jp_idx = idx(i, j+1)
                
                t_jm_idx = idx(i, j-1) if j > 0 else idx(i, j+1)
                
                coef_self = -(k_ip + k_im + k_jp + k_jm)
                A[row, row] = coef_self
                A[row, idx(i+1, j)] = k_ip
                A[row, idx(i-1, j)] = k_im
                
                if j == 0 or j == Ny - 1:
                    A[row, t_jp_idx] = k_jp + k_jm
                else:
                    A[row, t_jp_idx] = k_jp
                    A[row, t_jm_idx] = k_jm
                    
    A = A.tocsr()
    T_flat = spla.spsolve(A, b)
    return T_flat.reshape((Nx, Ny))

def main():
    parser = argparse.ArgumentParser(description="PSHS-2D Solver")
    parser.add_argument("--params", type=str, help="Path to input JSON params")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON results")
    parser.add_argument("--plot", type=str, required=True, help="Path to output PNG plot")
    args = parser.parse_args()
    
    params = {
        "T_hot": 100.0,
        "T_cold": 20.0,
        "simulation_coefficient": 0.088667
    }
    
    if args.params and os.path.exists(args.params):
        try:
            with open(args.params, "r") as f:
                params.update(json.load(f))
        except Exception as e:
            print(f"Error reading params: {e}")
            
    T_hot = float(params.get("T_hot", 100.0))
    T_cold = float(params.get("T_cold", 20.0))
    sim_coef = float(params.get("simulation_coefficient", 0.088667))
    sim_coef = max(1e-4, min(1.0, sim_coef))
    
    Nx, Ny = 20, 20
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    K_reference = np.ones((Nx, Ny))
    K_primary = np.ones((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if 0.2 <= x[i] <= 0.8:
                factor = (1.0 - sim_coef) * np.sin(np.pi * (x[i] - 0.2) / 0.6) * (0.5 + 0.5 * np.cos(2.0 * np.pi * y[j]))
                K_primary[i, j] = 1.0 - factor
                
    T_reference = solve_heat_equation(K_reference, T_hot, T_cold, Nx, Ny, dx, dy)
    T_primary = solve_heat_equation(K_primary, T_hot, T_cold, Nx, Ny, dx, dy)
    
    Q_reference = 0.0
    Q_primary = 0.0
    for j in range(Ny):
        k_ref_half = 0.5 * (K_reference[0, j] + K_reference[1, j])
        Q_reference += k_ref_half * (T_hot - T_reference[1, j]) / dx * dy
        
        k_pri_half = 0.5 * (K_primary[0, j] + K_primary[1, j])
        Q_primary += k_pri_half * (T_hot - T_primary[1, j]) / dx * dy
        
    if Q_reference == 0:
        performance_gain_pct = 0.0
    else:
        performance_gain_pct = float(100.0 * (Q_reference - Q_primary) / Q_reference)
        
    norm_ref = np.linalg.norm(T_reference)
    if norm_ref == 0:
        relative_error = 0.0
    else:
        relative_error = float(np.linalg.norm(T_primary - T_reference) / norm_ref)
        
    sample_points = []
    solution_primary = []
    solution_reference = []
    for i in range(Nx):
        for j in range(Ny):
            sample_points.append([float(x[i]), float(y[j])])
            solution_primary.append(float(T_primary[i, j]))
            solution_reference.append(float(T_reference[i, j]))
            
    output_data = {
        "success": True,
        "performance_gain_pct": performance_gain_pct,
        "relative_error": relative_error,
        "sample_points": sample_points,
        "solution_primary": solution_primary,
        "solution_reference": solution_reference,
        "primary_metric_value": float(Q_primary),
        "reference_metric_value": float(Q_reference)
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
        
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    X, Y = np.meshgrid(x, y)
    im = axes[0].contourf(X, Y, T_primary.T, levels=50, cmap="plasma")
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label("Temperature T", color="white")
    axes[0].set_title("2D Temperature Field (Bionic PSHS-2D)", color="white")
    axes[0].set_xlabel("x (Shield Thickness)")
    axes[0].set_ylabel("y (Shield Height)")
    
    y_mid_idx = Ny // 2
    axes[1].plot(x, T_primary[:, y_mid_idx], color="#00f0ff", linewidth=2.5, label="Bionic (PSHS-2D)")
    axes[1].plot(x, T_reference[:, y_mid_idx], color="#ff00ff", linestyle="--", linewidth=2.0, label="Reference (Solid)")
    axes[1].set_title(f"Centerline Profile at y = {y[y_mid_idx]:.2f}", color="white")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Temperature T")
    axes[1].grid(True, color="#444444", linestyle=":")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    plt.close()

if __name__ == "__main__":
    main()