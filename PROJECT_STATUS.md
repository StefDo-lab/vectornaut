# Vectornaut Omni - Project Status & Handoff Protocol

This document serves as the single source of truth for the **Vectornaut Omni** project state. It is designed to quickly bring any successor AI agent up to speed when starting a new session.

---

## 🎯 Project Overview & Core Goals

**Vectornaut Omni** is an evolution of a specialized "shark-skin drag simulator" into an open-ended, multi-domain physics solver platform. 
* **Target Vision**: A comprehensive, web-based dashboard where users can input design prompts (e.g., riblet structures, thermal shields, wings), which are then analyzed, simulated, and visualized.
* **Core Domains**: Fluid dynamics (drag reduction, boundary layers), thermodynamics (heat transfer, transient shields), structural mechanics, and general 1D/2D partial differential equations (PDEs).

---

## 🏗️ System Architecture & File Structure

The project resides in:
* `C:\Users\Stefan\.gemini\antigravity\scratch\vectornaut_project`

Key files and components:
* **[web_server.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/web_server.py)**: FastAPI web server running the backend API (port `8080`).
* **`vectornaut/`**:
  * **[auditor.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/vectornaut/auditor.py)**: Audits design prompts, determines equations, boundary conditions, and routes to appropriate solvers.
  * **[solver_dispatcher.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/vectornaut/solver_dispatcher.py)**: Routes requests to specific solver backends and processes outputs.
  * **[script_generator.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/vectornaut/script_generator.py)**: Generates and runs Python simulation scripts with an automated self-correction loop.
  * **[pinn_solver.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/vectornaut/pinn_solver.py)**: Physics-Informed Neural Network solver for 2D PDEs.
  * **[fdm_solver.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/vectornaut/fdm_solver.py)**: Finite Difference Method solver for 2D PDEs.
* **`static/`**:
  * **[index.html](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/static/index.html)**: Premium dark-mode dashboard UI.
  * **[app.js](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/static/app.js)**: Client-side frontend logic (API calls, heatmap rendering, marked.js integration).
  * **[style.css](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/static/style.css)**: CSS stylesheets (Harmonious bionic styling).
* **`generated_scripts/`**: Directory where the autonomous Python solver scripts are saved.
* **`static/plots/`**: Location where custom Matplotlib charts from dynamic solver scripts are served.

---

## 🚀 Current Status & Key Features Implemented

### 1. Autonomous Script Solver (`dynamic_script`)
* **Self-Correction Loop**: When users query a physical scenario containing script-keywords (e.g. `"script"`, `"skript"`, `"dynamic solver"`), the system writes a custom Python script using the Gemini API.
* **Execution & Verification**: The script is executed in a subprocess. If it crashes (e.g., division by zero, library errors), the error traceback is sent back to Gemini. The script is corrected and re-run (up to 3 attempts).
* **Plotting & Visualization**: Generates custom Matplotlib plots, which are saved to `static/plots/` and dynamically rendered in the UI with a cache-busting timestamp parameter (`?t=...`).

### 2. 2D PDE Solvers (FDM & PINN)
* **PINN Solver**: A PyTorch-based solver for 2D boundary value problems. Supported by disk-caching of trained networks to prevent redundant CPU/GPU training.
* **FDM Solver**: A classical grid solver serving as a reference check.
* **Epoch Synchronization**: The UI epoch slider dynamically configures the solver's training epochs. Loaded cache files correctly identify trained epoch length.

### 3. Premium UI & Markdown Report
* **Marked.js Integration**: The "Discovery Report" is rendered from markdown using custom CSS (blockquotes, clean tables, code blocks).
* **Developer Controls**: Features a "Toggle Developer JSON" and "Download Raw JSON" button.
* **Optimization Chat History**: The chat history is preserved during consecutive parameter optimization runs (tracked via `keepChatHistory` JS flag).

---

## 🧪 Verification & Unit Tests

The codebase includes several active verification test scripts:
* **[test_script_generator.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/test_script_generator.py)**: Mocks script crashes and verifies that the self-correction loop successfully heals the code.
* **[test_live_script_solver.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/test_live_script_solver.py)**: Performs a live end-to-end API call to verify automatic routing, script generation, and plotting.
* **[test_parameter_propagation.py](file:///C:/Users/Stefan/.gemini/antigravity/scratch/vectornaut_project/test_parameter_propagation.py)**: Verifies that consecutive optimization runs correctly propagate override parameters (e.g., `slip_length`).

---

## 🔮 Next Steps & Roadmap

When starting a new session, the following tasks are planned:
1. **Dynamic Test Script Generation**:
   * Implement an automated routine where Vectornaut writes custom Python integration/validation test scripts *on the fly* to verify newly emerging physical scenarios or simulation parameters.
2. **Further Extension of Physical Models**:
   * Add more physical schemas (e.g., structural beam bending, acoustics, multiphase flows) to the default auditor rules.
3. **Heatmap & Canvas Optimizations**:
   * Optimize rendering for higher FDM/PINN grid resolutions (e.g., 100x100 points) on the HTML5 Canvas.
