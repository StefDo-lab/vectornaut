import os
import json
import re
from datetime import datetime
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..storage import models_dir


def _models_dir() -> str:
    return models_dir()


def load_cached_pinn(
    prefix: str,
    label: str,
    loaded_message: str,
    gov_eq: str,
    bcs: List[str],
    params: Dict[str, float],
    build_model: Callable[[], torch.nn.Module],
    on_loaded: Callable[[torch.nn.Module, Dict[str, Any]], None],
    domain: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Looks for a pre-trained PINN model (<prefix>*.json metadata + matching .pth weights)
    whose governing equation, boundary conditions, params (and domain boundaries, if given)
    match the current problem. On a match the weights are loaded into build_model() and
    on_loaded(model, meta) is called. Returns True if a cached model was used.
    label is "" for the 1D and "2D " for the 2D log messages.
    """
    pretrained_loaded = False
    # Check if we have a pre-trained model matching these exact conditions
    try:
        cache_dir = _models_dir()
        if os.path.exists(cache_dir):
            # Find matching json files
            for f_name in sorted(os.listdir(cache_dir), reverse=True):
                if f_name.endswith(".json") and f_name.startswith(prefix):
                    meta_path = os.path.join(cache_dir, f_name)
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        # Check if PDE, BCs and domain boundaries match
                        if (meta.get("governing_equation") == gov_eq and
                            meta.get("boundary_conditions") == bcs and
                            (domain is None or (
                                abs(meta.get("domain_min", 0.0) - domain[0]) < 1e-5 and
                                abs(meta.get("domain_max", 1.0) - domain[1]) < 1e-5))):

                            # Check params match within tolerance
                            params_match = True
                            meta_params = meta.get("params", {})
                            for k, v in params.items():
                                if k not in meta_params or abs(meta_params[k] - v) > 1e-5:
                                    params_match = False
                                    break

                            if params_match:
                                pth_name = f_name.replace(".json", ".pth")
                                pth_path = os.path.join(cache_dir, pth_name)
                                if os.path.exists(pth_path):
                                    print(f"[*] Found matching pre-trained {label}PINN model weights: {pth_path}")
                                    model = build_model()
                                    model.load_state_dict(torch.load(pth_path))
                                    on_loaded(model, meta)
                                    pretrained_loaded = True
                                    print(loaded_message)
                                    break
                    except Exception as parse_err:
                        print(f"[*] Error parsing {label}model metadata {f_name}: {parse_err}")
    except Exception as cache_err:
        print(f"[*] {label}Model cache lookup failed: {cache_err}")
    return pretrained_loaded


def save_pinn_model(
    model: torch.nn.Module,
    prefix: str,
    label: str,
    design_name: Optional[str],
    gov_eq: str,
    bcs: List[str],
    params: Dict[str, float],
    final_loss: float,
    loss_history: List[float],
    domain: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Saves the trained PINN model weights (.pth) and metadata JSON for caching.
    label is "" for the 1D and "2D " for the 2D log messages.
    """
    try:
        cache_dir = _models_dir()
        os.makedirs(cache_dir, exist_ok=True)
        design_name = design_name or "unknown_design"
        slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save weights
        model_path = os.path.join(cache_dir, f"{prefix}{timestamp}_{slug}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[*] Saved trained {label}PINN model weights to: {model_path}")

        # Save metadata JSON for caching
        meta_path = os.path.join(cache_dir, f"{prefix}{timestamp}_{slug}.json")
        meta_data = {
            "governing_equation": gov_eq,
            "boundary_conditions": bcs,
            "params": params,
        }
        if domain is not None:
            meta_data["domain_min"] = domain[0]
            meta_data["domain_max"] = domain[1]
        meta_data["final_loss"] = final_loss
        meta_data["loss_history"] = loss_history
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta_data, mf, indent=4)
        print(f"[*] Saved {label}PINN model metadata to: {meta_path}")
    except Exception as save_err:
        print(f"[*] Failed to save {label}PINN model: {save_err}")
