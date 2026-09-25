# -*- coding: utf-8 -*-
"""Explorer profiles (descriptor axes + evaluator per kind of idea)."""
from typing import Any

from vectornaut.explorer.profiles.base import ExplorerProfile

PROFILE_NAMES = ("materials", "business")


def get_profile(name: str, **kwargs: Any) -> ExplorerProfile:
    if name == "materials":
        from vectornaut.explorer.profiles.materials import MaterialsProfile
        return MaterialsProfile(**kwargs)
    if name == "business":
        from vectornaut.explorer.profiles.business import BusinessProfile
        return BusinessProfile(**kwargs)
    raise ValueError(f"Unknown explorer profile '{name}' (available: {', '.join(PROFILE_NAMES)}).")
