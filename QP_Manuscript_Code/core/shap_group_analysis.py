"""Group-level SHAP aggregation used for Fig. 7-style interpretation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .config import MECHANICAL_TARGETS, PHASE_CHEMISTRY_TARGETS, TARGETS


ATTRIBUTION_GROUPS = ("T_Q", "Initial SEM", "T_IA", "t_IA")


def combine_initial_state_attribution(route_shap, sem_feature_shap) -> np.ndarray:
    """Combine route and route-conditioned image attribution without double counting."""
    route = np.asarray(route_shap, dtype=float)
    sem = np.asarray(sem_feature_shap, dtype=float)
    if sem.ndim > 1:
        sem = sem.sum(axis=tuple(range(1, sem.ndim)))
    return route + sem


def aggregate_group_level_shap(
    shap_by_target: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Normalize mean |SHAP| within each output, then average by output group.

    Each target mapping must contain arrays for `T_Q`, `Initial SEM`, `T_IA`,
    and `t_IA`. The `Initial SEM` array should already combine route and image
    attribution through `combine_initial_state_attribution`.
    """
    normalized: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        mean_abs = {
            group: float(np.mean(np.abs(shap_by_target[target][group])))
            for group in ATTRIBUTION_GROUPS
        }
        denominator = sum(mean_abs.values())
        normalized[target] = {group: value / denominator for group, value in mean_abs.items()}

    def average(target_names) -> dict[str, float]:
        return {
            group: float(np.mean([normalized[target][group] for target in target_names]))
            for group in ATTRIBUTION_GROUPS
        }

    return {
        "phase_and_chemistry": average(PHASE_CHEMISTRY_TARGETS),
        "mechanical_properties": average(MECHANICAL_TARGETS),
    }

