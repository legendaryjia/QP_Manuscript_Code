"""Route-specific screening within the investigated Q&P processing domain."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import product

import pandas as pd

from .config import CONFIG, ROUTES


def strength_elongation_product(uts_mpa: float, tel_pct: float) -> float:
    """Return UTS × TEL in GPa·%."""
    return float(uts_mpa) * float(tel_pct) / 1000.0


def screen_route_specific_conditions(
    investigated_domain: Mapping[str, Iterable[float]],
    predict: Callable[[str, float, float, float], Mapping[str, float]],
) -> pd.DataFrame:
    """Rank candidate conditions without extrapolating beyond supplied levels.

    `predict(route, T_IA, t_IA, T_Q)` should average predictions across the
    route-specific initial-SEM patch pool and return at least `UTS_MPa` and
    `TEL_pct` in physical units.
    """
    records = []
    for route in ROUTES:
        combinations = product(
            investigated_domain["T_IA_C"],
            investigated_domain["t_IA_s"],
            investigated_domain["T_Q_C"],
        )
        for anneal_temperature, anneal_time, quench_temperature in combinations:
            outputs = dict(predict(route, anneal_temperature, anneal_time, quench_temperature))
            records.append(
                {
                    "route": route,
                    "T_IA_C": anneal_temperature,
                    "t_IA_s": anneal_time,
                    "T_Q_C": quench_temperature,
                    "T_P_C": CONFIG.partition_temperature_c,
                    "t_P_s": CONFIG.partition_time_s,
                    **outputs,
                    "PSE_GPa_pct": strength_elongation_product(
                        outputs["UTS_MPa"], outputs["TEL_pct"]
                    ),
                }
            )
    table = pd.DataFrame(records)
    return table.sort_values(["route", "PSE_GPa_pct"], ascending=[True, False]).reset_index(drop=True)

