"""Specimen-level 70:15:15 split stratified by initial route."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG


def make_route_stratified_split(
    specimens: pd.DataFrame,
    sample_column: str = "sample_id",
    route_column: str = "route",
) -> pd.DataFrame:
    """Return one split label per unique specimen using random seed 42."""
    if specimens[sample_column].duplicated().any():
        raise ValueError("Each row must represent one unique Q&P-treated specimen")

    train, remainder = train_test_split(
        specimens,
        test_size=CONFIG.validation_ratio + CONFIG.test_ratio,
        random_state=CONFIG.seed,
        stratify=specimens[route_column],
    )
    validation, test = train_test_split(
        remainder,
        test_size=CONFIG.test_ratio / (CONFIG.validation_ratio + CONFIG.test_ratio),
        random_state=CONFIG.seed,
        stratify=remainder[route_column],
    )

    labelled = []
    for frame, name in ((train, "train"), (validation, "validation"), (test, "test")):
        part = frame[[sample_column, route_column]].copy()
        part["split"] = name
        labelled.append(part)
    return pd.concat(labelled, ignore_index=True).sort_values(sample_column)

