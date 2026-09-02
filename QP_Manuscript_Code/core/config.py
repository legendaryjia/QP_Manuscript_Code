"""Method constants reported in the manuscript."""

from __future__ import annotations

from dataclasses import dataclass


ROUTES = ("HR-AC", "HR-CC", "HR-CT", "HR-QC")
ROUTE_TO_INDEX = {route: index for index, route in enumerate(ROUTES)}

PROCESS_FEATURES = ("T_IA_C", "t_IA_s", "T_Q_C")

PHASE_CHEMISTRY_TARGETS = (
    "V_F_vol_pct",
    "V_RA_vol_pct",
    "V_TM_vol_pct",
    "V_FM_vol_pct",
    "V_B_vol_pct",
    "C_RA_wt_pct",
)
MECHANICAL_TARGETS = ("YS_MPa", "UTS_MPa", "TEL_pct")
TARGETS = PHASE_CHEMISTRY_TARGETS + MECHANICAL_TARGETS


@dataclass(frozen=True)
class MethodConfig:
    seed: int = 42
    original_fields_per_route: int = 10
    augmented_patches_per_route: int = 360
    image_size_px: int = 224
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    route_embedding_dim: int = 32
    process_feature_dim: int = 128
    common_embedding_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.10
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 8
    max_epochs: int = 200
    early_stopping_patience: int = 30
    partition_temperature_c: int = 400
    partition_time_s: int = 50


CONFIG = MethodConfig()

# Relative weights used by the weighted multi-task MSE. They follow the local
# training configuration while retaining only the nine outputs in the paper.
TASK_WEIGHTS = {
    "V_F_vol_pct": 1.00,
    "V_RA_vol_pct": 1.10,
    "V_TM_vol_pct": 1.00,
    "V_FM_vol_pct": 1.00,
    "V_B_vol_pct": 0.75,
    "C_RA_wt_pct": 1.10,
    "YS_MPa": 1.00,
    "UTS_MPa": 1.00,
    "TEL_pct": 1.00,
}

