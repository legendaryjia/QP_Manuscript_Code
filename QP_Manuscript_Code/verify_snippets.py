"""Syntax and manuscript-constant checks that do not require research data."""

from __future__ import annotations

import ast
from pathlib import Path

from core.config import CONFIG, TARGETS


def main() -> None:
    root = Path(__file__).resolve().parent
    files = sorted((root / "core").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    assert CONFIG.seed == 42
    assert CONFIG.original_fields_per_route == 10
    assert CONFIG.augmented_patches_per_route == 360
    assert CONFIG.image_size_px == 224
    assert (CONFIG.train_ratio, CONFIG.validation_ratio, CONFIG.test_ratio) == (0.70, 0.15, 0.15)
    assert CONFIG.route_embedding_dim == 32
    assert CONFIG.process_feature_dim == 128
    assert CONFIG.common_embedding_dim == 256
    assert CONFIG.learning_rate == 1.0e-4
    assert CONFIG.weight_decay == 1.0e-4
    assert CONFIG.batch_size == 8
    assert CONFIG.max_epochs == 200
    assert CONFIG.early_stopping_patience == 30
    assert len(TARGETS) == 9
    print(f"[OK] Parsed {len(files)} core files")
    print("[OK] Manuscript constants and nine output targets verified")


if __name__ == "__main__":
    main()

