"""Ablation variants described in the manuscript."""

from __future__ import annotations

from .model import MultimodalQPPredictor


def build_ablation_models(pretrained: bool = True) -> dict[str, MultimodalQPPredictor]:
    """Use identical encoders/heads where applicable; change only image/fusion use."""
    return {
        "process_only": MultimodalQPPredictor("process_only", pretrained=False),
        "simple_fusion": MultimodalQPPredictor("concat", pretrained=pretrained),
        "full_multimodal": MultimodalQPPredictor("transformer", pretrained=pretrained),
    }

