"""ConvNeXt/process encoders and Transformer-based cross-modal fusion."""

from __future__ import annotations

import timm
import torch
from torch import nn

from .config import CONFIG, PROCESS_FEATURES, ROUTES, TARGETS


class InitialSEMEncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.projection = nn.Linear(self.backbone.num_features, CONFIG.common_embedding_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(image))


class ProcessEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.route_embedding = nn.Embedding(len(ROUTES), CONFIG.route_embedding_dim)
        input_dim = CONFIG.route_embedding_dim + len(PROCESS_FEATURES)
        self.network = nn.Sequential(
            nn.Linear(input_dim, CONFIG.process_feature_dim),
            nn.GELU(),
            nn.LayerNorm(CONFIG.process_feature_dim),
            nn.Dropout(CONFIG.dropout),
            nn.Linear(CONFIG.process_feature_dim, CONFIG.process_feature_dim),
        )

    def forward(self, route_index: torch.Tensor, process_numeric: torch.Tensor) -> torch.Tensor:
        route_feature = self.route_embedding(route_index)
        return self.network(torch.cat((route_feature, process_numeric), dim=-1))


class MultimodalQPPredictor(nn.Module):
    """Nine-output model with process-only, concatenation, or Transformer fusion."""

    def __init__(self, fusion: str = "transformer", pretrained: bool = True) -> None:
        super().__init__()
        if fusion not in {"process_only", "concat", "transformer"}:
            raise ValueError(f"Unsupported fusion variant: {fusion}")
        self.fusion = fusion
        self.image_encoder = None if fusion == "process_only" else InitialSEMEncoder(pretrained)
        self.process_encoder = ProcessEncoder()
        self.process_projection = nn.Linear(CONFIG.process_feature_dim, CONFIG.common_embedding_dim)

        if fusion == "concat":
            self.concat_projection = nn.Sequential(
                nn.Linear(2 * CONFIG.common_embedding_dim, CONFIG.common_embedding_dim),
                nn.GELU(),
                nn.LayerNorm(CONFIG.common_embedding_dim),
            )
        elif fusion == "transformer":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, CONFIG.common_embedding_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=CONFIG.common_embedding_dim,
                nhead=CONFIG.transformer_heads,
                dim_feedforward=4 * CONFIG.common_embedding_dim,
                dropout=CONFIG.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=CONFIG.transformer_layers,
                norm=nn.LayerNorm(CONFIG.common_embedding_dim),
            )

        self.heads = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(CONFIG.common_embedding_dim, 128),
                    nn.GELU(),
                    nn.Dropout(CONFIG.dropout),
                    nn.Linear(128, 1),
                )
                for target in TARGETS
            }
        )

    def forward(
        self,
        image: torch.Tensor | None,
        route_index: torch.Tensor,
        process_numeric: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        process_token = self.process_projection(self.process_encoder(route_index, process_numeric))
        if self.fusion == "process_only":
            fused = process_token
        else:
            image_token = self.image_encoder(image)
            if self.fusion == "concat":
                fused = self.concat_projection(torch.cat((image_token, process_token), dim=-1))
            else:
                cls = self.cls_token.expand(image_token.shape[0], -1, -1)
                tokens = torch.cat((cls, image_token[:, None, :], process_token[:, None, :]), dim=1)
                fused = self.transformer(tokens)[:, 0, :]
        return {name: head(fused).squeeze(-1) for name, head in self.heads.items()}

