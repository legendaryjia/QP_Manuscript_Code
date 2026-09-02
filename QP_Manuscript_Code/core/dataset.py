"""Route-conditioned SEM patch sampling for specimen-level learning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .config import CONFIG, PROCESS_FEATURES, ROUTE_TO_INDEX, TARGETS


IMAGE_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class InitialSEMPatchDataset(Dataset):
    """Pair each specimen with a patch from its route-specific initial-SEM pool."""

    def __init__(
        self,
        specimens: pd.DataFrame,
        patch_manifest: pd.DataFrame,
        process_mean: np.ndarray,
        process_std: np.ndarray,
        training: bool,
    ) -> None:
        self.specimens = specimens.reset_index(drop=True)
        self.training = training
        self.epoch = 0
        self.process_mean = np.asarray(process_mean, dtype=np.float32)
        self.process_std = np.maximum(np.asarray(process_std, dtype=np.float32), 1.0e-8)
        self.patch_pool = {
            route: [Path(path) for path in group["patch_file"].tolist()]
            for route, group in patch_manifest.groupby("route", sort=False)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.specimens)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.specimens.iloc[index]
        route = row["route"]
        candidates = self.patch_pool[route]
        # Epoch-dependent sampling for training; fixed sampling for validation/test.
        local_seed = CONFIG.seed + index + (self.epoch * len(self)) if self.training else CONFIG.seed + index
        patch_path = candidates[int(np.random.default_rng(local_seed).integers(len(candidates)))]
        with Image.open(patch_path) as image:
            image_tensor = IMAGE_TRANSFORM(image.convert("RGB"))

        process = row[list(PROCESS_FEATURES)].to_numpy(dtype=np.float32)
        process = (process - self.process_mean) / self.process_std
        targets = {name: torch.tensor(float(row[name]), dtype=torch.float32) for name in TARGETS}
        return {
            "sample_id": row["sample_id"],
            "image": image_tensor,
            "route_index": torch.tensor(ROUTE_TO_INDEX[route], dtype=torch.long),
            "process_numeric": torch.from_numpy(process),
            "targets": targets,
        }

