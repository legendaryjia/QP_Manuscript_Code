"""Weighted multi-task training with AdamW, cosine annealing, and early stopping."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .config import CONFIG, TARGETS, TASK_WEIGHTS


def weighted_multitask_mse(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    weighted_losses = [
        TASK_WEIGHTS[name] * nn.functional.mse_loss(predictions[name], targets[name])
        for name in TARGETS
    ]
    return torch.stack(weighted_losses).sum() / sum(TASK_WEIGHTS.values())


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple:
    image = batch["image"].to(device)
    route = batch["route_index"].to(device)
    process = batch["process_numeric"].to(device)
    targets = {name: value.to(device) for name, value in batch["targets"].items()}
    return image, route, process, targets


def fit_model(model, train_loader, validation_loader, device: torch.device):
    """Train for at most 200 epochs and restore the best validation checkpoint."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG.max_epochs,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(CONFIG.max_epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            image, route, process, targets = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(image, route, process)
            loss = weighted_multitask_mse(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                image, route, process, targets = _move_batch(batch, device)
                predictions = model(image, route, process)
                validation_losses.append(weighted_multitask_mse(predictions, targets).item())
        validation_loss = sum(validation_losses) / len(validation_losses)
        scheduler.step()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CONFIG.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return model

