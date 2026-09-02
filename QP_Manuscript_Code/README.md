# Core Code Excerpts for the Multimodal Q&P Steel Manuscript

This package contains concise code excerpts corresponding to the Methods described in the manuscript **“Multimodal Learning-Guided Process Design and Experimental Validation of Quenching and Partitioning Steels.”**

## Method-to-code overview

| Manuscript method | Code excerpt |
|---|---|
| Route-stratified 70:15:15 split with seed 42 | `core/stratified_split.py` |
| Route-conditioned patch sampling and process normalization | `core/dataset.py` |
| ConvNeXt-Tiny image encoder; route/process encoder; Transformer cross-modal fusion; nine regression outputs | `core/model.py` |
| Weighted multi-task MSE; AdamW; cosine scheduling; early stopping | `core/training.py` |
| MAE, RMSE, and R² evaluation | `core/evaluation.py` |
| Process-only, simple-fusion, and full Transformer-fusion ablations | `core/ablation.py` |
| Output-normalized, group-level SHAP aggregation | `core/shap_group_analysis.py` |
| Route-specific process screening by predicted UTS–TEL product | `core/process_optimization.py` |

The constants used throughout the excerpts are centralized in `core/config.py`.

## Environment

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python verify_snippets.py
```

`verify_snippets.py` performs syntax and method-constant checks without requiring the experimental data or loading pretrained weights.

## Data-availability wording

`DATA_AVAILABILITY_STATEMENT.md` contains wording that accurately states that the data are not publicly deposited. The final statement should be approved by all authors and the journal before submission.
