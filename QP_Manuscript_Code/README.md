# Core Code Excerpts for the Multimodal Q&P Steel Manuscript

This package contains concise code excerpts corresponding to the Methods described in the manuscript **“Multimodal Learning-Guided Process Design and Experimental Validation of Quenching and Partitioning Steels.”**

No experimental measurements, sample-level labels, trained weights, predictions, or train/validation/test identifiers are included. Four representative initial-state SEM images and one four-panel montage are supplied only to illustrate the four routes; they are not the complete image dataset used for model development. The excerpts are intended to disclose the central computational logic without releasing the proprietary research dataset.

## Method-to-code overview

| Manuscript method | Code excerpt |
|---|---|
| Ten original SEM fields per initial state; 360 patches per state; geometry-preserving augmentation; 224 × 224 input | `core/image_preprocessing.py` |
| Route-stratified 70:15:15 split with seed 42 | `core/stratified_split.py` |
| Route-conditioned patch sampling and process normalization | `core/dataset.py` |
| ConvNeXt-Tiny image encoder; route/process encoder; Transformer cross-modal fusion; nine regression outputs | `core/model.py` |
| Weighted multi-task MSE; AdamW; cosine scheduling; early stopping | `core/training.py` |
| MAE, RMSE, and R² evaluation | `core/evaluation.py` |
| Process-only, simple-fusion, and full Transformer-fusion ablations | `core/ablation.py` |
| Output-normalized, group-level SHAP aggregation | `core/shap_group_analysis.py` |
| Route-specific process screening by predicted UTS–TEL product | `core/process_optimization.py` |

The constants used throughout the excerpts are centralized in `core/config.py`.

## Representative SEM images

`representative_sem_images/` contains one selected image for each initial route (HR-AC, HR-CC, HR-CT, and HR-QC) plus a four-panel montage. The image pixels were copied from the existing research files without generative alteration, synthetic texture, denoising, or contrast manipulation during packaging. See `representative_sem_images/IMAGE_NOTES.md` for panel mapping, hashes, and permitted interpretation.

## Scope

These files are method-focused excerpts, not an end-to-end executable repository. Data loaders expect author-supplied tables and the complete route-labelled patch manifests. The representative images are insufficient for model training. No reported metric can be regenerated from this package alone because the underlying experimental data, full image pool, and trained weights are not included.

## Environment

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python verify_snippets.py
```

`verify_snippets.py` performs syntax and method-constant checks without requiring the experimental data or loading pretrained weights.

## Data-availability wording

`DATA_AVAILABILITY_STATEMENT.md` contains wording that accurately states that the data are not publicly deposited. The final statement should be approved by all authors and the journal before submission.
