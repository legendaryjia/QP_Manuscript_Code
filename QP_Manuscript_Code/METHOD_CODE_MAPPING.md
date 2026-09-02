# Method–Code Mapping

The excerpts implement the following manuscript statements directly.

1. **Initial SEM representation.** Four initial routes are supported: HR-AC, HR-CC, HR-CT, and HR-QC. Each route is expected to contain 10 original fields of view. `generate_route_patch_pool` creates 360 patches per route using random square cropping, horizontal/vertical flipping, and right-angle rotation, followed by resizing to 224 × 224 pixels.
2. **Specimen-level separation.** `make_route_stratified_split` separates unique specimens into training, validation, and testing subsets in a 70:15:15 ratio while stratifying on the initial route and using random seed 42.
3. **Image encoder.** `InitialSEMEncoder` uses a pretrained ConvNeXt-Tiny backbone and projects its pooled representation to 256 dimensions.
4. **Process encoder.** `ProcessEncoder` combines a 32-dimensional learnable route embedding with normalized TIA, tIA, and TQ values, producing a 128-dimensional process feature.
5. **Cross-modal fusion.** `MultimodalQPPredictor` projects image and process features to a shared 256-dimensional space and passes a learnable CLS token, image token, and process token through a Transformer encoder.
6. **Outputs.** Nine task-specific heads predict five phase fractions, retained-austenite carbon content, YS, UTS, and TEL.
7. **Training.** `fit_model` uses AdamW with learning rate 1.0×10⁻⁴, weight decay 1.0×10⁻⁴, batch size 8, at most 200 epochs, cosine annealing, and early stopping with patience 30.
8. **Evaluation.** `regression_metrics` returns MAE, RMSE, and R².
9. **Ablation.** `build_ablation_models` creates process-only, simple-concatenation, and full Transformer-fusion variants while leaving encoders and output heads unchanged where applicable.
10. **SHAP aggregation.** `aggregate_group_level_shap` combines route and SEM attribution into one `Initial SEM` group, normalizes mean absolute SHAP values within each output, and then averages within the phase/chemistry and mechanical-property output groups.
11. **Process recommendation.** `screen_route_specific_conditions` evaluates only conditions inside the supplied experimental domain and ranks each route by predicted UTS × TEL / 1000 in GPa·%.

Fixed partitioning conditions (400 °C, 50 s) are retained as metadata during process screening but are not informative variable inputs because they are constant for all specimens.

