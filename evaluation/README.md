# Evaluation

MAD-LTX reports two kinds of evaluation:

- video quality/diversity metrics on generated RGB videos;
- control-following metrics for ego-motion and object-motion controls.

The scripts in this folder are generic entrypoints. They operate on directories
or manifests and are intended to be sharded by the user on their own compute
cluster for large evaluation runs. Full video generation and evaluation over a
benchmark split is not practical on a single GPU.

## Video Quality

Compute FVD, and optionally FID when `torchmetrics` is installed:

```bash
PYTHONPATH=src:. python -m evaluation.quality_metrics \
  --real-dir data/eval/real_rgb \
  --generated-dir outputs/eval/generated_rgb \
  --output-json outputs/eval/quality_metrics.json
```

Files are paired by stem. Use `--pattern` when your videos use a different
extension or naming convention.

## Control Following

For trajectory controls, first extract trajectories from generated and target
videos with the same estimator, then compare them:

```bash
PYTHONPATH=src:. python -m evaluation.trajectory_extraction \
  --videos-dir outputs/eval/generated_rgb \
  --output-dir outputs/eval/generated_traj

PYTHONPATH=src:. python -m evaluation.control_metrics \
  --target-traj-dir data/eval/target_traj \
  --generated-traj-dir outputs/eval/generated_traj \
  --output-json outputs/eval/control_metrics.json
```

The trajectory extractor uses MapAnything-style camera prediction when that
environment is installed. The metric script itself only needs NumPy.
