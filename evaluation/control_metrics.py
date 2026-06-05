"""Trajectory-based control-following metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def list_paired_trajectories(target_dir: Path, generated_dir: Path, pattern: str) -> list[tuple[Path, Path]]:
    target_by_stem = {path.stem: path for path in sorted(target_dir.glob(pattern))}
    generated_by_stem = {path.stem: path for path in sorted(generated_dir.glob(pattern))}
    stems = sorted(set(target_by_stem) & set(generated_by_stem))
    if not stems:
        raise ValueError(f"No paired trajectories found between {target_dir} and {generated_dir}")
    return [(target_by_stem[stem], generated_by_stem[stem]) for stem in stems]


def load_trajectory(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
        arr = arr[:, :3, 3]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected [T, D] trajectory or [T, 4, 4] poses in {path}, got {arr.shape}")
    return arr[:, :3] if arr.shape[1] >= 3 else arr[:, :2]


def resample_to_length(traj: np.ndarray, target_len: int) -> np.ndarray:
    if len(traj) == target_len:
        return traj
    old_x = np.linspace(0.0, 1.0, len(traj))
    new_x = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(new_x, old_x, traj[:, dim]) for dim in range(traj.shape[1])], axis=1)


def average_displacement_error(target: np.ndarray, generated: np.ndarray) -> float:
    n = min(len(target), len(generated))
    target = resample_to_length(target, n)
    generated = resample_to_length(generated, n)
    return float(np.linalg.norm(target - generated, axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MAD-LTX trajectory control metrics.")
    parser.add_argument("--target-traj-dir", type=Path, required=True)
    parser.add_argument("--generated-traj-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    pairs = list_paired_trajectories(args.target_traj_dir, args.generated_traj_dir, args.pattern)
    ades = []
    for target_path, generated_path in pairs:
        ades.append(average_displacement_error(load_trajectory(target_path), load_trajectory(generated_path)))
    metrics = {
        "num_pairs": len(pairs),
        "ade_mean": float(np.mean(ades)),
        "ade_std": float(np.std(ades)),
        "ade_median": float(np.median(ades)),
    }

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
