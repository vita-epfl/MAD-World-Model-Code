"""Extract camera trajectories from videos for control evaluation.

This entrypoint wraps MapAnything-style camera prediction when that optional
environment is installed. It keeps the public evaluation workflow separate from
cluster dispatch code; large runs should shard `--videos-dir` externally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.video_io import read_video_frames

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for trajectory extraction.")
    return torch


def extract_camera_trajectory(video_path: Path, model_id: str, device: str, max_frames: int | None) -> np.ndarray:
    torch_mod = _require_torch()
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise RuntimeError("Install the MapAnything evaluation environment before running trajectory extraction.") from exc

    frames, _ = read_video_frames(video_path, num_frames=max_frames)
    images = [torch_mod.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
    batch = torch_mod.stack(images, dim=0).to(device)

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    with torch_mod.no_grad():
        prediction = model(batch)

    if isinstance(prediction, dict):
        for key in ("camera_poses", "poses", "c2w", "extrinsics", "trajectory"):
            if key in prediction:
                return np.asarray(
                    prediction[key].detach().cpu() if torch_mod.is_tensor(prediction[key]) else prediction[key]
                )
    if torch_mod.is_tensor(prediction):
        return prediction.detach().cpu().numpy()
    raise RuntimeError(f"Could not find trajectory output in model prediction for {video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract trajectories from videos for MAD-LTX control evaluation.")
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--model-id", default="facebook/map-anything")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    torch_mod = _require_torch()
    device = "cuda" if args.device == "auto" and torch_mod.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in sorted(args.videos_dir.glob(args.pattern)):
        trajectory = extract_camera_trajectory(video_path, args.model_id, device, args.max_frames)
        output_path = args.output_dir / f"{video_path.stem}.npy"
        np.save(output_path, trajectory)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
