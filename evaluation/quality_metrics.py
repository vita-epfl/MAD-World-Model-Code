"""Generic video quality metrics for MAD-LTX evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.video_io import read_video_frames

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for video quality metrics.")
    return torch


def list_paired_videos(real_dir: Path, generated_dir: Path, pattern: str) -> list[tuple[Path, Path]]:
    real_by_stem = {path.stem: path for path in sorted(real_dir.glob(pattern))}
    generated_by_stem = {path.stem: path for path in sorted(generated_dir.glob(pattern))}
    stems = sorted(set(real_by_stem) & set(generated_by_stem))
    if not stems:
        raise ValueError(f"No paired videos found between {real_dir} and {generated_dir} with pattern {pattern}")
    return [(real_by_stem[stem], generated_by_stem[stem]) for stem in stems]


def load_video_batch_hwc(paths: list[Path], num_frames: int | None, size: tuple[int, int] | None) -> torch.Tensor:
    torch_mod = _require_torch()
    import cv2

    videos = []
    for path in paths:
        frames, _ = read_video_frames(path, num_frames=num_frames)
        if num_frames is not None and len(frames) < num_frames:
            raise ValueError(f"{path} has {len(frames)} frames, expected at least {num_frames}")
        if size is not None:
            width, height = size
            frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
        arr = np.stack(frames[:num_frames], axis=0)
        videos.append(torch_mod.from_numpy(arr))
    return torch_mod.stack(videos, dim=0)


def load_video_batch_chw(paths: list[Path], num_frames: int | None, size: tuple[int, int] | None) -> torch.Tensor:
    videos = load_video_batch_hwc(paths, num_frames, size)
    return videos.permute(0, 1, 4, 2, 3).float() / 255.0


def compute_fvd(real_paths: list[Path], generated_paths: list[Path], num_frames: int, batch_size: int, device: str) -> float:
    from evaluation.fvd_utils import get_fvd_logits, frechet_distance, load_fvd_model

    torch_mod = _require_torch()
    i3d = load_fvd_model(device)
    real_logits = []
    generated_logits = []
    for start in range(0, len(real_paths), batch_size):
        real_batch = load_video_batch_hwc(real_paths[start : start + batch_size], num_frames, size=None).to(device)
        generated_batch = load_video_batch_hwc(generated_paths[start : start + batch_size], num_frames, size=None).to(device)
        with torch_mod.no_grad():
            real_logits.append(get_fvd_logits(real_batch, i3d=i3d, device=device).cpu())
            generated_logits.append(get_fvd_logits(generated_batch, i3d=i3d, device=device).cpu())
    real_logits_t = torch_mod.cat(real_logits, dim=0)
    generated_logits_t = torch_mod.cat(generated_logits, dim=0)
    return float(frechet_distance(generated_logits_t, real_logits_t))


def compute_fid_if_available(real_paths: list[Path], generated_paths: list[Path], batch_size: int, device: str) -> float | None:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        return None

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for start in range(0, len(real_paths), batch_size):
        real_videos = load_video_batch_chw(real_paths[start : start + batch_size], num_frames=None, size=(299, 299))
        generated_videos = load_video_batch_chw(generated_paths[start : start + batch_size], num_frames=None, size=(299, 299))
        real_frames = real_videos.flatten(0, 1).to(device)
        generated_frames = generated_videos.flatten(0, 1).to(device)
        fid.update(real_frames, real=True)
        fid.update(generated_frames, real=False)
    return float(fid.compute().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MAD-LTX video quality metrics.")
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    torch_mod = _require_torch()
    device = "cuda" if args.device == "auto" and torch_mod.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    pairs = list_paired_videos(args.real_dir, args.generated_dir, args.pattern)
    real_paths = [pair[0] for pair in pairs]
    generated_paths = [pair[1] for pair in pairs]
    metrics = {
        "num_pairs": len(pairs),
        "fvd": compute_fvd(real_paths, generated_paths, args.num_frames, args.batch_size, device),
    }
    if not args.skip_fid:
        fid = compute_fid_if_available(real_paths, generated_paths, args.batch_size, device)
        if fid is not None:
            metrics["fid"] = fid

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
