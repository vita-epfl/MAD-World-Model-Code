"""Small video IO helpers for MAD-LTX evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - optional runtime dependency
    av = None


def read_video_frames(path: str | Path, start_frame: int = 0, num_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    if av is None:
        raise ImportError("PyAV is required to decode videos. Install it with `pip install av`.")

    video_path = str(path)
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        end_frame = None if num_frames is None else start_frame + num_frames
        frames: list[np.ndarray] = []
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx < start_frame:
                continue
            if end_frame is not None and frame_idx >= end_frame:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps
