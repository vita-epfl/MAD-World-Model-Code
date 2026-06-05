from pathlib import Path
import numpy as np
import torch
import av  # PyAV

class PyVideoReader:
    """
    Minimal video reader using PyAV (FFmpeg).
    Decodes to CPU, returns frames as torch.uint8 [F, H, W, C] (RGB).
    """
    def __init__(self, path: str | Path):
        self.path = str(path)
        container = av.open(self.path)
        vstream = container.streams.video[0]
        self._fps = float(vstream.average_rate) if vstream.average_rate else 0.0

        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        if not frames:
            raise RuntimeError(f"No frames decoded from {self.path}")

        arr = np.stack(frames, axis=0)           # [F, H, W, C] uint8
        self._frames = torch.from_numpy(arr)

    def __len__(self): return self._frames.shape[0]
    def __getitem__(self, idx): return self._frames[idx]
    def get_avg_fps(self): return self._fps
    def get_batch(self, indices):
        if isinstance(indices, torch.Tensor): indices = indices.tolist()
        return self._frames[indices]

# Optional alias if other code imports VideoReader
VideoReader = PyVideoReader