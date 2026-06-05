"""Video processing utilities for LTX Video training and inference."""

from pathlib import Path  # noqa: I001

from fractions import Fraction
import torch
from torch import Tensor
import torchvision.transforms as T  # noqa: N812
#import torchvision.io


  # imageio-ffmpeg ill be used under the hood
from pathlib import Path
from my_video_reader import VideoReader
import numpy as np
import av  # PyAV
#



def read_video(video_path: str | Path, target_frames: int | None = None) -> tuple[Tensor, float]:
    """Load and sample frames from a video file using PyVideoReader.

    Returns:
        frames: [F, C, H, W] float32 in [0,1]
        fps:    float
    """
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    if target_frames is None:
        indices = list(range(total_frames))
    else:
        if total_frames < target_frames:
            raise ValueError(f"Video has {total_frames} frames, but {target_frames} required")
        indices = torch.linspace(0, total_frames - 1, target_frames).long()

    frames = vr.get_batch(indices).float().div(255.0)  # [F, H, W, C]
    frames = frames.permute(0, 3, 1, 2)  # → [F, C, H, W]

    return frames, fps


def resize_video(frames: Tensor, target_width: int, target_height: int) -> Tensor:
    """Resize video frames while maintaining aspect ratio.

    Args:
        frames: Video tensor with shape [F, C, H, W]
        target_width: Target width for resizing
        target_height: Target height for resizing

    Returns:
        Resized video tensor with shape [F, C, H', W'] where H' >= target_height and W' >= target_width
    """
    # Resize maintaining aspect ratio
    current_height, current_width = frames.shape[2:]
    aspect_ratio = current_width / current_height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # Width is relatively larger, resize based on height
        resize_height = target_height
        resize_width = int(resize_height * aspect_ratio)
    else:
        # Height is relatively larger, resize based on width
        resize_width = target_width
        resize_height = int(resize_width / aspect_ratio)

    frames = T.functional.resize(
        frames,
        size=[resize_height, resize_width],
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    )

    return frames


def crop_video(video: Tensor, target_width: int, target_height: int) -> Tensor:
    """Center crop video frames to target dimensions.

    Args:
        video: Video tensor with shape [F, C, H, W]
        target_width: Target width for cropping
        target_height: Target height for cropping

    Returns:
        Cropped video tensor with shape [F, C, target_height, target_width]
    """
    current_height, current_width = video.shape[2:]

    if current_height < target_height or current_width < target_width:
        raise ValueError(
            "Video dimensions are too small for the target dimensions: "
            f"{current_height}x{current_width} -> {target_height}x{target_width}"
        )

    # Center crop to target dimensions
    crop_top = (current_height - target_height) // 2
    crop_left = (current_width - target_width) // 2

    video = T.functional.crop(
        video,
        top=crop_top,
        left=crop_left,
        height=target_height,
        width=target_width,
    )

    return video




def save_video(video_tensor: torch.Tensor, output_path, fps: float = 24.0) -> None:
    """
    Save [F, C, H, W] tensor (float in 0..1 or uint8 in 0..255) to H.264 MP4 using PyAV.
    """
    output_path = str(output_path)

    # Normalize dtype/range -> uint8 [0..255], RGB
    if video_tensor.dtype.is_floating_point:
        if video_tensor.max() <= 1:
            video_tensor = video_tensor * 255.0
        video_tensor = video_tensor.clamp(0, 255)
    video_tensor = video_tensor.to(torch.uint8)
    if video_tensor.shape[1] in (1, 3):  # [F, C, H, W] -> [F, H, W, C]
        video_np = video_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    else:
        raise ValueError(f"Expected [F, C, H, W], got {tuple(video_tensor.shape)}")

    F, H, W, C = video_np.shape
    if C != 3:
        raise ValueError("Expected RGB (3 channels)")

    # H.264 expects even dimensions for yuv420p
    if H % 2 or W % 2:
        H_even = H - (H % 2)
        W_even = W - (W % 2)
        video_np = video_np[:, :H_even, :W_even, :]
        H, W = H_even, W_even

    container = av.open(output_path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"      # broad compatibility
        stream.options = {"preset": "fast", "crf": "18"}  # adjust quality/speed

        for frame_np in video_np:  # HWC uint8 RGB
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        # flush
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()