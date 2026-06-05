"""Render ego-motion conditioning videos from camera extrinsics.

The MAD-LTX ego-motion control is a visual encoding of camera movement. This
CLI expects per-timestep camera-to-world matrices and camera intrinsics, then
renders the colored sphere plus speed-line video used by the temporal-control
LoRA family.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    F = None


def _require_cv2():
    if cv2 is None:
        raise ImportError("opencv-python is required for ego-motion rendering.")
    return cv2


def _require_torch():
    if torch is None or F is None:
        raise ImportError("PyTorch is required for ego-motion rendering.")
    return torch, F



def write_video_ffmpeg(frames: Iterable[np.ndarray], output_path: str | Path, fps: int, width: int, height: int) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "faststart",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame.astype(np.uint8)).tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
    ret = proc.wait()
    if ret:
        raise RuntimeError(f"ffmpeg exited with code {ret}\n{stderr}")


def load_camera_poses(path: str | Path) -> np.ndarray:
    poses = np.asarray(np.load(path), dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera poses with shape [T, 4, 4], got {poses.shape}")
    return poses


def load_intrinsics(path: str | Path | None, width: int | None, height: int | None) -> tuple[np.ndarray, int, int]:
    if path is None:
        if width is None or height is None:
            raise ValueError("Pass --intrinsics-npy or both --width and --height.")
        fx = fy = max(width, height)
        cx = width / 2.0
        cy = height / 2.0
        return build_intrinsics_matrix(fx, fy, cx, cy), width, height

    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()

    if isinstance(raw, dict):
        fx = float(raw["fx"])
        fy = float(raw["fy"])
        cx = float(raw["cx"])
        cy = float(raw["cy"])
        out_width = int(raw.get("width", width or 0))
        out_height = int(raw.get("height", height or 0))
    else:
        arr = np.asarray(raw)
        if arr.shape == (3, 3):
            if width is None or height is None:
                raise ValueError("3x3 intrinsics require --width and --height.")
            return arr.astype(np.float32), int(width), int(height)
        flat = arr.reshape(-1)
        if flat.size < 6:
            raise ValueError("Intrinsics .npy must be 3x3, dict-like, or [fx, fy, cx, cy, width, height].")
        fx, fy, cx, cy = map(float, flat[:4])
        out_width = int(width or flat[4])
        out_height = int(height or flat[5])

    if not out_width or not out_height:
        raise ValueError("Could not determine output width/height from intrinsics; pass --width and --height.")
    return build_intrinsics_matrix(fx, fy, cx, cy), out_width, out_height


def build_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def relative_to_first_pose(poses_c2w: np.ndarray, coordinate_frame: str = "waymo") -> np.ndarray:
    inv_first = np.linalg.inv(poses_c2w[0])
    poses = np.asarray([inv_first @ pose for pose in poses_c2w], dtype=np.float32)
    if coordinate_frame == "waymo":
        rot_x_90 = np.array(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        poses = np.asarray([rot_x_90 @ pose for pose in poses], dtype=np.float32)
    elif coordinate_frame != "raw":
        raise ValueError("coordinate_frame must be 'waymo' or 'raw'")
    return poses


def create_random_color_grid(height: int, width: int, block_size: int = 32, seed: int = 42) -> torch.Tensor:
    torch, _ = _require_torch()
    rng = np.random.default_rng(seed)
    phi_steps = 2 * block_size
    theta_steps = block_size
    phi = np.linspace(-np.pi, np.pi, width)
    theta = np.linspace(0, np.pi, height)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    i_idx = ((phi_grid + np.pi) / (2 * np.pi) * phi_steps).astype(int)
    j_idx = (theta_grid / np.pi * theta_steps).astype(int)
    palette = rng.integers(0, 256, size=(phi_steps + 1, theta_steps + 1, 3), dtype=np.uint8)
    colored_grid = palette[i_idx, j_idx]
    colored_grid[(i_idx + j_idx) % 2 == 0] = 0
    return torch.from_numpy(colored_grid).permute(2, 0, 1).unsqueeze(0).float()


def calculate_finite_sphere_projection(
    poses_c2w: torch.Tensor,
    k: torch.Tensor,
    height: int,
    width: int,
    sphere_radius: float = 500.0,
    block_size: int = 32,
) -> np.ndarray:
    torch, F = _require_torch()
    device = poses_c2w.device
    texture_map = create_random_color_grid(height * 2, width * 2, block_size=block_size).to(device)

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    p_homogeneous = torch.stack([x, y, torch.ones_like(x)], dim=-1).view(-1, 3)
    ray_dirs_local = F.normalize((torch.inverse(k).to(device) @ p_homogeneous.T).T, p=2, dim=-1)

    projected_frames = []
    for pose in poses_c2w:
        cam_origin_world = pose[:3, 3]
        ray_dirs_world = (pose[:3, :3] @ ray_dirs_local.T).T

        b = 2.0 * torch.sum(cam_origin_world * ray_dirs_world, dim=1)
        c = torch.dot(cam_origin_world, cam_origin_world) - sphere_radius**2
        delta = b**2 - 4.0 * c
        t = (-b + torch.sqrt(torch.clamp(delta, min=1e-8))) / 2.0

        intersection_points = cam_origin_world.unsqueeze(0) + t.unsqueeze(-1) * ray_dirs_world
        x_w, y_w, z_w = intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2]
        phi = torch.atan2(y_w, x_w)
        theta = torch.acos(torch.clamp(z_w / sphere_radius, -1.0, 1.0))
        sampling_grid = torch.stack([phi / torch.pi, (2.0 * theta / torch.pi) - 1.0], dim=-1)
        sampling_grid = sampling_grid.view(1, height, width, 2)

        projected_frame = F.grid_sample(
            texture_map,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        projected_frames.append(projected_frame)

    all_frames = torch.cat(projected_frames, dim=0)
    return all_frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def calculate_synthetic_flow_for_particles(
    poses_c2w: torch.Tensor,
    k: torch.Tensor,
    height: int,
    width: int,
    depth: float = 15.0,
) -> np.ndarray:
    torch, _ = _require_torch()
    device = poses_c2w.device
    k_inv = torch.inverse(k).to(device)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    p_coords = torch.stack([x, y], dim=-1)
    p_homogeneous = torch.cat([p_coords, torch.ones(height, width, 1, device=device)], dim=-1).view(-1, 3)
    p_cam_i = depth * (k_inv @ p_homogeneous.T).T
    p_cam_i_homo = torch.cat([p_cam_i, torch.ones(height * width, 1, device=device)], dim=-1)

    all_flows = []
    for i in range(poses_c2w.shape[0] - 1):
        t_rel = torch.inverse(poses_c2w[i + 1]) @ poses_c2w[i]
        p_cam_next_homo = (t_rel @ p_cam_i_homo.T).T
        z_next = torch.clamp(p_cam_next_homo[:, 2].unsqueeze(1), min=1e-6)
        p_next_uv = (k @ p_cam_next_homo[:, :3].T).T[:, :2] / z_next
        all_flows.append((p_next_uv - p_coords.view(-1, 2)).view(height, width, 2))
    return torch.stack(all_flows, dim=0).cpu().numpy()


def generate_motion_visualization_with_speed_lines(
    poses_c2w: np.ndarray | torch.Tensor,
    k: np.ndarray | torch.Tensor,
    height: int,
    width: int,
    sphere_radius: float = 500.0,
    num_particles: int = 500,
    particle_depth: float = 15.0,
    line_color: tuple[int, int, int] = (255, 255, 200),
    line_thickness: int = 5,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> list[np.ndarray]:
    cv2_mod = _require_cv2()
    torch, _ = _require_torch()
    device = torch.device(device)
    poses_tensor = torch.as_tensor(poses_c2w, dtype=torch.float32, device=device)
    k_tensor = torch.as_tensor(k, dtype=torch.float32, device=device)
    background_frames = calculate_finite_sphere_projection(poses_tensor, k_tensor, height, width, sphere_radius)
    flow_fields = calculate_synthetic_flow_for_particles(poses_tensor, k_tensor, height, width, depth=particle_depth)

    rng = np.random.default_rng(seed)
    particles = rng.random((num_particles, 2)) * np.array([width, height])
    output_frames: list[np.ndarray] = []
    for i in range(len(poses_tensor) - 1):
        frame_canvas = background_frames[i].copy()
        px_int = particles.astype(int)
        valid_mask = (
            (px_int[:, 0] >= 0)
            & (px_int[:, 0] < width)
            & (px_int[:, 1] >= 0)
            & (px_int[:, 1] < height)
        )
        valid_particles = particles[valid_mask]
        valid_px_int = px_int[valid_mask]
        flow_vectors = flow_fields[i, valid_px_int[:, 1], valid_px_int[:, 0]]
        new_particles = valid_particles + flow_vectors

        for p_old, p_new in zip(valid_particles, new_particles):
            cv2_mod.line(
                frame_canvas,
                (int(p_old[0]), int(p_old[1])),
                (int(p_new[0]), int(p_new[1])),
                line_color,
                line_thickness,
                cv2_mod.LINE_AA,
            )

        particles[valid_mask] = new_particles
        out_of_bounds = (
            ~valid_mask
            | (particles[:, 0] < 0)
            | (particles[:, 0] >= width)
            | (particles[:, 1] < 0)
            | (particles[:, 1] >= height)
        )
        if np.any(out_of_bounds):
            particles[out_of_bounds] = rng.random((int(np.sum(out_of_bounds)), 2)) * np.array([width, height])
        output_frames.append(frame_canvas)

    if output_frames:
        output_frames.append(output_frames[-1].copy())
    return output_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MAD-LTX ego-motion conditioning from camera c2w matrices.")
    parser.add_argument("--c2w", type=Path, required=True, help="Numpy file containing [T, 4, 4] camera-to-world matrices.")
    parser.add_argument("--intrinsics-npy", type=Path, default=None, help="3x3 K or [fx, fy, cx, cy, width, height].")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--width", type=int, default=None, help="Output width; required when intrinsics do not store it.")
    parser.add_argument("--height", type=int, default=None, help="Output height; required when intrinsics do not store it.")
    parser.add_argument("--fps", type=int, default=24, help="Output frame rate.")
    parser.add_argument("--sphere-radius", type=float, default=500.0)
    parser.add_argument("--num-particles", type=int, default=500)
    parser.add_argument("--particle-depth", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--coordinate-frame",
        choices=["waymo", "raw"],
        default="waymo",
        help="Use 'waymo' to match the coordinate transform used for released MAD-LTX ego-motion LoRAs.",
    )
    args = parser.parse_args()

    torch_mod, _ = _require_torch()
    device = "cuda" if args.device == "auto" and torch_mod.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    poses = relative_to_first_pose(load_camera_poses(args.c2w), coordinate_frame=args.coordinate_frame)
    k, width, height = load_intrinsics(args.intrinsics_npy, args.width, args.height)
    frames = generate_motion_visualization_with_speed_lines(
        poses_c2w=poses,
        k=k,
        height=height,
        width=width,
        sphere_radius=args.sphere_radius,
        num_particles=args.num_particles,
        particle_depth=args.particle_depth,
        seed=args.seed,
        device=device,
    )
    write_video_ffmpeg(frames, args.output, args.fps, width, height)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
