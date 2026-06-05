#!/usr/bin/env python
"""Generic config-driven MAD-LTX inference entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import PIL.Image
import PIL.ImageOps
import torch
import yaml
from diffusers.utils import export_to_video
from torch.amp import autocast
from torchvision.transforms.functional import to_tensor

from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.model_loader import load_ltxv_components
from ltxv_trainer.utils import open_image_as_srgb
from ltxv_trainer.video_utils import read_video

CONTROL_IMAGE_MODES = {"rgb-pose", "rgb-seg", "rgb-hdmap"}
TEMPORAL_CONTROL_MODES = {"rgb-pose-motion", "rgb-pose-bbox"}
SUPPORTED_MODES = CONTROL_IMAGE_MODES | TEMPORAL_CONTROL_MODES | {"pose-rgb", "rgb-rgb"}


@dataclass
class InferenceItem:
    prompt: str
    image: str | None = None
    reference_image: str | None = None
    reference_video: str | None = None
    reference_video_fps: int | None = None
    seed: int | None = None
    output_name: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fit_image(image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
    if image.size == (width, height):
        return image
    resampling = getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS
    return PIL.ImageOps.fit(image, (width, height), method=resampling, centering=(0.5, 0.5))


def _prepare_image(image_path: str | None, width: int, height: int) -> PIL.Image.Image | None:
    if not image_path:
        return None
    image = open_image_as_srgb(image_path)
    return _fit_image(image, width, height)


def _prepare_reference_image(image_path: str | None, width: int, height: int) -> torch.Tensor | None:
    image = _prepare_image(image_path, width, height)
    if image is None:
        return None
    return to_tensor(image).unsqueeze(0)


def _resize_video(video: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize/crop a [F, C, H, W] video tensor in [0, 1] to target size."""
    if video.shape[-2:] == (height, width):
        return video

    frames = []
    for frame in video:
        frame_np = (frame.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        frame_pil = PIL.Image.fromarray(frame_np)
        frames.append(to_tensor(_fit_image(frame_pil, width, height)))
    return torch.stack(frames, dim=0)


def _prepare_reference_video(
    reference_path: str | None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[torch.Tensor | None, int | None]:
    if not reference_path:
        return None, None
    video, fps = read_video(reference_path, target_frames=None)
    if width is None or height is None:
        return video, int(fps)
    return _resize_video(video, width, height), int(fps)


def _prepare_motion_reference_video(
    video: torch.Tensor | None,
    fps: int | None,
    width: int,
    height: int,
    num_frames: int,
) -> tuple[torch.Tensor | None, int | None]:
    if video is None:
        return None, fps

    video = _resize_video(video, width, height)
    expected_frames = 1 + 2 * video.shape[0] + 2
    if expected_frames != num_frames:
        raise ValueError(
            f"Motion reference preprocessing expects num_frames={expected_frames} "
            f"from {video.shape[0]} source frames, but inference requested {num_frames}."
        )

    expanded_video = torch.zeros(num_frames, *video.shape[1:], dtype=video.dtype, device=video.device)
    expanded_video[1:-2:2] = video
    expanded_video[2:-2:2] = video
    expanded_video[0] = 1.0
    expanded_video[-2:] = video[-1:].repeat(2, 1, 1, 1)

    return expanded_video, None if fps is None else fps * 2


def _load_lora_weights(pipe: LTXConditionPipeline, cfg: dict[str, Any], mode: str) -> None:
    model_cfg = cfg.get("model", {})

    lora_path = model_cfg.get("lora_path")
    lora_weight_name = model_cfg.get("lora_weight_name")
    mode_loras = model_cfg.get("mode_loras", {}) or {}
    mode_spec = mode_loras.get(mode, {}) or {}

    if lora_path:
        source = str(lora_path)
        source_path = Path(source).expanduser()
        if source_path.is_file():
            pipe.load_lora_weights(str(source_path.parent), weight_name=source_path.name)
        elif lora_weight_name:
            pipe.load_lora_weights(source, weight_name=str(lora_weight_name))
        else:
            pipe.load_lora_weights(str(source_path) if source_path.exists() else source)
        return

    repo_id = mode_spec.get("repo_id") or model_cfg.get("lora_repo")
    weight_name = mode_spec.get("weight_name")
    if repo_id:
        pipe.load_lora_weights(str(repo_id), weight_name=weight_name)


def _build_pipeline(cfg: dict[str, Any], mode: str) -> LTXConditionPipeline:
    model_cfg = cfg.get("model", {})
    components = load_ltxv_components(
        model_source=model_cfg.get("model_source"),
        load_text_encoder_in_8bit=bool(model_cfg.get("load_text_encoder_in_8bit", False)),
        transformer_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )

    pipe = LTXConditionPipeline(
        scheduler=components.scheduler,
        vae=components.vae,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        transformer=components.transformer,
    )

    _load_lora_weights(pipe, cfg, mode)

    if cfg.get("inference", {}).get("enable_cpu_offload", False):
        pipe.enable_model_cpu_offload()

    pipe.set_progress_bar_config(disable=True)
    return pipe


def _as_list(value: Any, length: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"Expected list of length {length}, got {len(value)}")
        return value
    return [value for _ in range(length)]


def _item_from_mode(mode: str, raw: dict[str, Any]) -> InferenceItem:
    prompt = raw.get("prompt") or raw.get("caption") or ""
    rgb_image = raw.get("rgb_image") or raw.get("rgb") or raw.get("first_frame")
    control_image = raw.get("control_image") or raw.get("pose_image") or raw.get("image")
    control_video = raw.get("control_video") or raw.get("pose_video")
    reference_video = raw.get("reference_video") or raw.get("motion_video") or raw.get("bbox_video")
    output_name = raw.get("output_name") or raw.get("id")

    if mode in CONTROL_IMAGE_MODES:
        image = control_image
        reference_image = raw.get("reference_image") or rgb_image
        reference_video = None
    elif mode == "pose-rgb":
        image = rgb_image or raw.get("image")
        reference_image = None
        reference_video = control_video or reference_video
    elif mode in TEMPORAL_CONTROL_MODES:
        image = control_image
        reference_image = raw.get("reference_image") or rgb_image
        reference_video = control_video or reference_video
    elif mode == "rgb-rgb":
        image = rgb_image or raw.get("image")
        reference_image = None
        reference_video = None
    else:
        raise ValueError(f"Unsupported inference mode: {mode}")

    return InferenceItem(
        prompt=prompt,
        image=image,
        reference_image=reference_image,
        reference_video=reference_video,
        reference_video_fps=raw.get("reference_video_fps"),
        seed=raw.get("seed"),
        output_name=output_name,
    )


def _normalize_inputs(cfg: dict[str, Any], mode: str) -> list[InferenceItem]:
    inputs = cfg.get("inputs", {}) or {}
    if isinstance(inputs, list):
        return [_item_from_mode(mode, item) for item in inputs]

    if "items" in inputs:
        items = inputs["items"] or []
        if not isinstance(items, list):
            raise ValueError("inputs.items must be a list")
        return [_item_from_mode(mode, item) for item in items]

    prompts = inputs.get("prompts", [])
    if not prompts:
        return []

    items = []
    images = _as_list(inputs.get("images"), len(prompts))
    reference_images = _as_list(inputs.get("reference_images"), len(prompts))
    reference_videos = _as_list(inputs.get("reference_videos"), len(prompts))
    output_names = _as_list(inputs.get("output_names"), len(prompts))
    for idx, prompt in enumerate(prompts):
        items.append(
            _item_from_mode(
                mode,
                {
                    "prompt": prompt,
                    "image": images[idx],
                    "reference_image": reference_images[idx],
                    "reference_video": reference_videos[idx],
                    "output_name": output_names[idx],
                },
            )
        )
    return items


def _apply_cli_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> None:
    inference_cfg = cfg.setdefault("inference", {})
    model_cfg = cfg.setdefault("model", {})

    for key in ("mode", "seed", "out_dir"):
        if overrides.get(key) is not None:
            inference_cfg[key] = overrides[key]
    if overrides.get("video_dims") is not None:
        inference_cfg["video_dims"] = overrides["video_dims"]
    if overrides.get("lora_path") is not None:
        model_cfg["lora_path"] = overrides["lora_path"]
    if overrides.get("lora_weight_name") is not None:
        model_cfg["lora_weight_name"] = overrides["lora_weight_name"]

    item_keys = {
        "prompt",
        "rgb_image",
        "control_image",
        "control_video",
        "reference_image",
        "reference_video",
        "output_name",
    }
    if any(overrides.get(key) is not None for key in item_keys):
        item = {key: overrides[key] for key in item_keys if overrides.get(key) is not None}
        cfg["inputs"] = {"items": [item]}


def run_inference(config_path: Path, overrides: dict[str, Any] | None = None) -> list[Path]:
    cfg = _load_yaml(config_path)
    _apply_cli_overrides(cfg, overrides or {})

    inf_cfg = cfg.setdefault("inference", {})
    mode = inf_cfg.get("mode", "rgb-pose")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported inference mode '{mode}'. Expected one of: {sorted(SUPPORTED_MODES)}")

    width, height, num_frames = [int(v) for v in inf_cfg.get("video_dims", [768, 512, 121])]
    steps = int(inf_cfg.get("steps", 50))
    guidance = float(inf_cfg.get("guidance_scale", 3.5))
    negative_prompt = inf_cfg.get("negative_prompt", "")
    seed = int(inf_cfg.get("seed", 42))
    fps = int(inf_cfg.get("fps", 24))
    out_dir = _ensure_out_dir(Path(inf_cfg.get("out_dir", "outputs/inference")))
    output_reference_comparison = bool(inf_cfg.get("output_reference_comparison", False))
    save_all = bool(inf_cfg.get("save_all_videos", False))

    items = _normalize_inputs(cfg, mode)
    if not items:
        raise ValueError("No inference inputs found. Add inputs.items to the config or pass CLI overrides.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tic = time.time()
    pipe = _build_pipeline(cfg, mode).to(device)
    print(f"Pipeline built in {time.time() - tic:.1f}s; running mode={mode} on {device}.")

    saved_paths: list[Path] = []
    for idx, item in enumerate(items):
        item_seed = seed if item.seed is None else int(item.seed)
        generator = torch.Generator(device=device).manual_seed(item_seed)

        tic = time.time()
        image = _prepare_image(item.image, width, height)
        reference_image = _prepare_reference_image(item.reference_image, width, height)
        reference_video, reference_video_fps = _prepare_reference_video(item.reference_video)
        if mode == "rgb-pose-motion":
            reference_video, reference_video_fps = _prepare_motion_reference_video(
                reference_video,
                reference_video_fps,
                width,
                height,
                num_frames,
            )
        if item.reference_video_fps is not None:
            reference_video_fps = int(item.reference_video_fps)

        call_kwargs: dict[str, Any] = {
            "prompt": item.prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "output_reference_comparison": output_reference_comparison,
            "frame_rate": fps,
            "image": image,
            "reference_image": reference_image,
            "reference_video": reference_video,
            "reference_video_fps": reference_video_fps or fps,
            "use_reference_video_as_noise": False,
        }
        print(f"Prepared sample {idx} in {time.time() - tic:.1f}s.")
        tic = time.time()
        with autocast(device_type=device.type, dtype=torch_dtype):
            result = pipe(**call_kwargs)
        print(f"Generated sample {idx} in {time.time() - tic:.1f}s.")

        output_stem = item.output_name or f"sample_{idx:05d}"
        for video_idx, video in enumerate(result.frames):
            if not save_all and video_idx > 0:
                break
            suffix = "" if not save_all else f"_{video_idx}"
            out_path = out_dir / f"{output_stem}{suffix}.mp4"
            export_to_video(video, str(out_path), fps=fps)
            saved_paths.append(out_path)

    return saved_paths


def _parse_json_overrides(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--overrides-json must point to a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="MAD-LTX inference.")
    parser.add_argument("--config", type=Path, required=True, help="Path to an inference YAML config.")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), default=None, help="Override inference mode.")
    parser.add_argument("--prompt", default=None, help="Prompt for a single CLI-provided sample.")
    parser.add_argument("--image", dest="control_image", default=None, help="Alias for --control-image.")
    parser.add_argument("--rgb-image", default=None, help="First RGB frame for the sample.")
    parser.add_argument("--control-image", default=None, help="First pose/seg/HD-map control frame.")
    parser.add_argument("--control-video", default=None, help="Pose/seg/HD-map control video.")
    parser.add_argument("--reference-image", default=None, help="Explicit reference image override.")
    parser.add_argument("--reference-video", default=None, help="Temporal conditioning video.")
    parser.add_argument("--output-name", default=None, help="Output stem for the generated MP4.")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--w", type=int, default=None, help="Width override.")
    parser.add_argument("--h", type=int, default=None, help="Height override.")
    parser.add_argument("--t", type=int, default=None, help="Frame-count override.")
    parser.add_argument("--lora-path", default=None, help="Local LoRA path or Hugging Face repo id.")
    parser.add_argument("--lora-weight-name", default=None, help="LoRA filename inside a Hugging Face repo.")
    parser.add_argument("--overrides-json", default=None, help="Optional JSON object merged into CLI overrides.")
    args = parser.parse_args()

    overrides = _parse_json_overrides(args.overrides_json)
    cli_overrides = {
        "mode": args.mode,
        "prompt": args.prompt,
        "rgb_image": args.rgb_image,
        "control_image": args.control_image,
        "control_video": args.control_video,
        "reference_image": args.reference_image,
        "reference_video": args.reference_video,
        "output_name": args.output_name,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "lora_path": args.lora_path,
        "lora_weight_name": args.lora_weight_name,
    }
    overrides.update({key: value for key, value in cli_overrides.items() if value is not None})
    if args.w is not None or args.h is not None or args.t is not None:
        cfg = _load_yaml(args.config)
        current = cfg.get("inference", {}).get("video_dims", [768, 512, 121])
        overrides["video_dims"] = [
            int(args.w if args.w is not None else current[0]),
            int(args.h if args.h is not None else current[1]),
            int(args.t if args.t is not None else current[2]),
        ]

    saved = run_inference(args.config, overrides)
    print("\nSaved:")
    for path in saved:
        print(f" - {path}")


if __name__ == "__main__":
    main()
