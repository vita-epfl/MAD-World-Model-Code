"""Training strategies for MAD-LTX conditioning modes."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel, computed_field
from torch import Tensor

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import get_rope_scale_factors, prepare_video_coordinates
from ltxv_trainer.timestep_samplers import TimestepSampler

DEFAULT_FPS = 24
REFERENCE_IMAGE_FRAME_COORD = 1000


@dataclass(frozen=True)
class LatentMetadata:
    num_frames: int
    height: int
    width: int
    fps: float


class TrainingBatch(BaseModel):
    """Prepared training tensors for one optimizer step."""

    latents: Tensor
    targets: Tensor
    prompt_embeds: Tensor
    prompt_attention_mask: Tensor
    timesteps: Tensor
    sigmas: Tensor
    conditioning_mask: Tensor
    num_frames: int
    height: int
    width: int
    fps: float
    rope_interpolation_scale: list[float]
    video_coords: Tensor | None = None

    @computed_field
    @property
    def batch_size(self) -> int:
        return self.latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        return self.latents.shape[1]

    model_config = {"arbitrary_types_allowed": True}


class TrainingStrategy(ABC):
    """Base class for conditioning-specific batch preparation."""

    def __init__(self, conditioning_config: ConditioningConfig):
        self.conditioning_config = conditioning_config

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Return dataset directories required by this strategy."""

    @abstractmethod
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare a dataloader batch for the transformer."""

    @abstractmethod
    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute the strategy-specific loss."""

    @staticmethod
    def prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        return {
            "hidden_states": batch.latents,
            "encoder_hidden_states": batch.prompt_embeds,
            "timestep": batch.timesteps,
            "encoder_attention_mask": batch.prompt_attention_mask,
            "num_frames": batch.num_frames,
            "height": batch.height,
            "width": batch.width,
            "rope_interpolation_scale": batch.rope_interpolation_scale,
            "video_coords": batch.video_coords,
            "return_dict": False,
        }

    def _data_sources(self, **extra_sources: str) -> dict[str, str]:
        sources = {
            self.conditioning_config.latents_dir: "latents",
            "conditions": "conditions",
        }
        sources.update(extra_sources)
        return sources

    def _create_timesteps_from_conditioning_mask(
        self,
        conditioning_mask: Tensor,
        sampled_timestep_values: Tensor,
    ) -> Tensor:
        expanded_timesteps = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)
        return torch.where(conditioning_mask, 0, expanded_timesteps)

    def _create_first_frame_conditioning_mask(
        self,
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
        if (
            self.conditioning_config.first_frame_conditioning_p > 0
            and random.random() < self.conditioning_config.first_frame_conditioning_p
        ):
            first_frame_end_idx = min(height * width, sequence_length)
            conditioning_mask[:, :first_frame_end_idx] = True
        return conditioning_mask

    def _latent_metadata(self, latents: dict[str, Tensor], label: str = "target") -> LatentMetadata:
        fps_tensor = latents.get("fps", None)
        if fps_tensor is not None and not torch.all(fps_tensor == fps_tensor[0]):
            logger.warning(
                "Different FPS values found in the %s batch. Found: %s, using the first one: %s",
                label,
                fps_tensor.tolist(),
                fps_tensor[0].item(),
            )
        if fps_tensor is None:
            logger.warning("FPS metadata not found in the %s batch. Using default FPS: %s", label, DEFAULT_FPS)

        return LatentMetadata(
            num_frames=int(latents["num_frames"][0].item()),
            height=int(latents["height"][0].item()),
            width=int(latents["width"][0].item()),
            fps=float(fps_tensor[0].item()) if fps_tensor is not None else DEFAULT_FPS,
        )

    @staticmethod
    def _prompt_conditioning(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        conditions = batch["conditions"]
        return conditions["prompt_embeds"], conditions["prompt_attention_mask"]

    def _sample_noisy_target(
        self,
        target_latents: Tensor,
        target_conditioning_mask: Tensor,
        timestep_sampler: TimestepSampler,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sigmas = timestep_sampler.sample_for(target_latents).view(-1, 1, 1)
        if noise is None:
            noise = torch.randn_like(target_latents, device=target_latents.device)

        noisy_target = (1 - sigmas) * target_latents + sigmas * noise
        noisy_target = torch.where(target_conditioning_mask.unsqueeze(-1), target_latents, noisy_target)
        targets = noise - target_latents
        sampled_timestep_values = torch.round(sigmas.flatten(start_dim=1)[:, 0] * 1000.0).long()
        return noisy_target, targets, sigmas, sampled_timestep_values

    @staticmethod
    def _masked_mse_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = (pred - target).pow(2)
        loss_mask = mask.unsqueeze(-1).float()
        return loss.mul(loss_mask).div(loss_mask.mean().clamp_min(1e-6)).mean()

    def _full_sequence_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        return self._masked_mse_loss(model_pred, batch.targets, ~batch.conditioning_mask)

    def _target_tail_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        target_seq_len = batch.targets.shape[1]
        target_pred = model_pred[:, -target_seq_len:]
        target_mask = ~batch.conditioning_mask[:, -target_seq_len:]
        return self._masked_mse_loss(target_pred, batch.targets, target_mask)

    @staticmethod
    def _video_coords(
        *,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        fps: float,
        frame_value: int | None = None,
    ) -> Tensor:
        raw_coords = prepare_video_coordinates(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            sequence_multiplier=1,
            device=device,
        )
        if frame_value is not None:
            raw_coords[..., 0] = frame_value

        rope_scale_factors = get_rope_scale_factors(fps)
        prescaled_f = raw_coords[..., 0] * rope_scale_factors[0]
        prescaled_h = raw_coords[..., 1] * rope_scale_factors[1]
        prescaled_w = raw_coords[..., 2] * rope_scale_factors[2]
        return torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)

    def _target_conditioning(
        self,
        target_latents: Tensor,
        meta: LatentMetadata,
    ) -> Tensor:
        return self._create_first_frame_conditioning_mask(
            batch_size=target_latents.shape[0],
            sequence_length=target_latents.shape[1],
            height=meta.height,
            width=meta.width,
            device=target_latents.device,
        )

    @staticmethod
    def _reference_frame_latents(ref_latent_info: dict[str, Tensor]) -> Tensor:
        ref_height = int(ref_latent_info["height"][0].item())
        ref_width = int(ref_latent_info["width"][0].item())
        return ref_latent_info["latents"][:, : ref_height * ref_width, :]

    def _reference_frame_coords(
        self,
        ref_latents: Tensor,
        ref_meta: LatentMetadata,
        target_meta: LatentMetadata,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        expected_target_frame_tokens = target_meta.height * target_meta.width
        if ref_latents.shape[1] == expected_target_frame_tokens:
            height, width = target_meta.height, target_meta.width
        else:
            height, width = ref_meta.height, ref_meta.width

        return self._video_coords(
            num_frames=1,
            height=height,
            width=width,
            batch_size=batch_size,
            device=device,
            fps=target_meta.fps,
            frame_value=REFERENCE_IMAGE_FRAME_COORD,
        )


class StandardTrainingStrategy(TrainingStrategy):
    """Standard I2V training with optional first-frame conditioning."""

    def get_data_sources(self) -> dict[str, str]:
        return self._data_sources()

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        latents = batch["latents"]
        target_latents = latents["latents"]
        meta = self._latent_metadata(latents)
        prompt_embeds, prompt_attention_mask = self._prompt_conditioning(batch)
        target_conditioning_mask = self._target_conditioning(target_latents, meta)
        noisy_target, targets, sigmas, sampled_timestep_values = self._sample_noisy_target(
            target_latents,
            target_conditioning_mask,
            timestep_sampler,
        )
        timesteps = self._create_timesteps_from_conditioning_mask(target_conditioning_mask, sampled_timestep_values)

        return TrainingBatch(
            latents=noisy_target,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=target_conditioning_mask,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            rope_interpolation_scale=get_rope_scale_factors(meta.fps),
            video_coords=None,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        return self._full_sequence_loss(model_pred, batch)


class PoseToRGBTrainingStrategy(StandardTrainingStrategy):
    """Direct pose-to-RGB strategy where reference latents replace Gaussian noise."""

    def get_data_sources(self) -> dict[str, str]:
        return self._data_sources(**{self.conditioning_config.reference_latents_dir: "ref_latents"})

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]
        if target_latents.shape != ref_latents.shape:
            raise ValueError(
                f"Target latents shape {target_latents.shape} must match reference latents shape {ref_latents.shape}"
            )

        meta = self._latent_metadata(latents)
        prompt_embeds, prompt_attention_mask = self._prompt_conditioning(batch)
        target_conditioning_mask = self._target_conditioning(target_latents, meta)
        noisy_target, targets, sigmas, sampled_timestep_values = self._sample_noisy_target(
            target_latents,
            target_conditioning_mask,
            timestep_sampler,
            noise=ref_latents,
        )
        timesteps = self._create_timesteps_from_conditioning_mask(target_conditioning_mask, sampled_timestep_values)

        return TrainingBatch(
            latents=noisy_target,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=target_conditioning_mask,
            num_frames=meta.num_frames,
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            rope_interpolation_scale=get_rope_scale_factors(meta.fps),
            video_coords=None,
        )


class ReferenceVideoTrainingStrategy(TrainingStrategy):
    """Condition on a full reference/control video before target tokens."""

    def get_data_sources(self) -> dict[str, str]:
        return self._data_sources(**{self.conditioning_config.reference_latents_dir: "ref_latents"})

    def fix_mask_length(self, mask: Tensor, seq_len: int) -> Tensor:
        """Adjust a boolean mask so each row has exactly ``seq_len`` true values."""
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        new_mask = mask.clone()
        for batch_idx in range(mask.shape[0]):
            true_indices = torch.nonzero(mask[batch_idx], as_tuple=False).squeeze(-1)
            false_indices = torch.nonzero(~mask[batch_idx], as_tuple=False).squeeze(-1)
            num_true = true_indices.numel()
            if num_true < seq_len and false_indices.numel() > 0:
                add_idx = false_indices[torch.randperm(false_indices.numel(), device=mask.device)[: seq_len - num_true]]
                new_mask[batch_idx, add_idx] = True
            elif num_true > seq_len:
                remove_idx = true_indices[torch.randperm(true_indices.numel(), device=mask.device)[: num_true - seq_len]]
                new_mask[batch_idx, remove_idx] = False
        return new_mask

    def _maybe_noise_reference_latents(self, ref_latents: Tensor, batch: dict[str, Any]) -> Tensor:
        if "ref_latents_mask" not in batch:
            return ref_latents

        ref_latents_mask = batch["ref_latents_mask"]["patch_masks"]
        noise_scale = torch.rand(1, device=ref_latents.device, dtype=ref_latents.dtype) * 0.3
        ref_noise = torch.randn_like(ref_latents, device=ref_latents.device)
        return torch.where(
            ref_latents_mask.unsqueeze(-1),
            ref_latents * (1 - noise_scale) + ref_noise * noise_scale,
            ref_latents,
        )

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        latents = batch["latents"]
        ref_latent_info = batch["ref_latents"]
        target_latents = latents["latents"]
        ref_latents = self._maybe_noise_reference_latents(ref_latent_info["latents"], batch)

        target_meta = self._latent_metadata(latents)
        ref_meta = self._latent_metadata(ref_latent_info, label="reference")
        prompt_embeds, prompt_attention_mask = self._prompt_conditioning(batch)

        target_conditioning_mask = self._target_conditioning(target_latents, target_meta)
        noisy_target, targets, sigmas, sampled_timestep_values = self._sample_noisy_target(
            target_latents,
            target_conditioning_mask,
            timestep_sampler,
        )

        batch_size = target_latents.shape[0]
        ref_conditioning_mask = torch.ones(
            batch_size,
            ref_latents.shape[1],
            dtype=torch.bool,
            device=target_latents.device,
        )
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)
        ref_video_coords = self._video_coords(
            num_frames=ref_meta.num_frames,
            height=ref_meta.height,
            width=ref_meta.width,
            batch_size=batch_size,
            device=target_latents.device,
            fps=ref_meta.fps,
        )
        target_video_coords = self._video_coords(
            num_frames=target_meta.num_frames,
            height=target_meta.height,
            width=target_meta.width,
            batch_size=batch_size,
            device=target_latents.device,
            fps=target_meta.fps,
        )
        video_coords = torch.cat([ref_video_coords, target_video_coords], dim=2)

        return TrainingBatch(
            latents=combined_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=target_meta.num_frames,
            height=target_meta.height,
            width=target_meta.width,
            fps=target_meta.fps,
            rope_interpolation_scale=get_rope_scale_factors(target_meta.fps),
            video_coords=video_coords,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        return self._target_tail_loss(model_pred, batch)


class ReferenceFrameTrainingStrategy(TrainingStrategy):
    """Condition on one reference image before target tokens."""

    def get_data_sources(self) -> dict[str, str]:
        return self._data_sources(**{self.conditioning_config.reference_latents_dir: "ref_latents"})

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        latents = batch["latents"]
        ref_latent_info = batch["ref_latents"]
        target_latents = latents["latents"]
        ref_latents = self._reference_frame_latents(ref_latent_info)

        target_meta = self._latent_metadata(latents)
        ref_meta = self._latent_metadata(ref_latent_info, label="reference image")
        prompt_embeds, prompt_attention_mask = self._prompt_conditioning(batch)

        target_conditioning_mask = self._target_conditioning(target_latents, target_meta)
        noisy_target, targets, sigmas, sampled_timestep_values = self._sample_noisy_target(
            target_latents,
            target_conditioning_mask,
            timestep_sampler,
        )

        batch_size = target_latents.shape[0]
        ref_conditioning_mask = torch.ones(
            batch_size,
            ref_latents.shape[1],
            dtype=torch.bool,
            device=target_latents.device,
        )
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

        ref_image_coords = self._reference_frame_coords(
            ref_latents,
            ref_meta,
            target_meta,
            batch_size,
            target_latents.device,
        )
        target_video_coords = self._video_coords(
            num_frames=target_meta.num_frames,
            height=target_meta.height,
            width=target_meta.width,
            batch_size=batch_size,
            device=target_latents.device,
            fps=target_meta.fps,
        )
        video_coords = torch.cat([ref_image_coords, target_video_coords], dim=2)

        return TrainingBatch(
            latents=combined_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=target_meta.num_frames,
            height=target_meta.height,
            width=target_meta.width,
            fps=target_meta.fps,
            rope_interpolation_scale=get_rope_scale_factors(target_meta.fps),
            video_coords=video_coords,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        return self._target_tail_loss(model_pred, batch)


class ReferenceFrameVideoTrainingStrategy(ReferenceFrameTrainingStrategy):
    """Condition on temporal controls, a reference image, and target tokens."""

    def get_data_sources(self) -> dict[str, str]:
        sources = self._data_sources(
            **{
                self.conditioning_config.reference_latents_dir: "ref_latents",
                self.conditioning_config.reference_latents_video_dir: "ref_latents_video",
            }
        )
        if self.conditioning_config.bbox_latents_dir:
            sources[self.conditioning_config.bbox_latents_dir] = "bbox_latents"
        return sources

    def _select_temporal_reference(self, batch: dict[str, Any]) -> tuple[str, dict[str, Tensor]] | None:
        motion_info = batch.get("ref_latents_video")
        bbox_info = batch.get("bbox_latents")
        if motion_info is None and bbox_info is None:
            return None
        if motion_info is not None and bbox_info is not None:
            use_motion = random.random() < self.conditioning_config.motion_video_conditioning_p
            return ("motion", motion_info) if use_motion else ("bbox", bbox_info)
        if motion_info is not None:
            return "motion", motion_info
        return "bbox", bbox_info

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        latents = batch["latents"]
        ref_latent_info = batch["ref_latents"]
        target_latents = latents["latents"]
        ref_latents = self._reference_frame_latents(ref_latent_info)

        target_meta = self._latent_metadata(latents)
        ref_meta = self._latent_metadata(ref_latent_info, label="reference image")
        prompt_embeds, prompt_attention_mask = self._prompt_conditioning(batch)

        target_conditioning_mask = self._target_conditioning(target_latents, target_meta)
        noisy_target, targets, sigmas, sampled_timestep_values = self._sample_noisy_target(
            target_latents,
            target_conditioning_mask,
            timestep_sampler,
        )

        batch_size = target_latents.shape[0]
        conditioning_parts = []
        latent_parts = []
        coord_parts = []

        selected_temporal = self._select_temporal_reference(batch)
        if selected_temporal is not None:
            temporal_name, temporal_info = selected_temporal
            temporal_latents = temporal_info["latents"]
            temporal_meta = self._latent_metadata(temporal_info, label=f"{temporal_name} reference video")
            latent_parts.append(temporal_latents)
            conditioning_parts.append(
                torch.ones(batch_size, temporal_latents.shape[1], dtype=torch.bool, device=target_latents.device)
            )
            coord_parts.append(
                self._video_coords(
                    num_frames=temporal_meta.num_frames,
                    height=temporal_meta.height,
                    width=temporal_meta.width,
                    batch_size=batch_size,
                    device=target_latents.device,
                    fps=temporal_meta.fps,
                )
            )

        latent_parts.append(ref_latents)
        conditioning_parts.append(torch.ones(batch_size, ref_latents.shape[1], dtype=torch.bool, device=target_latents.device))
        coord_parts.append(
            self._reference_frame_coords(ref_latents, ref_meta, target_meta, batch_size, target_latents.device)
        )

        latent_parts.append(noisy_target)
        conditioning_parts.append(target_conditioning_mask)
        coord_parts.append(
            self._video_coords(
                num_frames=target_meta.num_frames,
                height=target_meta.height,
                width=target_meta.width,
                batch_size=batch_size,
                device=target_latents.device,
                fps=target_meta.fps,
            )
        )

        conditioning_mask = torch.cat(conditioning_parts, dim=1)
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        return TrainingBatch(
            latents=torch.cat(latent_parts, dim=1),
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=target_meta.num_frames,
            height=target_meta.height,
            width=target_meta.width,
            fps=target_meta.fps,
            rope_interpolation_scale=get_rope_scale_factors(target_meta.fps),
            video_coords=torch.cat(coord_parts, dim=2),
        )


def get_training_strategy(conditioning_config: ConditioningConfig) -> TrainingStrategy:
    """Create the strategy matching ``conditioning.mode``."""
    strategies: dict[str, type[TrainingStrategy]] = {
        "none": StandardTrainingStrategy,
        "reference_video": ReferenceVideoTrainingStrategy,
        "reference_image": ReferenceFrameTrainingStrategy,
        "pose_to_rgb": PoseToRGBTrainingStrategy,
        "reference_image_video": ReferenceFrameVideoTrainingStrategy,
    }
    try:
        strategy = strategies[conditioning_config.mode](conditioning_config)
    except KeyError as exc:
        raise ValueError(f"Unknown conditioning mode: {conditioning_config.mode}") from exc

    logger.debug("Using %s", strategy.__class__.__name__)
    return strategy
