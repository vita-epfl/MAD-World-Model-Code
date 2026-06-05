# Inference

MAD-LTX inference is config-driven. Each config selects the base LTX model, the released LoRA checkpoint, the conditioning mode, and a small set of example inputs.

## Installation

Use the tested environment in [../Dockerfile](../Dockerfile), or pull the published image:

```bash
docker pull ahparyald/mad-ltx
```

For a native install, use the same package set listed in the Dockerfile. Then expose the local source tree:

```bash
export PYTHONPATH=$PWD/src:$PWD
```

## Inputs

All modes take a prompt and write one MP4 per input item. The generic keys are:

- `rgb_image`: first RGB frame.
- `control_image`: first pose, segmentation, or HD-map frame.
- `control_video`: pose, segmentation, or HD-map video.
- `reference_video`: temporal conditioning video, such as ego motion or bbox control.

You can provide inputs in YAML under `inputs.items`, or override a single item from the CLI.

## Preparing Control Inputs

The bundled assets under `examples/inference/` include two samples for the skeleton, segmentation, HD-map, and ego-motion modes. The bbox-control configs are templates because bbox-control videos are not included in the example assets. For new OpenDV samples, use the pose files and rendering utilities released with [AhmadRH/OpenDV_Poses](https://huggingface.co/datasets/AhmadRH/OpenDV_Poses). For other datasets, you can extract compatible car/lane/human poses with external OpenPifPaf and DWPose pipelines, then render them in the same style.

For two-stage generation, you can either infer the intermediate control video yourself with `rgb-pose`, `rgb-hdmap`, or `rgb-seg`, or use the provided example control videos to test the second-stage `pose-rgb`, `hdmap-rgb`, and `seg-rgb` checkpoints directly.

### Ego-Motion Videos

The retained preprocessing utility in this repository renders the ego-motion video used by `rgb-pose-motion`. It expects per-timestep camera-to-world matrices and camera intrinsics:

```bash
PYTHONPATH=src:. python -m preprocessing.ego_motion \
  --c2w data/camera/example_c2w.npy \
  --intrinsics-npy data/camera/example_intrinsics.npy \
  --output data/controls/example_egomotion.mp4 \
  --fps 24
```

`--c2w` must point to a NumPy array with shape `[T, 4, 4]`. `--intrinsics-npy` can be a 3x3 intrinsics matrix, a dict containing `fx`, `fy`, `cx`, `cy`, `width`, and `height`, or a flat array `[fx, fy, cx, cy, width, height]`. The default `--coordinate-frame waymo` matches the released temporal-control LoRA training setup.

## Modes

Run the skeleton, segmentation, HD-map, or ego-motion configs without CLI input overrides to process their two bundled samples. For bbox-control, provide your own `--reference-video`.

### RGB To Control Video

Use `rgb-pose`, `rgb-seg`, or `rgb-hdmap` to animate a first control frame from an RGB first frame.

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/rgb_pose_2b.yaml
```

A single-sample override looks like:

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/rgb_pose_2b.yaml \
  --prompt "The image depicts a multi-lane urban road under an overpass with several vehicles, brick walls on both sides, and daytime light filtering through the structure. The ego vehicle continues with moderate city traffic ahead." \
  --rgb-image examples/inference/skeleton/rgb_1.jpg \
  --control-image examples/inference/skeleton/pose_1.jpg \
  --out outputs/rgb_pose
```

Use `configs/inference/rgb_seg_2b.yaml` or `configs/inference/rgb_hdmap_2b.yaml` for segmentation and HD-map controls.

### Control Video To RGB

Use a generated or provided control video to synthesize RGB frames:

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/pose_rgb_2b.yaml \
  --prompt "The image depicts a multi-lane urban road under an overpass with several vehicles, brick walls on both sides, and daytime light filtering through the structure. The ego vehicle continues with moderate city traffic ahead." \
  --rgb-image examples/inference/skeleton/rgb_1.jpg \
  --control-video examples/inference/skeleton/pose_1.mp4 \
  --out outputs/pose_rgb
```

For segmentation-to-RGB and HD-map-to-RGB:

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/seg_rgb_2b.yaml \
  --prompt "The image depicts a residential urban road with parked vehicles along the street, buildings and trees nearby, and daytime lighting. The ego vehicle moves through light city traffic." \
  --rgb-image examples/inference/segmentation/rgb_2.jpg \
  --control-video examples/inference/segmentation/seg_2.mp4 \
  --out outputs/seg_rgb

PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/hdmap_rgb_2b.yaml \
  --prompt "The image depicts a multi-lane urban road with a dark sedan ahead, another black sedan in the right lane, buildings, trees, crosswalks, parked vehicles, and a green traffic light suggesting forward motion." \
  --rgb-image examples/inference/hdmap/rgb_1.jpg \
  --control-video examples/inference/hdmap/hdmap_1.mp4 \
  --out outputs/hdmap_rgb
```

### RGB And Pose To Pose With Temporal Control

Use `rgb-pose-motion` for ego-motion control:

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/rgb_pose_motion_13b.yaml \
  --prompt "The image depicts a residential urban road. A number of parked vehicles are present in the road, one parked directly in front of the ego vehicle. The surrounding environment includes buildings and trees. The lighting suggests daytime." \
  --rgb-image examples/inference/ego_motion/rgb_2.jpg \
  --control-image examples/inference/ego_motion/pose_2.jpg \
  --reference-video examples/inference/ego_motion/egomotion_2.mp4 \
  --out outputs/rgb_pose_motion
```

Use `rgb-pose-bbox` for object bounding-box temporal control when you provide a bbox-control video:

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/rgb_pose_bbox_2b.yaml \
  --prompt "The image depicts an urban road with multiple vehicles ahead. The ego vehicle follows traffic while the object-motion control keeps the lead vehicle trajectory stable across the clip." \
  --rgb-image examples/inference/ego_motion/rgb_1.jpg \
  --control-image examples/inference/ego_motion/pose_1.jpg \
  --reference-video path/to/bbox_control.mp4 \
  --out outputs/rgb_pose_bbox
```

## Checkpoints

Configs default to the public Hugging Face repository:

```yaml
model:
  lora_repo: AhmadRH/MAD-LTX
```

To use a local LoRA instead:

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/rgb_pose_2b.yaml \
  --lora-path outputs/my_run/checkpoints/lora_weights_step_10000.safetensors
```

The current 2B public LoRAs used by the `_2b` configs are:

- `LTX_2B_r512_Lora_Pose_Forecaster.safetensors`
- `LTX_2B_r512_Lora_Segmentation_Forecaster.safetensors`
- `LTX_2B_r512_Lora_HDMap_Forecaster.safetensors`
- `LTX_2B_r512_Lora_Conditional_Forecaster.safetensors`
- `LTX_2B_r512_Lora_Synthesizer_Noised_Poses.safetensors`
- `LTX_2B_r512_Lora_Synthesizer_Segmentation.safetensors`
- `LTX_2B_r512_Lora_Synthesizer_HDMap.safetensors`

The current 13B public LoRAs used by the `_13b` configs are:

- `LTX_13B_r512_Lora_Pose_Forecaster.safetensors`
- `LTX_13B_r512_Lora_Conditional_Forecaster.safetensors`
- `LTX_13B_r512_Lora_Synthesizer_Noised_Poses.safetensors`
