# Training

MAD-LTX training uses released pose and caption data. The OpenDV reproduction path is: prepare the RGB clips, render the released intermediate motion representations, cache LTX latents, then train the two LoRA models.

## Installation

Use the same environment for training and inference. The tested setup is in [../Dockerfile](../Dockerfile), and the published image is available as:

```bash
docker pull ahparyald/mad-ltx
```

For a native install, create a Python environment and install the packages listed in the Dockerfile.

## 1. Prepare OpenDV Clips And Poses

Download the raw videos from the OpenDV-YouTube dataset, then resize/extract the clips used for training at 1056x704 resolution and 24 fps. MAD-LTX uses 5-second clips, so each training sample contains 121 frames at 24 fps.

Download the released pose files and captions from [AhmadRH/OpenDV_Poses](https://huggingface.co/datasets/AhmadRH/OpenDV_Poses). That dataset contains the preprocessed pose data and utility code for loading the pose files and rendering the pose/control videos. This repository does not ship the OpenPifPaf/DWPose extraction and rasterization code.

For our OpenDV subset, we computed a car-count signal for every 5-second clip from the released pose files, filtered the candidate pool using that signal, kept roughly half of the dataset, and randomly sampled the required amount of data from the retained clips. To reproduce the same style of training set, apply the car-count filtering before rasterization, then rasterize only the selected clips and extract the matching 5-second RGB videos.

Create manifests that pair captions, RGB clips, and rendered pose videos. The latent-caching script accepts CSV, JSON, or JSONL. A JSONL item can look like:

```json
{"id":"opendv_000001","caption":"A daytime urban driving scene with moderate traffic.","rgb_path":"data/opendv/rgb/opendv_000001.mp4","pose_path":"data/opendv/pose/opendv_000001.mp4"}
```

For `rgb_to_pose.yaml`, the target video is the pose/control video and the reference video is the RGB clip. For `pose_to_rgb.yaml`, the target video is the RGB clip and the reference video is the pose/control video.

## 2. Cache Latents

Cache text embeddings plus target/reference VAE latents before training.

For RGB-to-pose forecasting:

```bash
PYTHONPATH=src python scripts/preprocess_dataset.py data/manifests/opendv_rgb_to_pose.jsonl \
  --caption-column caption \
  --video-column pose_path \
  --reference-column rgb_path \
  --resolution-buckets 768x512x121 \
  --resolution-buckets-reference 768x512x121 \
  --latents-dir-name pose_latents_24fps_768x512 \
  --reference-latents-dir-name rgb_latents_24fps_768x512 \
  --model-source LTXV_2B_0.9.6_DEV \
  --output-dir data/precomputed/pose_predictor
```

For pose-to-RGB synthesis:

```bash
PYTHONPATH=src python scripts/preprocess_dataset.py data/manifests/opendv_pose_to_rgb.jsonl \
  --caption-column caption \
  --video-column rgb_path \
  --reference-column pose_path \
  --resolution-buckets 1056x704x121 \
  --resolution-buckets-reference 768x512x121 \
  --latents-dir-name rgb_latents_24fps_1056x704 \
  --reference-latents-dir-name pose_latents_24fps_768x512 \
  --model-source LTXV_2B_0.9.6_DEV \
  --output-dir data/precomputed/pose_to_rgb
```

The directory names in these commands match the defaults in `configs/training/rgb_to_pose.yaml` and `configs/training/pose_to_rgb.yaml`.

## 3. Train LoRAs

Train the motion forecaster:

```bash
PYTHONPATH=src python scripts/train.py configs/training/rgb_to_pose.yaml
```

Train the RGB synthesizer:

```bash
PYTHONPATH=src python scripts/train.py configs/training/pose_to_rgb.yaml
```

Both configs target the 2B LTX model by default. You can change `model.model_source`, LoRA rank, resolution, and number of training iterations to match the 13B or longer-training settings reported in the paper.

## Controllable Generation Experiments

For controllable generation, we trained on Waymo data. Reproducing that experiment requires extracting Waymo poses, rendering pose controls together with ego-motion and object bounding-box controls, caching all corresponding latents, and then training the conditional forecaster. We are not releasing the preprocessed Waymo data or the full preprocessing code for that experiment at this time.

## Run A Trained Checkpoint

Use the inference entrypoint with a local LoRA path:

```bash
PYTHONPATH=src python -m ltxv_trainer.inference \
  --config configs/inference/rgb_pose_2b.yaml \
  --lora-path outputs/rgb_to_pose/checkpoints/lora_weights_step_20000.safetensors
```
