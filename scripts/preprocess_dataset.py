#!/usr/bin/env python3

"""
Preprocess a video dataset by computing video clips latents and text captions embeddings.

This script provides a command-line interface for preprocessing video datasets by computing
latent representations of video clips and text embeddings of their captions. The preprocessed
data can be used to accelerate training of video generation models and to save GPU memory.

Basic usage:
    preprocess_dataset.py /path/to/dataset.json --resolution-buckets 768x768x49

The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
"""
import os
from pathlib import Path

import typer
from decode_latents import LatentsDecoder
from rich.console import Console

from ltxv_trainer import logger
from ltxv_trainer.model_loader import LtxvModelVersion
from scripts.process_captions import compute_captions_embeddings
from scripts.process_videos import compute_video_latents, parse_resolution_buckets, compute_video_latents_mask

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess a video dataset by computing video clips latents and text captions embeddings. "
    "The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.",
)


def preprocess_dataset(  # noqa: PLR0913
    dataset_file: str,
    caption_column: str,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    resolution_buckets_reference: list[tuple[int, int, int]] | None,
    batch_size: int,
    output_dir: str | None,
    id_token: str | None,
    vae_tiling: bool,
    decode_videos: bool,
    model_source: str,
    device: str,
    load_text_encoder_in_8bit: bool,
    remove_llm_prefixes: bool = False,
    reference_column: str | None = None,
    latents_dir_name: str = "latents",
    conditions_dir_name: str = "conditions",
    reference_latents_dir_name: str = "ref_latents",
    skip_captions: bool = False,
    skip_videos: bool = False,
) -> None:
    """Run the preprocessing pipeline with the given arguments"""
    # Validate dataset file
    _validate_dataset_file(dataset_file)

    # Set up output directories
    output_base = Path(output_dir) if output_dir else Path(dataset_file).parent / ".precomputed"
    conditions_dir = output_base / conditions_dir_name
    latents_dir = output_base / latents_dir_name

    if id_token:
        logger.info(f'LoRA trigger word "{id_token}" will be prepended to all captions')

    # Process captions using the dedicated function
    if not skip_captions:
        compute_captions_embeddings(
            dataset_file=dataset_file,
            output_dir=str(conditions_dir),
            caption_column=caption_column,
            media_column=video_column,
            id_token=id_token,
            remove_llm_prefixes=remove_llm_prefixes,
            batch_size=batch_size,
            device=device,
            load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        )

    # Process target videos using the dedicated function
    if not skip_videos:
        compute_video_latents(
            dataset_file=dataset_file,
            video_column=video_column,
            resolution_buckets=resolution_buckets,
            output_dir=str(latents_dir),
            model_source=model_source,
            batch_size=batch_size,
            device=device,
            vae_tiling=vae_tiling,
        )

    # Process reference videos if reference_column is provided
    if reference_column:
        logger.info("Processing reference videos for IC-LoRA training...")
        reference_latents_dir = output_base / reference_latents_dir_name
        reference_buckets = resolution_buckets_reference or resolution_buckets

        compute_video_latents(
            dataset_file=dataset_file,
            main_media_column=video_column,
            video_column=reference_column,
            resolution_buckets=reference_buckets,
            output_dir=str(reference_latents_dir),
            model_source=model_source,
            batch_size=batch_size,
            device=device,
            vae_tiling=vae_tiling,
        )

        # compute_video_latents_mask(
        #     dataset_file=dataset_file,
        #     main_media_column=video_column,
        #     video_column=reference_column,
        #     resolution_buckets=resolution_buckets_reference,
        #     output_dir=str(reference_latents_dir),
        #     model_source=model_source,
        #     batch_size=batch_size,
        #     device=device,
        #     vae_tiling=vae_tiling,
        # )

    # Handle video decoding if requested
    if decode_videos:
        logger.info("Decoding videos for verification...")

        decoder = LatentsDecoder(
            model_source=model_source,
            device=device,
            vae_tiling=vae_tiling,
        )
        # decoder.decode(latents_dir, output_base / "decoded_videos", num_videos=10)  # Decode only 10 videos for verification

        # Also decode reference videos if they exist
        if reference_column:
            reference_latents_dir = output_base / reference_latents_dir_name
            if reference_latents_dir.exists():
                logger.info("Decoding reference videos for verification...")
                decoder.decode(reference_latents_dir, output_base / "decoded_reference_videos", num_videos=10)

    # Print summary
    logger.info(f"Dataset preprocessing complete! Results saved to {output_base}")
    if reference_column:
        logger.info("Reference videos processed and saved to reference_latents/ directory for IC-LoRA training")


def _validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has the correct format"""
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")

    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")

    if dataset_file.suffix.lower() not in [".csv", ".json", ".jsonl"]:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    resolution_buckets_reference: str | None = typer.Option(
            default=None,
            help='Reference buckets in format "WxHxF;...". Defaults to --resolution-buckets.',
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name containing captions in the dataset JSON/JSONL/CSV file",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name containing video paths in the dataset JSON/JSONL/CSV file",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the T5 text encoder in 8-bit precision to save memory",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    model_source: str = typer.Option(
        default=str(LtxvModelVersion.latest()),
        help="Model source - can be a version string (e.g. 'LTXV_2B_0.9.5'), HF repo, or local path",
    ),
    id_token: str | None = typer.Option(
        default=None,
        help="Optional token to prepend to each caption (acts as a trigger word when training a LoRA)",
    ),
    decode_videos: bool = typer.Option(
        default=False,
        help="Decode and save videos after encoding (for verification purposes)",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove LLM prefixes from captions",
    ),
    reference_column: str | None = typer.Option(
        default=None,
        help="Column name containing reference video paths in the dataset JSON/JSONL/CSV file",
    ),
    latents_dir_name: str = typer.Option(
        default="latents",
        help="Directory name under output-dir for target video latents.",
    ),
    conditions_dir_name: str = typer.Option(
        default="conditions",
        help="Directory name under output-dir for text embeddings.",
    ),
    reference_latents_dir_name: str = typer.Option(
        default="ref_latents",
        help="Directory name under output-dir for reference/control latents.",
    ),
    skip_captions: bool = typer.Option(
        default=False,
        help="Skip text embedding computation.",
    ),
    skip_videos: bool = typer.Option(
        default=False,
        help="Skip target video latent computation.",
    ),
) -> None:
    """Preprocess a video dataset by computing and saving latents and text embeddings.

    The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.

    Examples:
        # Process a CSV dataset
        python preprocess_dataset.py dataset.csv --resolution-buckets 768x768x25

        # Process a JSON dataset with custom column names
        python preprocess_dataset.py dataset.json
            --resolution-buckets 768x768x25 --caption-column "text" --video-column "video_path"

        # Process dataset with reference videos for IC-LoRA training
        python preprocess_dataset.py dataset.json
            --resolution-buckets 768x768x25 --caption-column "caption"
            --video-column "media_path" --reference-column "reference_path"
    """
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)
    parsed_resolution_buckets_reference = (
        parse_resolution_buckets(resolution_buckets_reference) if resolution_buckets_reference else None
    )
    if len(parsed_resolution_buckets) > 1:
        raise typer.BadParameter(
            "Multiple resolution buckets are not yet supported. Please specify only one bucket.",
            param_hint="resolution-buckets",
        )

    preprocess_dataset(
        dataset_file=dataset_path,
        caption_column=caption_column,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        resolution_buckets_reference=parsed_resolution_buckets_reference,
        batch_size=batch_size,
        output_dir=output_dir,
        id_token=id_token,
        vae_tiling=vae_tiling,
        decode_videos=decode_videos,
        model_source=model_source,
        device=device,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        remove_llm_prefixes=remove_llm_prefixes,
        reference_column=reference_column,
        latents_dir_name=latents_dir_name,
        conditions_dir_name=conditions_dir_name,
        reference_latents_dir_name=reference_latents_dir_name,
        skip_captions=skip_captions,
        skip_videos=skip_videos,
    )


if __name__ == "__main__":
    app()
