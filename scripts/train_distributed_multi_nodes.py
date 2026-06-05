#!/usr/bin/env python
"""
Launch distributed training for LTXV models using Hugging Face Accelerate.

Works on:
- Single node, single/multi GPU
- Multi node (N nodes × G GPUs per node)

Examples
Node 0:
  python train_distributed.py configs/ltxv.yaml --num_processes 8 \
      --num_machines 2 --machine_rank 0 \
      --main_process_ip 10.0.0.1 --main_process_port 29500

Node 1:
  python train_distributed.py configs/ltxv.yaml --num_processes 8 \
      --num_machines 2 --machine_rank 1 \
      --main_process_ip 10.0.0.1 --main_process_port 29500
"""

import os
import subprocess
from pathlib import Path

import click
from accelerate.commands.launch import launch_command, launch_command_parser

from ltxv_trainer import logger


def _detect_local_gpus() -> int:
    try:
        gpu_list = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
        n = len([l for l in gpu_list.splitlines() if l.strip()])
        logger.debug(f"Found {n} GPUs:\n{gpu_list}")
        return max(n, 1)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to get GPU count from nvidia-smi: {e}")
        logger.error("Falling back to 1 GPU")
        return 1


@click.command(help="Launch distributed training for LTXV (single or multi node)")
@click.argument("config", type=click.Path(exists=True), required=True)
@click.option("--num_processes", type=int,
              help="Processes per node (usually GPUs per node). Defaults to local GPU count.")
@click.option("--num_machines", type=int, default=1, show_default=True,
              help="Total number of machines (nodes).")
@click.option("--machine_rank", type=int, default=0, show_default=True,
              help="Rank of this machine in [0..num_machines-1].")
@click.option("--main_process_ip", type=str, default=None,
              help="IP of rank-0 machine (required if num_machines>1 unless MASTER_ADDR is set).")
@click.option("--main_process_port", type=int, default=29500, show_default=True,
              help="Port on rank-0 machine for rendezvous.")
@click.option("--disable_progress_bars", is_flag=True, help="Disable progress bars during training")
def main(
    config: str,
    num_processes: int | None,
    num_machines: int,
    machine_rank: int,
    main_process_ip: str | None,
    main_process_port: int,
    disable_progress_bars: bool,
) -> None:
    # Resolve training script path
    script_dir = Path(__file__).parent
    training_script = str(script_dir / "train.py")

    # Default processes per node = local GPUs
    if num_processes is None:
        num_processes = _detect_local_gpus()

    # Build args to pass through to train.py
    training_args = [config]
    if disable_progress_bars:
        training_args.append("--disable_progress_bars")

    # Accelerate launch args
    launch_args: list[str] = []

    # Multi-GPU (even on 1 node)
    if num_processes > 1:
        launch_args.append("--multi_gpu")

    launch_args += ["--num_processes", str(num_processes)]

    # Multi-node wiring
    if num_machines > 1:
        # Allow env fallbacks from external launchers.
        master_addr = main_process_ip or os.environ.get("MASTER_ADDR")
        master_port = str(main_process_port or os.environ.get("MASTER_PORT", "29500"))

        if master_addr is None:
            raise click.UsageError(
                "For multi-node runs you must provide --main_process_ip on all nodes "
                "OR set MASTER_ADDR in the environment on every node (pointing to rank 0)."
            )

        launch_args += [
            "--num_machines", str(num_machines),
            "--machine_rank", str(machine_rank),
            "--main_process_ip", master_addr,
            "--main_process_port", master_port,
        ]

    # Append the training program + its args
    launch_args += [training_script, *training_args]

    # Parse & launch with Accelerate
    launch_parser = launch_command_parser()
    parsed = launch_parser.parse_args(launch_args)
    launch_command(parsed)


if __name__ == "__main__":
    main()
