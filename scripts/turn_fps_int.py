import torch
import pathlib
import argparse

def to_int(x, default=None):
    """Safely convert torch/np scalars or floats to plain int."""
    try:
        import numpy as np
        if x is None:
            return default
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(round(x))
        if isinstance(x, np.generic):
            return int(x.item())
        if torch.is_tensor(x):
            return int(x.item()) if x.ndim == 0 else int(x.flatten()[0].item())
        return int(x)
    except Exception:
        return default

def inspect_fps(folder: pathlib.Path):
    folder = pathlib.Path(folder)
    files = sorted(folder.glob("*.pt"))
    print(f"\nFound {len(files)} latent files in {folder}")
    for f in files:
        try:
            d = torch.load(f, map_location="cpu")
            fps = to_int(d.get("fps"))
            print(f"{f.name}: fps={fps}")
        except Exception as e:
            print(f"{f.name}: ERROR {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect fps field in .pt latent files from two folders")
    parser.add_argument("folder1", type=str, help="First folder path containing .pt files")
    parser.add_argument("folder2", type=str, help="Second folder path containing .pt files")
    args = parser.parse_args()

    inspect_fps(args.folder1)
    inspect_fps(args.folder2)

if __name__ == "__main__":
    main()