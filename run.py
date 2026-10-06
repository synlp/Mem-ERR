import argparse
import subprocess
import sys
from pathlib import Path

from memerr.config import load_config, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    config = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    pool = output / "pool.pt"
    checkpoint = output / "checkpoint.pt"
    predictions = output / "predictions.json"
    commands = [
        ("segment.py", "--annotations", require(config, "data", "annotations"), "--image-root", require(config, "data", "image_root"), "--output-root", require(config, "data", "segmented_root"), "--threshold", str(require(config, "fallbacks", "segmentation_threshold"))),
        ("build_pool.py", "--config", args.config, "--output", str(pool)),
        ("train.py", "--config", args.config, "--pool", str(pool), "--epochs", str(args.epochs), "--output", str(checkpoint)),
        ("infer.py", "--config", args.config, "--pool", str(pool), "--checkpoint", str(checkpoint), "--split", "test", "--output", str(predictions)),
        ("evaluate.py", "--annotations", require(config, "data", "annotations"), "--predictions", str(predictions), "--split", "test", "--output", str(output / "metrics.json")),
    ]
    for script, *arguments in commands:
        subprocess.run([sys.executable, str(root / script), *arguments], check=True)


if __name__ == "__main__":
    main()
