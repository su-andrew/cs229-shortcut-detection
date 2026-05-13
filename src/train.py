"""Training entrypoint placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shortcut detection model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML experiment config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 229))

    output_dir = Path(config.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded config from {args.config}")
    print("Training code goes here.")
    print(f"Writing outputs to {output_dir}")


if __name__ == "__main__":
    main()
