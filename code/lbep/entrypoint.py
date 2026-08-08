import argparse
import json
import logging
from pathlib import Path

import torch

from .models import EquitableUNet, RadiographDenseNet
from .settings import ExperimentSettings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="lbep")
    result.add_argument("command", choices=("inspect", "train", "classify"))
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path("outputs"))
    return result


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = ExperimentSettings.from_file(arguments.config)
    if arguments.command == "inspect":
        payload = {"effective_batch_size": settings.effective_batch_size, "epochs": settings.epochs, "world_size": settings.world_size}
        logging.info(json.dumps(payload, sort_keys=True))
        return
    model = EquitableUNet() if arguments.command == "train" else RadiographDenseNet()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    arguments.output.mkdir(parents=True, exist_ok=True)
    destination = arguments.output / "model_initialization.pt"
    torch.save({"model": model.state_dict(), "seed": settings.seed}, destination)
    logging.info("initialized %d parameters", parameter_count)


if __name__ == "__main__":
    main()

