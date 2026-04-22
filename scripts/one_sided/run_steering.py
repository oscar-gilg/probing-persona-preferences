"""Run a config-driven steering experiment.

Usage:
    python -m scripts.one_sided.run_steering configs/steering/<config>.yaml
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.steering.runner import run

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
run(parser.parse_args().config)
