"""
Config for the finetune experiments web UI.
Uses environment variables with defaults; can be overridden by CLI when starting the server.
"""
import os
import re
from pathlib import Path

# Project root (directory containing finetuning.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Experiments root: exp/<name>/result/<name>/ or exp/<name>/result/ (default: exp/)
EXPERIMENTS_ROOT = Path(
    os.environ.get("MXFONT_EXPERIMENTS_ROOT", str(PROJECT_ROOT / "exp"))
).resolve()
# Allowed base directories for dataset_path when starting a new run.
# Dataset path must be under one of these (or be a subpath of project root if PROJECT_ROOT is in the list).
ALLOWED_DATASET_BASES = [
    str(PROJECT_ROOT),
]

# Safe experiment name: alphanumeric, underscore, hyphen only
EXP_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
