import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default directories
raw_data_dir = PROJECT_ROOT / "data" / "raw_data"
valid_instances_json_dir = PROJECT_ROOT / "data" / "valid_instances.json"

model_checkpoint_dir = PROJECT_ROOT / "checkpoints"

results_dir = PROJECT_ROOT / "results"

onnx_export_dir = PROJECT_ROOT / "onnx_export"
