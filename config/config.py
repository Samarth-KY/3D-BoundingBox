import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default directories
raw_data_dir = PROJECT_ROOT / "data" / "raw_data"