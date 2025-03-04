"""
Path configuration for GRACE Groundwater Downscaling.
"""
from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Configuration directory
CONFIG_DIR = ROOT_DIR / 'config'

# Data directories
DATA_DIR = ROOT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = DATA_DIR / 'output'

# Data subdirectories
GRACE_DIR = RAW_DIR / 'grace'
AUXILIARY_DIR = RAW_DIR / 'auxiliary'
GROUNDWATER_DIR = RAW_DIR / 'groundwater'

# Model directory
MODEL_DIR = ROOT_DIR / 'models'

# Figures directory
FIGURES_DIR = ROOT_DIR / 'figures'

# Notebooks directory
NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'

# Create directories if they don't exist
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        OUTPUT_DIR,
        GRACE_DIR,
        AUXILIARY_DIR,
        GROUNDWATER_DIR,
        MODEL_DIR,
        FIGURES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)