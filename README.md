# GRACE Groundwater Downscaling

A Python-based framework for downscaling GRACE satellite terrestrial water storage anomalies to high-resolution groundwater estimates using Convolutional Neural Networks (CNN).

## Overview

This project aims to develop a CNN-based approach to downscale coarse resolution GRACE data (~300 km) to fine-resolution groundwater estimates, initially focusing on Mississippi as a test case. The method integrates multiple complementary datasets including:

- GRACE Terrestrial Water Storage Anomalies
- Soil Moisture (SMAP/SMOS)
- Precipitation (CHIRPS)
- Evapotranspiration (MODIS)
- Surface water (various sources)
- USGS groundwater well measurements (for validation)

## Repository Structure

- `config/`: Configuration files
- `data/`: Data storage (raw, processed, output)
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `scripts/`: Executable scripts
- `src/`: Source code modules for data processing, modeling, and evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/GRACE-Groundwater-Downscaling.git
cd GRACE-Groundwater-Downscaling

# Create a conda environment (recommended)
conda create -n grace-env python=3.9
conda activate grace-env

# Install dependencies
pip install -r requirements.txt

# Install this package in development mode
pip install -e .
```

## Google Earth Engine Authentication

This project uses Google Earth Engine for data access. You need to authenticate:

```bash
earthengine authenticate
```

## Usage

### 1. Data Extraction

```bash
python scripts/run_data_extraction.py --region mississippi --start_date 2002-04-01 --end_date 2023-01-01
```

### 2. Preprocessing

```bash
python scripts/run_preprocessing.py --input_dir data/raw --output_dir data/processed
```

### 3. Model Training

```bash
python scripts/run_training.py --config config/training_config.yaml
```

## Example Notebooks

- `01_data_exploration.ipynb`: Explore the GRACE and auxiliary datasets
- `02_model_development.ipynb`: CNN model development and training
- `03_results_visualization.ipynb`: Visualize and evaluate downscaling results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{grace_groundwater_downscaling,
  author = Saurav Bhattarai,
  title = {GRACE Groundwater Downscaling},
  url = {https://github.com/Saurav-JSU/GRACE-Groundwater-Downscaling},
  year = {2025},
}
```

## Acknowledgments

- NASA for providing GRACE and other Earth observation data
- USGS for groundwater monitoring data
- Google Earth Engine for data access and processing capabilities