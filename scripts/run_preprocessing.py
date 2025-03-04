#!/usr/bin/env python
"""
Script to preprocess data for GRACE downscaling.

This script processes raw GRACE, auxiliary, and groundwater data to create
datasets for training and evaluation of downscaling models.

Usage:
    python run_preprocessing.py --input_dir data/raw --output_dir data/processed
"""
import os
import sys
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from glob import glob
import rasterio
from tqdm import tqdm

# Add the src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
sys.path.append(str(src_dir))

# Import modules
from src.features.preprocessing import (
    load_raster, create_dataset_from_files, create_patches,
    normalize_data, create_training_data, save_dataset
)
from src.data.usgs import get_groundwater_grid

def setup_logging(log_file=None):
    """Set up logging configuration."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess data for GRACE downscaling')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory with raw data')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for processed data')
    parser.add_argument('--log_file', type=str,
                        help='Log file path')
    
    return parser.parse_args()

def get_file_lists(input_dir):
    """Get lists of files in the input directory."""
    # Initialize file lists
    grace_files = []
    auxiliary_files = {}
    static_files = {}
    
    # Create input paths
    input_dir = Path(input_dir)
    grace_dir = input_dir / 'grace'
    auxiliary_dir = input_dir / 'auxiliary'
    static_dir = input_dir / 'auxiliary'  # Static files are in the auxiliary directory
    
    # Get GRACE files
    grace_files = sorted([f.name for f in grace_dir.glob('GRACE_*.tif')])
    
    # Get auxiliary files
    aux_categories = ['SM', 'PRECIP', 'ET']
    
    for category in aux_categories:
        pattern = f"{category}_*.tif"
        files = sorted([f.name for f in auxiliary_dir.glob(pattern)])
        auxiliary_files[category.lower()] = files
    
    # Get static files
    static_patterns = {
        'elevation': 'static_elevation.tif',
        'slope': 'static_slope.tif',
        'aspect': 'static_aspect.tif',
        'water_occurrence': 'static_water_occurrence.tif'
    }
    
    for name, pattern in static_patterns.items():
        files = list(static_dir.glob(pattern))
        if files:
            static_files[name] = files[0].name
    
    return grace_files, auxiliary_files, static_files, grace_dir, auxiliary_dir, static_dir

def load_groundwater_data(input_dir):
    """Load groundwater data from input directory."""
    input_dir = Path(input_dir)
    groundwater_dir = input_dir / 'groundwater'
    
    # Load groundwater data
    gw_file = groundwater_dir / 'groundwater_data.csv'
    
    if gw_file.exists():
        gw_data = pd.read_csv(gw_file)
        
        # Convert date to datetime
        gw_data['date'] = pd.to_datetime(gw_data['date'])
        
        return gw_data
    else:
        logging.warning(f"Groundwater data file not found: {gw_file}")
        return None

def create_groundwater_targets(gw_data, grace_dates, lats, lons):
    """Create groundwater target grids for each GRACE date."""
    if gw_data is None:
        logging.warning("No groundwater data available")
        return None
    
    # Initialize target array
    targets = np.zeros((len(grace_dates), len(lats), len(lons)))
    
    # Create GeoDataFrame
    from src.data.usgs import convert_to_geodataframe
    gw_gdf = convert_to_geodataframe(gw_data)
    
    if gw_gdf is None:
        logging.warning("Failed to create GeoDataFrame from groundwater data")
        return None
    
    # Create grids for each date
    logging.info("Creating groundwater grids...")
    for i, date in enumerate(tqdm(grace_dates)):
        grid = get_groundwater_grid(gw_gdf, date, resolution=0.01)
        
        if grid is not None:
            # Convert xarray to numpy and add to targets
            targets[i] = grid.values
    
    return targets

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_file)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.input_dir:
        raw_dir = args.input_dir
    else:
        raw_dir = config['paths']['raw_dir']
    
    if args.output_dir:
        processed_dir = args.output_dir
    else:
        processed_dir = config['paths']['processed_dir']
    
    # Get file lists
    logging.info("Getting file lists...")
    grace_files, auxiliary_files, static_files, grace_dir, auxiliary_dir, static_dir = get_file_lists(raw_dir)
    
    # Check if files are available
    if not grace_files:
        logging.error("No GRACE files found")
        return
    
    # Create dataset
    logging.info("Creating dataset from files...")
    X, dates, lats, lons = create_dataset_from_files(
        grace_files, auxiliary_files, static_files,
        grace_dir, auxiliary_dir, static_dir,
        target_resolution=config['data']['target_resolution']
    )
    
    # Load groundwater data
    logging.info("Loading groundwater data...")
    gw_data = load_groundwater_data(raw_dir)
    
    # Create groundwater targets
    logging.info("Creating groundwater targets...")
    y = create_groundwater_targets(gw_data, dates, lats, lons)
    
    if y is None:
        logging.warning("Using GRACE data as proxy targets for downscaling")
        # Use GRACE data as target (for super-resolution only)
        y = X[:, :, :, 0]  # First channel is GRACE
    
    # Create output directory
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    logging.info("Saving full dataset...")
    save_dataset(X, y, dates, lats, lons, processed_dir, 'full')
    
    # Create patches
    logging.info("Creating patches...")
    patch_size = config['preprocessing']['patches']['size']
    patch_stride = config['preprocessing']['patches']['stride']
    
    X_patches, patch_indices = create_patches(X, patch_size, patch_stride)
    y_patches = []
    
    for t, h, w in patch_indices:
        y_patch = y[t, h:h+patch_size, w:w+patch_size]
        y_patches.append(y_patch)
    
    y_patches = np.array(y_patches)
    
    # Normalize data
    logging.info("Normalizing data...")
    method = config['preprocessing']['normalization']['method']
    
    X_norm, X_scaler = normalize_data(X_patches, method=method)
    y_norm, y_scaler = normalize_data(y_patches, method=method)
    
    # Create training, validation, and test sets
    logging.info("Creating training, validation, and test sets...")
    test_ratio = config['preprocessing']['split']['test_ratio']
    val_ratio = config['preprocessing']['split']['validation_ratio']
    random_seed = config['preprocessing']['split']['random_seed']
    
    X_train, y_train, X_val, y_val, X_test, y_test = create_training_data(
        X_norm, y_norm, test_ratio, val_ratio, random_seed
    )
    
    # Save datasets
    logging.info("Saving datasets...")
    datasets_dir = processed_dir / 'datasets'
    datasets_dir.mkdir(exist_ok=True)
    
    np.save(str(datasets_dir / 'X_train.npy'), X_train)
    np.save(str(datasets_dir / 'y_train.npy'), y_train)
    np.save(str(datasets_dir / 'X_val.npy'), X_val)
    np.save(str(datasets_dir / 'y_val.npy'), y_val)
    np.save(str(datasets_dir / 'X_test.npy'), X_test)
    np.save(str(datasets_dir / 'y_test.npy'), y_test)
    
    # Save patch indices
    np.save(str(datasets_dir / 'patch_indices.npy'), patch_indices)
    
    # Save scalers
    import pickle
    with open(str(datasets_dir / 'X_scaler.pkl'), 'wb') as f:
        pickle.dump(X_scaler, f)
    
    with open(str(datasets_dir / 'y_scaler.pkl'), 'wb') as f:
        pickle.dump(y_scaler, f)
    
    # Save dataset metadata
    metadata = {
        'X_shape': X.shape,
        'y_shape': y.shape,
        'X_train_shape': X_train.shape,
        'y_train_shape': y_train.shape,
        'X_val_shape': X_val.shape,
        'y_val_shape': y_val.shape,
        'X_test_shape': X_test.shape,
        'y_test_shape': y_test.shape,
        'patch_size': patch_size,
        'patch_stride': patch_stride,
        'normalization_method': method,
        'date_range': [str(dates[0]), str(dates[-1])],
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    pd.DataFrame([metadata]).to_csv(str(datasets_dir / 'metadata.csv'), index=False)
    
    logging.info("Preprocessing complete.")
    logging.info(f"Datasets saved to {datasets_dir}")

if __name__ == '__main__':
    main()