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
    """Get lists of files in the input directory.
    
    Parameters
    ----------
    input_dir : str
        Input directory with raw data
        
    Returns
    -------
    tuple
        (grace_files, auxiliary_files, static_files, grace_dir, auxiliary_dir, static_dir)
    """
    import logging
    logger = logging.getLogger(__name__)
    # Initialize file lists
    grace_files = []
    auxiliary_files = {}
    static_files = {}
    
    # Create input paths
    input_dir = Path(input_dir)
    grace_dir = input_dir / 'grace'
    auxiliary_dir = input_dir / 'auxiliary'
    static_dir = input_dir / 'auxiliary' / 'static'
    
    # Get GRACE files
    if grace_dir.exists():
        grace_files = sorted([f.name for f in grace_dir.glob('GRACE_*.tif')])
    
    # Define auxiliary categories
    aux_categories = [
        'soil_moisture', 'precipitation', 'evapotranspiration',
        'groundwater', 'tws', 'baseflow', 'rootzone', 'profile_moisture'
    ]
    
    # Define file prefixes for each category
    aux_prefixes = {
        'soil_moisture': 'SM_',
        'precipitation': 'PRECIP_',
        'evapotranspiration': 'ET_',
        'groundwater': 'GW_',
        'tws': 'TWS_',
        'baseflow': 'BF_',
        'rootzone': 'RZ_',
        'profile_moisture': 'PM_'
    }
    
    # Get auxiliary files for each category
    for category in aux_categories:
        category_dir = auxiliary_dir / category
        
        if category_dir.exists():
            prefix = aux_prefixes.get(category, f"{category.upper()}_")
            pattern = f"{prefix}*.tif"
            files = sorted([f.name for f in category_dir.glob(pattern)])
            
            if files:
                auxiliary_files[category] = files
                logger.info(f"Found {len(files)} {category} files")
            else:
                logger.warning(f"No {category} files found with pattern {pattern} in {category_dir}")
        else:
            logger.warning(f"Directory {category_dir} does not exist")
    
    # Get static files
    if static_dir.exists():
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
                logger.info(f"Found static {name} file: {files[0].name}")
            else:
                logger.warning(f"No static {name} file found with pattern {pattern}")
    else:
        logger.warning(f"Static directory {static_dir} does not exist")
    
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

# In scripts/run_preprocessing.py
# Find the create_groundwater_targets function and modify it:

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
    
    # Get target resolution from config
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        target_resolution = config['data']['target_resolution']
        logging.info(f"Using target resolution from config: {target_resolution}")
    except:
        target_resolution = 0.01  # Default value
        logging.warning(f"Could not load resolution from config, using default: {target_resolution}")
    
    # Create empty grid with correct dimensions as a template
    from src.data.usgs import create_empty_grid
    template_grid = create_empty_grid(
        min_lon=min(lons), 
        max_lon=max(lons), 
        min_lat=min(lats), 
        max_lat=max(lats), 
        resolution=target_resolution
    )
    
    if template_grid is None:
        logging.error("Failed to create template grid. Using GRACE data as proxy targets.")
        return None
    
    # Log grid dimensions
    logging.info(f"Template grid shape: {template_grid.shape}, Expected shape in targets: {len(lats)}x{len(lons)}")
    
    if template_grid.shape != (len(lats), len(lons)):
        logging.warning(f"Grid shape mismatch. Will attempt to resample groundwater grids.")
    
    for i, date in enumerate(tqdm(grace_dates)):
        try:
            # Create grid for this date with same dimensions as template
            grid = get_groundwater_grid(
                gw_gdf, date, 
                resolution=target_resolution,
                template_grid=template_grid
            )
            
            if grid is not None:
                if grid.shape == (len(lats), len(lons)):
                    targets[i] = grid.values
                else:
                    # If shapes still don't match, use resampling
                    import xarray as xr
                    from scipy.interpolate import griddata
                    
                    # Get grid coordinates
                    grid_lats = grid.coords['latitude'].values
                    grid_lons = grid.coords['longitude'].values
                    
                    # Create meshgrid for original points
                    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)
                    
                    # Create meshgrid for target points
                    target_lon, target_lat = np.meshgrid(lons, lats)
                    
                    # Interpolate
                    valid_data = ~np.isnan(grid.values)
                    if np.any(valid_data):
                        points = np.column_stack((grid_lat[valid_data].ravel(), grid_lon[valid_data].ravel()))
                        values = grid.values[valid_data].ravel()
                        
                        # Use griddata to interpolate to new grid
                        interpolated = griddata(
                            points, values, 
                            (target_lat, target_lon), 
                            method='linear', 
                            fill_value=np.nan
                        )
                        
                        targets[i] = interpolated
                        logging.info(f"Successfully resampled grid for date {date}")
                    else:
                        logging.warning(f"No valid data for date {date}")
            else:
                logging.warning(f"Could not create groundwater grid for date {date}")
        
        except Exception as e:
            logging.error(f"Error creating groundwater grid for date {date}: {e}")
    
    # Check for NaN values
    nan_count = np.isnan(targets).sum()
    if nan_count > 0:
        logging.warning(f"Target array contains {nan_count} NaN values out of {targets.size}")
        # Fill NaNs with 0
        targets = np.nan_to_num(targets, nan=0.0)
    
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