#!/usr/bin/env python
"""
Script to generate downscaled groundwater grids from trained model.

This script loads a trained CNN model and applies it to create monthly
high-resolution groundwater grids.

Usage:
    python scripts/run_downscaling.py --model_dir models/unet_20250306_135146
"""
import os
import sys
import logging
import argparse
import yaml
import numpy as np
import tensorflow as tf
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import pickle
from tqdm import tqdm

# Add the src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
sys.path.append(str(src_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate downscaled groundwater grids from trained model')
    
    parser.add_argument('--model_dir', type=str, default='models/unet_20250306_135146',
                        help='Directory containing the trained model')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='data/output/downscaled_groundwater',
                        help='Output directory for downscaled grids')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str,
                        help='Start date (YYYY-MM-DD) for downscaling')
    parser.add_argument('--end_date', type=str,
                        help='End date (YYYY-MM-DD) for downscaling')
    
    return parser.parse_args()

def load_model_with_custom_objects(model_path):
    """Load model with custom objects to handle serialization issues."""
    try:
        # Define custom objects for loss functions
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.losses.MeanAbsoluteError()
        }
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        
        # If regular loading fails, try to load weights into a new model
        try:
            from src.models.cnn import unet_model
            
            # Create metadata file path
            model_dir = Path(model_path).parent
            metadata_path = model_dir / 'config.yaml'
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    model_config = yaml.safe_load(f)
                
                # Extract model parameters
                model_type = model_config.get('model', {}).get('type', 'unet')
                n_filters = model_config.get('model', {}).get(model_type, {}).get('n_filters', 64)
                dropout = model_config.get('model', {}).get(model_type, {}).get('dropout', 0.2)
            else:
                # Default values
                model_type = 'unet'
                n_filters = 64
                dropout = 0.2
            
            # Try to determine input shape from X_train
            try:
                processed_dir = Path('data/processed/datasets')
                X_sample = np.load(str(processed_dir / 'X_train.npy'))
                input_shape = X_sample.shape[1:]
            except:
                # Default shape if X_train not available
                input_shape = (64, 64, 13)
                logger.warning(f"Could not determine input shape, using default: {input_shape}")
            
            # Create new model
            if model_type == 'unet':
                model = unet_model(input_shape, n_classes=1, n_filters=n_filters, dropout=dropout)
            else:
                from src.models.cnn import cnn_super_resolution
                model = cnn_super_resolution(input_shape, scale_factor=4, n_filters=n_filters)
            
            # Compile model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Try loading just the weights
            model.load_weights(model_path)
            logger.info(f"Model weights loaded successfully")
            return model
        
        except Exception as nested_e:
            logger.error(f"Error creating new model and loading weights: {nested_e}")
            return None

def find_dataset_files(processed_dir):
    """Find dataset files in the processed directory."""
    processed_dir = Path(processed_dir)
    dataset_files = {}
    
    # Try to find full dataset first
    full_files = {
        'X': processed_dir / 'full_X.npy',
        'dates': processed_dir / 'full_dates.csv',
        'lats': processed_dir / 'full_lats.npy',
        'lons': processed_dir / 'full_lons.npy'
    }
    
    # Check if full files exist
    if all(file.exists() for file in full_files.values()):
        logger.info("Found full dataset files")
        return full_files
    
    # Otherwise try the datasets subdirectory
    datasets_dir = processed_dir / 'datasets'
    if datasets_dir.exists():
        logger.info("Looking in datasets subdirectory...")
        
        # Look for test datasets
        test_files = {
            'X': datasets_dir / 'X_test.npy',
            'y': datasets_dir / 'y_test.npy',
        }
        
        if all(file.exists() for file in test_files.values()):
            logger.info("Found test dataset files")
            # Need to create dates, lats, lons from metadata or config
            metadata_file = datasets_dir / 'metadata.csv'
            if metadata_file.exists():
                metadata = pd.read_csv(str(metadata_file))
                logger.info(f"Loaded metadata: {metadata.columns}")
            else:
                logger.warning("No metadata file found")
            
            return test_files
    
    # If no files found, log error
    logger.error("Could not find dataset files")
    return None

def load_scalers(processed_dir):
    """Load normalization scalers."""
    processed_dir = Path(processed_dir)
    
    # Try main processed directory first
    scaler_files = {
        'X_scaler': processed_dir / 'X_scaler.pkl',
        'y_scaler': processed_dir / 'y_scaler.pkl'
    }
    
    # Check if files exist
    if all(file.exists() for file in scaler_files.values()):
        logger.info("Found scalers in main processed directory")
    else:
        # Try datasets subdirectory
        datasets_dir = processed_dir / 'datasets'
        if datasets_dir.exists():
            scaler_files = {
                'X_scaler': datasets_dir / 'X_scaler.pkl',
                'y_scaler': datasets_dir / 'y_scaler.pkl'
            }
            
            if all(file.exists() for file in scaler_files.values()):
                logger.info("Found scalers in datasets subdirectory")
            else:
                logger.error("Could not find scaler files")
                return None
    
    # Load scalers
    try:
        with open(str(scaler_files['X_scaler']), 'rb') as f:
            X_scaler = pickle.load(f)
        
        with open(str(scaler_files['y_scaler']), 'rb') as f:
            y_scaler = pickle.load(f)
        
        logger.info("Scalers loaded successfully")
        return X_scaler, y_scaler
    
    except Exception as e:
        logger.error(f"Error loading scalers: {e}")
        return None, None

def create_downscaled_grids(model, X, dates, lats, lons, X_scaler, y_scaler, output_dir, start_date=None, end_date=None):
    """Create downscaled groundwater grids with improved boundary handling."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter dates if requested
    if start_date and end_date:
        mask = (dates >= start_date) & (dates <= end_date)
        X = X[mask]
        dates = dates[mask]
    
    # Create geo-transform for GeoTIFFs
    lat_resolution = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.01
    lon_resolution = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.01
    transform = rasterio.transform.from_origin(
        lons.min(), lats.max(), lon_resolution, lat_resolution
    )
    
    # Process parameters
    patch_size = 64
    stride = 16  # Increased overlap to reduce boundary effects
    
    # Load a test sample to verify denormalization
    try:
        from src.features.preprocessing import load_dataset
        X_test = np.load(str(Path('data/processed/datasets/X_test.npy')))
        y_test = np.load(str(Path('data/processed/datasets/y_test.npy')))
        
        # Make a test prediction to verify scaling
        sample_x = X_test[0:1]
        sample_y_true = y_test[0:1]
        sample_y_pred = model.predict(sample_x)
        
        # Print stats to help diagnose scaling issues
        logger.info(f"Test true values range: [{np.min(sample_y_true):.4f}, {np.max(sample_y_true):.4f}]")
        logger.info(f"Test pred values range: [{np.min(sample_y_pred):.4f}, {np.max(sample_y_pred):.4f}]")
    except:
        logger.warning("Could not load test samples to verify scaling")
    
    # Process each time step
    logger.info(f"Processing {len(dates)} time steps")
    
    for i, date in enumerate(tqdm(dates)):
        X_month = X[i:i+1]
        height, width = X_month.shape[1:3]
        
        # Initialize weighted output
        output_grid = np.zeros((height, width))
        weight_grid = np.zeros((height, width))
        
        # Create weight kernel for smoother blending (higher weight in center)
        y, x = np.mgrid[0:patch_size, 0:patch_size]
        center_y, center_x = patch_size // 2, patch_size // 2
        # Gaussian-like weighting, higher in center
        kernel = np.exp(-0.5 * ((x - center_x)**2 + (y - center_y)**2) / (patch_size/4)**2)
        
        # Process patches
        for h in range(0, height - patch_size + 1, stride):
            for w in range(0, width - patch_size + 1, stride):
                # Extract patch
                patch = X_month[:, h:h+patch_size, w:w+patch_size, :]
                
                # Skip patches with NaN
                if np.isnan(patch).any():
                    continue
                
                # Normalize and predict
                patch_flat = patch.reshape(-1, patch.shape[-1])
                patch_norm = X_scaler.transform(patch_flat).reshape(patch.shape)
                pred_norm = model.predict(patch_norm, verbose=0)
                
                # Handle denormalization carefully
                if pred_norm.ndim == 4:
                    # Model outputs 4D tensor [batch, height, width, channels]
                    pred_flat = pred_norm.reshape(-1, pred_norm.shape[-1])
                    try:
                        # Try direct inverse transform
                        pred = y_scaler.inverse_transform(pred_flat)
                        pred = pred.reshape(pred_norm.shape[0], pred_norm.shape[1], pred_norm.shape[2])
                    except:
                        # Fallback to manual scaling
                        pred = pred_norm[0, :, :, 0]  # Extract the 2D grid
                        # Apply scaling manually - adjust these values based on your data
                        scale_value = np.mean(y_scaler.scale_) if hasattr(y_scaler, 'scale_') else 1.0
                        mean_value = np.mean(y_scaler.mean_) if hasattr(y_scaler, 'mean_') else 0.0
                        pred = pred * scale_value + mean_value
                else:
                    # Model outputs 3D tensor [batch, height, width]
                    pred = pred_norm[0]
                
                # Apply weighted addition to output
                weighted_pred = pred * kernel
                output_grid[h:h+patch_size, w:w+patch_size] += weighted_pred
                weight_grid[h:h+patch_size, w:w+patch_size] += kernel
        
        # Normalize by weights
        mask = weight_grid > 0
        output_grid[mask] /= weight_grid[mask]
        
        # Apply light smoothing to reduce artifacts
        from scipy.ndimage import gaussian_filter
        output_grid = gaussian_filter(output_grid, sigma=0.5)
        
        # Save outputs
        date_str = pd.to_datetime(date).strftime('%Y%m%d')
        
        # Save as GeoTIFF
        output_path = output_dir / f"groundwater_{date_str}.tif"
        with rasterio.open(
            str(output_path), 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=output_grid.dtype,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=transform
        ) as dst:
            dst.write(output_grid, 1)
        
        # Create visualization with additional information
        plt.figure(figsize=(12, 10))
        
        # Main heatmap
        plt.subplot(111)
        im = plt.imshow(output_grid, cmap='viridis')
        plt.colorbar(im, label='Groundwater Value')
        
        # Add informative title
        value_range = f"Range: [{np.nanmin(output_grid):.2f}, {np.nanmax(output_grid):.2f}]"
        plt.title(f'Downscaled Groundwater: {pd.to_datetime(date).strftime("%Y-%m-%d")}\n{value_range}')
        
        # Add grid lines to help identify artifacts
        plt.grid(False)
        
        # Save visualization
        plt.savefig(str(output_dir / f"groundwater_{date_str}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log progress
        if (i+1) % 10 == 0 or i == 0:
            logger.info(f"Processed {i+1}/{len(dates)} time steps")
            logger.info(f"Value range: [{np.nanmin(output_grid):.4f}, {np.nanmax(output_grid):.4f}]")
    
    logger.info(f"Downscaled groundwater data saved to {output_dir}")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration if needed
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set paths
    model_dir = Path(args.model_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    
    # Load model
    model_path = model_dir / 'best_model.h5'
    model = load_model_with_custom_objects(str(model_path))
    
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Find and load dataset files
    dataset_files = find_dataset_files(processed_dir)
    
    if dataset_files is None:
        logger.error("Failed to find dataset files. Exiting.")
        return
    
    # Load scalers
    X_scaler, y_scaler = load_scalers(processed_dir)
    
    if X_scaler is None or y_scaler is None:
        logger.error("Failed to load scalers. Exiting.")
        return
    
    # Load data
    try:
        # Check which type of files we have
        if 'X' in dataset_files and dataset_files['X'].exists():
            logger.info(f"Loading input data from {dataset_files['X']}")
            X = np.load(str(dataset_files['X']))
            
            # Load dates if available, otherwise create dummy dates
            if 'dates' in dataset_files and dataset_files['dates'].exists():
                dates_df = pd.read_csv(str(dataset_files['dates']))
                dates = pd.to_datetime(dates_df['date'])
            else:
                # Create monthly dates covering the data range
                logger.warning("No dates file found, creating dummy dates")
                dates = pd.date_range(start='2003-01-01', periods=X.shape[0], freq='M')
            
            # Load lat/lon if available, otherwise create dummy grid
            if 'lats' in dataset_files and dataset_files['lats'].exists() and 'lons' in dataset_files and dataset_files['lons'].exists():
                lats = np.load(str(dataset_files['lats']))
                lons = np.load(str(dataset_files['lons']))
            else:
                # Create dummy grid
                logger.warning("No lat/lon files found, creating dummy grid")
                height, width = X.shape[1:3]
                lats = np.linspace(35, 31, height)  # Mississippi approximate latitude range
                lons = np.linspace(-91, -88, width)  # Mississippi approximate longitude range
            
            logger.info(f"Loaded dataset with {len(dates)} time steps")
            
            # Create downscaled grids
            create_downscaled_grids(
                model, X, dates, lats, lons, X_scaler, y_scaler, output_dir,
                args.start_date, args.end_date
            )
        
        else:
            logger.error("Could not load input data")
            return
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("Processing complete")

if __name__ == '__main__':
    main()