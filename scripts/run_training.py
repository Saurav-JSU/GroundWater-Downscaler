#!/usr/bin/env python
"""
Script to train GRACE downscaling models.

This script trains CNN models for downscaling GRACE data to high-resolution
groundwater estimates.

Usage:
    python run_training.py --config config/config.yaml
"""
import os
import sys
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
sys.path.append(str(src_dir))

# Import modules
from src.models.cnn import (
    unet_model, cnn_super_resolution, compile_model,
    train_model, save_model
)
from src.models.evaluation import (
    evaluate_model, plot_prediction_vs_truth
)

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
    parser = argparse.ArgumentParser(description='Train GRACE downscaling models')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory with processed data')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for model')
    parser.add_argument('--model_type', type=str,
                        help='Model type (unet, super_resolution)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size')
    parser.add_argument('--log_file', type=str,
                        help='Log file path')
    
    return parser.parse_args()

def load_datasets(input_dir):
    """Load training, validation, and test datasets."""
    input_dir = Path(input_dir) / 'datasets'
    
    # Load datasets
    X_train = np.load(str(input_dir / 'X_train.npy'))
    y_train = np.load(str(input_dir / 'y_train.npy'))
    X_val = np.load(str(input_dir / 'X_val.npy'))
    y_val = np.load(str(input_dir / 'y_val.npy'))
    X_test = np.load(str(input_dir / 'X_test.npy'))
    y_test = np.load(str(input_dir / 'y_test.npy'))
    
    # Load patch indices
    patch_indices = np.load(str(input_dir / 'patch_indices.npy'))
    
    # Reshape targets to match model output format
    y_train = y_train[..., np.newaxis]
    y_val = y_val[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, patch_indices

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
        processed_dir = args.input_dir
    else:
        processed_dir = config['paths']['processed_dir']
    
    if args.output_dir:
        model_dir = args.output_dir
    else:
        model_dir = config['paths']['model_dir']
    
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = config['model']['type']
    
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = config['model']['training']['epochs']
    
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = config['model']['training']['batch_size']
    
    # Load datasets
    logging.info("Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, y_test, patch_indices = load_datasets(processed_dir)
    
    # Create model
    logging.info(f"Creating {model_type} model...")
    input_shape = X_train.shape[1:]
    
    if model_type == 'unet':
        n_filters = config['model']['unet']['n_filters']
        dropout = config['model']['unet']['dropout']
        
        model = unet_model(input_shape, n_classes=1, n_filters=n_filters, dropout=dropout)
    elif model_type == 'super_resolution':
        n_filters = config['model']['super_resolution']['n_filters']
        n_blocks = config['model']['super_resolution']['n_blocks']
        scale_factor = config['model']['super_resolution']['scale_factor']
        
        model = cnn_super_resolution(input_shape, scale_factor, n_filters, n_blocks)
    else:
        logging.error(f"Unknown model type: {model_type}")
        return
    
    # Compile model
    learning_rate = config['model']['training']['learning_rate']
    loss = config['model']['training']['loss']
    
    model = compile_model(model, lr=learning_rate, loss=loss)
    
    # Print model summary
    model.summary()
    
    # Create output directory
    model_output_dir = Path(model_dir) / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    logging.info("Training model...")
    patience = config['model']['training']['patience']
    
    trained_model, history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=batch_size, epochs=epochs, patience=patience,
        output_dir=model_output_dir
    )
    
    # Evaluate model
    logging.info("Evaluating model...")
    evaluation_dir = model_output_dir / 'evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    
    metrics = evaluate_model(trained_model, X_test, y_test, evaluation_dir)
    
    # Plot sample predictions
    logging.info("Plotting sample predictions...")
    n_samples = min(5, len(X_test))
    
    for i in range(n_samples):
        # Get sample
        x = X_test[i:i+1]
        y_true = y_test[i:i+1]
        
        # Make prediction
        y_pred = trained_model.predict(x)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot input (GRACE)
        im0 = axes[0].imshow(x[0, :, :, 0], cmap='viridis')
        axes[0].set_title('Input (GRACE)')
        plt.colorbar(im0, ax=axes[0])
        
        # Plot true
        im1 = axes[1].imshow(y_true[0, :, :, 0], cmap='viridis')
        axes[1].set_title('True')
        plt.colorbar(im1, ax=axes[1])
        
        # Plot prediction
        im2 = axes[2].imshow(y_pred[0, :, :, 0], cmap='viridis')
        axes[2].set_title('Prediction')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(str(evaluation_dir / f'sample_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save model
    logging.info("Saving model...")
    save_model(trained_model, model_output_dir, f"{model_type}_model")
    
    # Save configuration
    with open(str(model_output_dir / 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info("Training complete.")
    logging.info(f"Model saved to {model_output_dir}")

if __name__ == '__main__':
    main()