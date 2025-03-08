"""
Functions for preprocessing data for GRACE downscaling.
"""
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import logging
from pathlib import Path
from datetime import datetime
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

def load_raster(file_path):
    """Load raster file into numpy array.
    
    Parameters
    ----------
    file_path : str
        Path to raster file
        
    Returns
    -------
    tuple
        (data, transform, crs, bounds)
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read the first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
        return data, transform, crs, bounds
    except Exception as e:
        logger.error(f"Error loading raster {file_path}: {e}")
        return None, None, None, None

def reproject_raster(data, src_transform, src_crs, 
                    dst_crs, dst_shape, dst_transform=None, 
                    resampling=Resampling.bilinear):
    """Reproject raster data to new projection.
    
    Parameters
    ----------
    data : numpy.ndarray
        Source raster data
    src_transform : affine.Affine
        Source affine transformation
    src_crs : rasterio.crs.CRS
        Source coordinate reference system
    dst_crs : rasterio.crs.CRS
        Destination coordinate reference system
    dst_shape : tuple
        Destination shape (height, width)
    dst_transform : affine.Affine, optional
        Destination affine transformation, by default None
    resampling : Resampling, optional
        Resampling method, by default Resampling.bilinear
        
    Returns
    -------
    tuple
        (reprojected_data, destination_transform)
    """
    # Create destination array
    dst_data = np.zeros(dst_shape, dtype=data.dtype)
    
    # Reproject
    reproject(
        source=data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        resampling=resampling
    )
    
    return dst_data, dst_transform

def create_dataset_from_files(grace_files, auxiliary_files, static_files,
                            grace_dir, auxiliary_dir, static_dir,
                            target_resolution=0.01):
    """Create dataset from GRACE and auxiliary files.
    
    Parameters
    ----------
    grace_files : list
        List of GRACE file paths
    auxiliary_files : dict
        Dictionary of auxiliary file lists
    static_files : dict
        Dictionary of static file paths
    grace_dir : str
        Directory with GRACE files
    auxiliary_dir : str
        Directory with auxiliary files
    static_dir : str
        Directory with static files
    target_resolution : float, optional
        Target resolution in degrees, by default 0.01
        
    Returns
    -------
    tuple
        (X, dates, lats, lons)
    """
    # Process GRACE files
    grace_data = []
    dates = []
    
    # Define target CRS
    dst_crs = rasterio.crs.CRS.from_epsg(4326)  # WGS84
    
    # Get target extent from first GRACE file
    grace_path = os.path.join(grace_dir, grace_files[0])
    _, src_transform, src_crs, bounds = load_raster(grace_path)
    
    # Calculate target shape
    width = int((bounds.right - bounds.left) / target_resolution)
    height = int((bounds.top - bounds.bottom) / target_resolution)
    dst_shape = (height, width)
    
    # Create target transform
    dst_transform = rasterio.transform.from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top,
        width, height
    )
    
    # Calculate lat/lon arrays
    lons = np.linspace(bounds.left, bounds.right, width)
    lats = np.linspace(bounds.top, bounds.bottom, height)
    
    # Process all GRACE files
    logger.info("Processing GRACE files...")
    for file_name in tqdm(grace_files):
        try:
            # Extract date from filename
            # Handle format 'GRACE_YYYY_MM.tif'
            parts = file_name.replace('.tif', '').split('_')
            if len(parts) >= 3:
                year = int(parts[1])
                month = int(parts[2])
            else:
                # Try to extract from the second part assuming GRACE_YYYYMM.tif
                date_part = parts[1]
                if len(date_part) >= 6:  # Long enough for YYYYMM
                    year = int(date_part[:4])
                    month = int(date_part[4:6])
                else:
                    raise ValueError(f"Unexpected date format in {file_name}")
            
            date = datetime(year, month, 1)
            dates.append(date)
            
            # Log successful parsing
            logger.debug(f"Parsed date {date} from {file_name}")
            
            # Load GRACE data
            grace_path = os.path.join(grace_dir, file_name)
            data, transform, crs, _ = load_raster(grace_path)
            
            # Check if data was loaded
            if data is None:
                logger.warning(f"Skipping {file_name} due to loading error")
                continue
            
            # Reproject if needed
            if crs != dst_crs or transform != dst_transform:
                data, _ = reproject_raster(
                    data, transform, crs,
                    dst_crs, dst_shape, dst_transform
                )
            
            grace_data.append(data)
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing date from {file_name}: {e}. Skipping this file.")
            continue  # Skip this file
    
    # Process auxiliary files
    auxiliary_datasets = {}
    
    # Define auxiliary category paths
    aux_category_paths = {
        'soil_moisture': os.path.join(auxiliary_dir, 'soil_moisture'),
        'precipitation': os.path.join(auxiliary_dir, 'precipitation'),
        'evapotranspiration': os.path.join(auxiliary_dir, 'evapotranspiration'),
        'groundwater': os.path.join(auxiliary_dir, 'groundwater'),
        'tws': os.path.join(auxiliary_dir, 'tws'),
        'baseflow': os.path.join(auxiliary_dir, 'baseflow'),
        'rootzone': os.path.join(auxiliary_dir, 'rootzone'),
        'profile_moisture': os.path.join(auxiliary_dir, 'profile_moisture')
    }
    
    # Create auxiliary directories if they don't exist
    for path in aux_category_paths.values():
        os.makedirs(path, exist_ok=True)
    
    for var_name, file_list in auxiliary_files.items():
        if var_name in aux_category_paths:
            category_path = aux_category_paths[var_name]
        else:
            category_path = auxiliary_dir  # Default to main auxiliary directory
            
        logger.info(f"Processing {var_name} files...")
        var_data = []
        
        for file_name in tqdm(file_list):
            # Load auxiliary data
            aux_path = os.path.join(category_path, file_name)
            data, transform, crs, _ = load_raster(aux_path)
            
            # Check if data was loaded
            if data is None:
                logger.warning(f"Skipping {file_name} due to loading error")
                continue
            
            # Reproject if needed
            if crs != dst_crs or transform != dst_transform:
                data, _ = reproject_raster(
                    data, transform, crs,
                    dst_crs, dst_shape, dst_transform
                )
            
            var_data.append(data)
        
        auxiliary_datasets[var_name] = var_data
    
    # Process static files
    static_datasets = {}
    
    for var_name, file_name in static_files.items():
        logger.info(f"Processing {var_name} static data...")
        
        # Load static data
        static_path = os.path.join(static_dir, file_name)
        data, transform, crs, _ = load_raster(static_path)
        
        # Check if data was loaded
        if data is None:
            logger.warning(f"Skipping {file_name} due to loading error")
            continue
        
        # Reproject if needed
        if crs != dst_crs or transform != dst_transform:
            data, _ = reproject_raster(
                data, transform, crs,
                dst_crs, dst_shape, dst_transform
            )
        
        static_datasets[var_name] = data
    
    # Create X array for each time step
    X = []
    
    for i in range(len(grace_data)):
        # Initialize feature array for this time step
        features = []
        
        # Add GRACE data
        features.append(grace_data[i])
        
        # Add auxiliary data for this time step
        for var_name, var_data in auxiliary_datasets.items():
            if i < len(var_data):
                features.append(var_data[i])
            else:
                logger.warning(f"Missing {var_name} data for time step {i}")
                features.append(np.zeros_like(grace_data[i]))
        
        # Add static data
        for var_name, data in static_datasets.items():
            features.append(data)
        
        # Stack features into single array
        X.append(np.stack(features, axis=-1))
    
    # Stack all time steps
    X = np.stack(X, axis=0)
    
    # Log feature information
    n_features = X.shape[-1]
    feature_names = ['GRACE'] + list(auxiliary_datasets.keys()) + list(static_datasets.keys())
    logger.info(f"Created dataset with {n_features} features: {', '.join(feature_names)}")
    logger.info(f"Dataset shape: {X.shape} (time steps, height, width, features)")
    
    return X, dates, lats, lons

def create_patches(X, patch_size=64, stride=32):
    """Create patches from input array.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input array (time, height, width, channels)
    patch_size : int, optional
        Patch size, by default 64
    stride : int, optional
        Stride for patch extraction, by default 32
        
    Returns
    -------
    tuple
        (patches, patch_indices)
    """
    # Get dimensions
    time_steps, height, width, channels = X.shape
    
    # Calculate number of patches
    n_h = 1 + (height - patch_size) // stride
    n_w = 1 + (width - patch_size) // stride
    
    # Initialize arrays
    patches = []
    patch_indices = []
    
    # Extract patches
    for t in range(time_steps):
        for i in range(n_h):
            for j in range(n_w):
                # Calculate patch coordinates
                h_start = i * stride
                h_end = h_start + patch_size
                w_start = j * stride
                w_end = w_start + patch_size
                
                # Extract patch
                patch = X[t, h_start:h_end, w_start:w_end, :]
                
                # Skip if patch has NaN values
                if np.isnan(patch).any():
                    continue
                
                # Add patch and indices
                patches.append(patch)
                patch_indices.append((t, h_start, w_start))
    
    # Convert to arrays
    patches = np.array(patches)
    patch_indices = np.array(patch_indices)
    
    return patches, patch_indices

def normalize_data(X, method='standard', feature_axis=-1, sample_axis=0):
    """Normalize data.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input array
    method : str, optional
        Normalization method ('standard', 'minmax'), by default 'standard'
    feature_axis : int, optional
        Feature axis, by default -1
    sample_axis : int, optional
        Sample axis, by default 0
        
    Returns
    -------
    tuple
        (normalized_data, scaler)
    """
    # Get original shape
    original_shape = X.shape
    
    # Reshape to 2D
    X_reshaped = np.moveaxis(X, feature_axis, -1)
    X_reshaped = X_reshaped.reshape(-1, X_reshaped.shape[-1])
    
    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform
    X_norm = scaler.fit_transform(X_reshaped)
    
    # Reshape back to original shape
    X_norm = X_norm.reshape(X.shape)
    
    return X_norm, scaler

def create_training_data(X, y, test_ratio=0.2, val_ratio=0.1, random_seed=42):
    """Create training, validation, and test datasets.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input features
    y : numpy.ndarray
        Target data
    test_ratio : float, optional
        Test set ratio, by default 0.2
    val_ratio : float, optional
        Validation set ratio, by default 0.1
    random_seed : int, optional
        Random seed, by default 42
        
    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Get number of samples
    n_samples = len(X)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split indices
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test+n_val]
    train_indices = indices[n_test+n_val:]
    
    # Split data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_dataset(X, y, dates, lats, lons, output_dir, prefix='dataset'):
    """Save dataset to disk.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input features
    y : numpy.ndarray
        Target data
    dates : list
        List of dates
    lats : numpy.ndarray
        Latitude values
    lons : numpy.ndarray
        Longitude values
    output_dir : str
        Output directory
    prefix : str, optional
        Filename prefix, by default 'dataset'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(str(output_dir / f"{prefix}_X.npy"), X)
    np.save(str(output_dir / f"{prefix}_y.npy"), y)
    np.save(str(output_dir / f"{prefix}_lats.npy"), lats)
    np.save(str(output_dir / f"{prefix}_lons.npy"), lons)
    
    # Save dates
    pd.DataFrame({'date': dates}).to_csv(
        str(output_dir / f"{prefix}_dates.csv"),
        index=False
    )
    
    # Create metadata
    metadata = {
        'X_shape': X.shape,
        'y_shape': y.shape,
        'n_samples': len(X),
        'n_time_steps': len(dates),
        'date_range': [str(dates[0]), str(dates[-1])],
        'lat_range': [float(lats.min()), float(lats.max())],
        'lon_range': [float(lons.min()), float(lons.max())],
        'features': X.shape[-1],
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save metadata
    pd.DataFrame([metadata]).to_csv(
        str(output_dir / f"{prefix}_metadata.csv"),
        index=False
    )
    
    logger.info(f"Dataset saved to {output_dir}")

def load_dataset(input_dir, prefix='dataset'):
    """Load dataset from disk.
    
    Parameters
    ----------
    input_dir : str
        Input directory
    prefix : str, optional
        Filename prefix, by default 'dataset'
        
    Returns
    -------
    tuple
        (X, y, dates, lats, lons)
    """
    input_dir = Path(input_dir)
    
    # Load arrays
    X = np.load(str(input_dir / f"{prefix}_X.npy"))
    y = np.load(str(input_dir / f"{prefix}_y.npy"))
    lats = np.load(str(input_dir / f"{prefix}_lats.npy"))
    lons = np.load(str(input_dir / f"{prefix}_lons.npy"))
    
    # Load dates
    dates_df = pd.read_csv(str(input_dir / f"{prefix}_dates.csv"))
    dates = pd.to_datetime(dates_df['date']).tolist()
    
    return X, y, dates, lats, lons