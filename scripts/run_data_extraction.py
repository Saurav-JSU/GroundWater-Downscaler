#!/usr/bin/env python
"""
Script to extract data for GRACE downscaling.

This script extracts GRACE, auxiliary, and USGS groundwater data for a specific region
and time period. It uses Google Earth Engine for GRACE and auxiliary data, and the USGS
REST API for groundwater data.

Usage:
    python run_data_extraction.py --region mississippi --start_date 2002-04-01 --end_date 2023-01-01
"""
import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
sys.path.append(str(src_dir))

# Import Earth Engine first and initialize it
import ee
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    ee.Initialize()
    logger.info("Earth Engine pre-initialized successfully")
except Exception as e:
    logger.info(f"Earth Engine pre-initialization not successful, will try again later: {e}")

# Now import other modules
from src.data.earth_engine import initialize_earth_engine, get_region_geometry, display_region
from src.data.grace import get_grace_collection, filter_grace_collection, extract_grace_time_series, export_grace_images
from src.data.usgs import get_usgs_groundwater_data, convert_to_geodataframe, get_monthly_aggregation
from src.data.auxiliary import (
    get_soil_moisture_collection, get_precipitation_collection, get_evapotranspiration_collection,
    get_surface_water_collection, filter_collection, extract_time_series, export_monthly_composites,
    get_static_features, export_static_features
)

def setup_logging(log_file=None):
    """Set up logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Log file path, by default None
    """
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
    """Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def parse_args():
    """Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Command line arguments
    """
    parser = argparse.ArgumentParser(description='Extract data for GRACE downscaling')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--region', type=str,
                        help='Region name (e.g., "Mississippi")')
    parser.add_argument('--start_date', type=str,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory')
    parser.add_argument('--log_file', type=str,
                        help='Log file path')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_file)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.region:
        config['data']['region']['name'] = args.region
    if args.start_date:
        config['data']['date_range']['start_date'] = args.start_date
    if args.end_date:
        config['data']['date_range']['end_date'] = args.end_date
    if args.output_dir:
        config['paths']['raw_dir'] = args.output_dir
    
    # Extract configuration
    region_name = config['data']['region']['name']
    region_source = config['data']['region']['source']
    start_date = config['data']['date_range']['start_date']
    end_date = config['data']['date_range']['end_date']
    grace_collection = config['data']['grace']['collection']
    grace_band = config['data']['grace']['band']
    grace_scale = config['data']['grace']['scale']
    state_code = config['data']['groundwater']['state_code']
    raw_dir = config['paths']['raw_dir']
    
    # Create output directories
    raw_dir = Path(raw_dir)
    grace_dir = raw_dir / 'grace'
    auxiliary_dir = raw_dir / 'auxiliary'
    groundwater_dir = raw_dir / 'groundwater'
    
    for directory in [grace_dir, auxiliary_dir, groundwater_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize Earth Engine
    logging.info("Initializing Earth Engine...")
    success = initialize_earth_engine()
    
    if not success:
        logging.error("Failed to initialize Earth Engine. Exiting.")
        return
    
    # Get region geometry
    logging.info(f"Getting geometry for {region_name}...")
    region = get_region_geometry(region_name, region_source)
    
    if region is None:
        logging.error(f"Failed to get geometry for {region_name}. Exiting.")
        logging.info("Try using the proper case: 'Mississippi' instead of 'mississippi'")
        return
    
    # Display region (optional)
    # m = display_region(region, region_name)
    # m.to_notebook()
    
    # Extract GRACE data
    logging.info("Extracting GRACE data...")
    grace = get_grace_collection(grace_collection)
    grace_filtered = filter_grace_collection(grace, start_date, end_date, grace_band)
    
    # Extract GRACE time series
    logging.info("Extracting GRACE time series...")
    grace_ts = extract_grace_time_series(grace_filtered, region, grace_scale)
    
    # Export GRACE time series
    grace_ts.to_csv(str(grace_dir / 'grace_time_series.csv'), index=False)
    
    # Export GRACE images
    logging.info("Exporting GRACE images...")
    output_folder = f"GRACE_{region_name}"
    grace_tasks = export_grace_images(grace_filtered, region, output_folder, 'GRACE', grace_scale)
    
    # Extract auxiliary data
    logging.info("Extracting auxiliary data...")
    
    # Extract soil moisture
    logging.info("Extracting soil moisture data...")
    sm_collection = get_soil_moisture_collection(config['data']['auxiliary']['soil_moisture']['collection'])
    sm_band = config['data']['auxiliary']['soil_moisture']['band']
    sm_scale = config['data']['auxiliary']['soil_moisture']['scale']
    
    # Increase scale (lower resolution) for memory-intensive datasets
    sm_scale_adjusted = sm_scale * 2  # Double the scale to reduce memory usage
    
    sm_filtered = filter_collection(sm_collection, start_date, end_date, sm_band)
    # Process in smaller chunks to avoid memory limits
    sm_ts = extract_time_series(sm_filtered, region, sm_scale_adjusted, chunk_size=3)
    
    # Export soil moisture time series if we have data
    if sm_ts is not None:
        sm_ts.to_csv(str(auxiliary_dir / 'soil_moisture_time_series.csv'), index=False)
    else:
        logging.warning("No soil moisture time series data extracted")
    
    # Export soil moisture images with adjusted scale
    logging.info("Exporting soil moisture images...")
    output_folder = f"SM_{region_name}"
    sm_tasks = export_monthly_composites(
        sm_collection, start_date, end_date, region, sm_band,
        output_folder, 'SM', sm_scale_adjusted
    )
    
    # Extract precipitation
    logging.info("Extracting precipitation data...")
    precip_collection = get_precipitation_collection(config['data']['auxiliary']['precipitation']['collection'])
    precip_band = config['data']['auxiliary']['precipitation']['band']
    precip_scale = config['data']['auxiliary']['precipitation']['scale']
    
    precip_filtered = filter_collection(precip_collection, start_date, end_date, precip_band)
    precip_ts = extract_time_series(precip_filtered, region, precip_scale, chunk_size=6)
    
    # Export precipitation time series if we have data
    if precip_ts is not None:
        precip_ts.to_csv(str(auxiliary_dir / 'precipitation_time_series.csv'), index=False)
    else:
        logging.warning("No precipitation time series data extracted")
    
    # Export precipitation images
    logging.info("Exporting precipitation images...")
    output_folder = f"PRECIP_{region_name}"
    precip_tasks = export_monthly_composites(
        precip_collection, start_date, end_date, region, precip_band,
        output_folder, 'PRECIP', precip_scale
    )
    
    # Extract evapotranspiration
    logging.info("Extracting evapotranspiration data...")
    et_collection = get_evapotranspiration_collection(config['data']['auxiliary']['evapotranspiration']['collection'])
    et_band = config['data']['auxiliary']['evapotranspiration']['band']
    et_scale = config['data']['auxiliary']['evapotranspiration']['scale']
    
    # Increase scale for memory-intensive datasets
    et_scale_adjusted = et_scale * 2  # Double the scale to reduce memory usage
    
    et_filtered = filter_collection(et_collection, start_date, end_date, et_band)
    et_ts = extract_time_series(et_filtered, region, et_scale_adjusted, chunk_size=3)
    
    # Export evapotranspiration time series if we have data
    if et_ts is not None:
        et_ts.to_csv(str(auxiliary_dir / 'evapotranspiration_time_series.csv'), index=False)
    else:
        logging.warning("No evapotranspiration time series data extracted")
    
    # Export evapotranspiration images
    logging.info("Exporting evapotranspiration images...")
    output_folder = f"ET_{region_name}"
    et_tasks = export_monthly_composites(
        et_collection, start_date, end_date, region, et_band,
        output_folder, 'ET', et_scale_adjusted
    )
    
    # Extract static features
    logging.info("Extracting static features...")
    static_features = get_static_features(region)
    
    # Export static features
    logging.info("Exporting static features...")
    output_folder = f"STATIC_{region_name}"
    static_tasks = export_static_features(static_features, region, output_folder)
    
    # Extract USGS groundwater data
    logging.info(f"Extracting USGS groundwater data for {state_code}...")
    gw_data = get_usgs_groundwater_data(state_code, start_date, end_date)
    
    if gw_data is not None:
        # Export groundwater data
        gw_data.to_csv(str(groundwater_dir / 'groundwater_data.csv'), index=False)
        
        # Convert to GeoDataFrame
        gw_gdf = convert_to_geodataframe(gw_data)
        
        # Export GeoDataFrame
        if gw_gdf is not None:
            gw_gdf.to_file(str(groundwater_dir / 'groundwater_data.geojson'), driver='GeoJSON')
        
        # Create monthly aggregation
        gw_monthly = get_monthly_aggregation(gw_data)
        
        if gw_monthly is not None:
            # Export monthly data
            gw_monthly.to_csv(str(groundwater_dir / 'groundwater_monthly.csv'), index=False)
    
    logging.info("Data extraction complete.")
    logging.info(f"GRACE data exported to {grace_dir}")
    logging.info(f"Auxiliary data exported to {auxiliary_dir}")
    logging.info(f"Groundwater data exported to {groundwater_dir}")

if __name__ == '__main__':
    main()