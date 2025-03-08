#!/usr/bin/env python
"""
Script to extract data for GRACE downscaling.

This script extracts GRACE, auxiliary, and USGS groundwater data for a specific region
and time period. It uses Google Earth Engine for GRACE and auxiliary data, and the USGS
REST API for groundwater data.

Usage:
    python run_data_extraction.py --region mississippi --start_date 2003-01-01 --end_date 2023-01-01
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
    get_groundwater_collection, get_tws_collection, get_baseflow_collection, get_rootzone_collection,
    get_profile_moisture_collection, get_surface_water_collection,
    filter_collection, extract_time_series, export_monthly_composites,
    get_static_features, export_static_features, export_to_drive
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
    parser.add_argument('--skip_grace', action='store_true',
                        help='Skip GRACE data extraction (use if GRACE data is already downloaded)')
    
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
    
    # Create subdirectories for different auxiliary data types
    auxiliary_subdirs = [
        'soil_moisture', 'precipitation', 'evapotranspiration', 
        'groundwater', 'tws', 'baseflow', 'rootzone',
        'profile_moisture', 'static'
    ]
    
    # Create all directories
    for directory in [grace_dir, groundwater_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    for subdir in auxiliary_subdirs:
        (auxiliary_dir / subdir).mkdir(parents=True, exist_ok=True)
    
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
    
    # Check if GRACE extraction should be skipped
    if args.skip_grace:
        logging.info("Skipping GRACE data extraction as requested (--skip_grace flag is set)...")
    else:
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
    
    # 1. Extract soil moisture data
    if 'soil_moisture' in config['data']['auxiliary']:
        logging.info("Exporting soil moisture data to Google Drive...")
        sm_source = config['data']['auxiliary']['soil_moisture']['collection']
        sm_band = config['data']['auxiliary']['soil_moisture']['band']
        sm_scale = config['data']['auxiliary']['soil_moisture']['scale']
        
        sm_collection = get_soil_moisture_collection(sm_source)
        output_folder = f"SM_{region_name}"
        
        sm_tasks = export_to_drive(
            sm_collection, region, sm_band, output_folder, 'SM',
            start_date, end_date, sm_scale, time_step='month'
        )
        logging.info(f"Started {len(sm_tasks)} export tasks for soil moisture data")
    
    # 2. Extract precipitation data
    if 'precipitation' in config['data']['auxiliary']:
        logging.info("Exporting precipitation data to Google Drive...")
        precip_source = config['data']['auxiliary']['precipitation']['collection']
        precip_band = config['data']['auxiliary']['precipitation']['band']
        precip_scale = config['data']['auxiliary']['precipitation']['scale']
        
        precip_collection = get_precipitation_collection(precip_source)
        output_folder = f"PRECIP_{region_name}"
        
        precip_tasks = export_to_drive(
            precip_collection, region, precip_band, output_folder, 'PRECIP',
            start_date, end_date, precip_scale, time_step='month'
        )
        logging.info(f"Started {len(precip_tasks)} export tasks for precipitation data")

    # 3. Extract evapotranspiration data
    if 'evapotranspiration' in config['data']['auxiliary']:
        logging.info("Exporting evapotranspiration data to Google Drive...")
        et_source = config['data']['auxiliary']['evapotranspiration']['collection']
        et_band = config['data']['auxiliary']['evapotranspiration']['band']
        et_scale = config['data']['auxiliary']['evapotranspiration']['scale']
        
        et_collection = get_evapotranspiration_collection(et_source)
        output_folder = f"ET_{region_name}"
        
        et_tasks = export_to_drive(
            et_collection, region, et_band, output_folder, 'ET',
            start_date, end_date, et_scale, time_step='month'
        )
        logging.info(f"Started {len(et_tasks)} export tasks for evapotranspiration data")
    
    # 4. Extract groundwater data from GLDAS
    if 'groundwater' in config['data']['auxiliary']:
        logging.info("Exporting GLDAS groundwater data to Google Drive...")
        gw_source = config['data']['auxiliary']['groundwater']['collection']
        gw_band = config['data']['auxiliary']['groundwater']['band']
        gw_scale = config['data']['auxiliary']['groundwater']['scale']
        
        gw_collection = get_groundwater_collection(gw_source)
        output_folder = f"GW_{region_name}"
        
        gw_tasks = export_to_drive(
            gw_collection, region, gw_band, output_folder, 'GW',
            start_date, end_date, gw_scale, time_step='month'
        )
        logging.info(f"Started {len(gw_tasks)} export tasks for groundwater data")

    # 5. Extract TWS data from GLDAS
    if 'tws' in config['data']['auxiliary']:
        logging.info("Exporting TWS data to Google Drive...")
        tws_source = config['data']['auxiliary']['tws']['collection']
        tws_band = config['data']['auxiliary']['tws']['band']
        tws_scale = config['data']['auxiliary']['tws']['scale']
        
        tws_collection = get_tws_collection(tws_source)
        output_folder = f"TWS_{region_name}"
        
        tws_tasks = export_to_drive(
            tws_collection, region, tws_band, output_folder, 'TWS',
            start_date, end_date, tws_scale, time_step='month'
        )
        logging.info(f"Started {len(tws_tasks)} export tasks for TWS data")

    # 6. Extract baseflow data from GLDAS
    if 'baseflow' in config['data']['auxiliary']:
        logging.info("Exporting baseflow data to Google Drive...")
        bf_source = config['data']['auxiliary']['baseflow']['collection']
        bf_band = config['data']['auxiliary']['baseflow']['band']
        bf_scale = config['data']['auxiliary']['baseflow']['scale']
        
        bf_collection = get_baseflow_collection(bf_source)
        output_folder = f"BF_{region_name}"
        
        bf_tasks = export_to_drive(
            bf_collection, region, bf_band, output_folder, 'BF',
            start_date, end_date, bf_scale, time_step='month'
        )
        logging.info(f"Started {len(bf_tasks)} export tasks for baseflow data")

    # 7. Extract root zone soil moisture data from GLDAS
    if 'rootzone' in config['data']['auxiliary']:
        logging.info("Exporting root zone soil moisture data to Google Drive...")
        rz_source = config['data']['auxiliary']['rootzone']['collection']
        rz_band = config['data']['auxiliary']['rootzone']['band']
        rz_scale = config['data']['auxiliary']['rootzone']['scale']
        
        rz_collection = get_rootzone_collection(rz_source)
        output_folder = f"RZ_{region_name}"
        
        rz_tasks = export_to_drive(
            rz_collection, region, rz_band, output_folder, 'RZ',
            start_date, end_date, rz_scale, time_step='month'
        )
        logging.info(f"Started {len(rz_tasks)} export tasks for root zone soil moisture data")

    # 8. Extract profile soil moisture data from GLDAS
    if 'profile_moisture' in config['data']['auxiliary']:
        logging.info("Exporting profile soil moisture data to Google Drive...")
        pm_source = config['data']['auxiliary']['profile_moisture']['collection']
        pm_band = config['data']['auxiliary']['profile_moisture']['band']
        pm_scale = config['data']['auxiliary']['profile_moisture']['scale']
        
        pm_collection = get_profile_moisture_collection(pm_source)
        output_folder = f"PM_{region_name}"
        
        pm_tasks = export_to_drive(
            pm_collection, region, pm_band, output_folder, 'PM',
            start_date, end_date, pm_scale, time_step='month'
        )
        logging.info(f"Started {len(pm_tasks)} export tasks for profile soil moisture data")
    
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