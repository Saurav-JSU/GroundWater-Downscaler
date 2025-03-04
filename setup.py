"""
Setup script for GRACE Groundwater Downscaling package.
"""
from setuptools import setup, find_packages

setup(
    name="grace_downscaling",
    version="0.1.0",
    description="A CNN-based approach for downscaling GRACE data to high-resolution groundwater estimates",
    author="Saurav Bhattarai",
    author_email="saurav.bhattarai.1999@gmail.com",
    url="https://github.com/Saurav-JSU/GRACE-Groundwater-Downscaling",
    packages=find_packages(),
    install_requires=[
        "earthengine-api>=0.1.332",
        "geemap>=0.20.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "rasterio>=1.2.0",
        "requests>=2.26.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "geopandas>=0.10.0",
        "xarray>=0.20.0",
        "netCDF4>=1.5.0",
        "plotly>=5.5.0",
        "jupyterlab>=3.2.0",
        "pyproj>=3.3.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    python_requires=">=3.8",
)