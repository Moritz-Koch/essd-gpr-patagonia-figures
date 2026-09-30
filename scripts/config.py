"""
Input and output paths shared by all scripts.

Put the Zenodo files (https://doi.org/10.5281/zenodo.21099957) into ./data, or
point the environment variable ESSD_GPR_DATA to the folder that holds them.
NASADEM (only for the terrain in fig_3d_bedrock_lake_level.py) is expected at
data/nasadem_2000.tif unless ESSD_GPR_NASADEM says otherwise.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ESSD_GPR_DATA", ROOT / "data"))
OUT = Path(os.environ.get("ESSD_GPR_OUT", ROOT / "output"))
OUT.mkdir(parents=True, exist_ok=True)

# Zenodo record 10.5281/zenodo.21099957 (v4)
GPR_CSV = DATA / "01_GPR_measurement_data.csv"
BED_TIF = DATA / "03_Bedrock_Elevation_20x20_res.tif"
THK_TIF = DATA / "04_Ice_Thickness_Fields_100x100_res.tif"
UNC_TIF = DATA / "05_Ice_thickness_Fields_uncertainty.tif"

# RGI v7.0 outlines (year 2000) of the three glaciers, shipped with this repository
OUTLINES = ROOT / "auxiliary" / "rgi7_outlines_2000_study_glaciers.gpkg"

# NASADEM (year 2000), EPSG:32718 mosaic of the study area; terrain around the glaciers
NASADEM = Path(os.environ.get("ESSD_GPR_NASADEM", DATA / "nasadem_2000.tif"))

# lake levels (m a.s.l., from SRTM), as in the manuscript
LAKE_LEVELS = {"Lago Argentino": 179.0, "Lago Viedma": 252.0}

GLACIERS = {
    # name: (RGI v7.0 id, lake the glacier calves into)
    "Perito Moreno": ("RGI2000-v7.0-G-17-14719", "Lago Argentino"),
    "Upsala": ("RGI2000-v7.0-G-17-04760", "Lago Argentino"),
    "Viedma": ("RGI2000-v7.0-G-17-05272", "Lago Viedma"),
}


def fmt_num(x, decimals=0):
    """1,234.5 formatting (comma thousands, point decimal) with a typographic minus."""
    return f"{x:,.{decimals}f}".replace("-", "−")
