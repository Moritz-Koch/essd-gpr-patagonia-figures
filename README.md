# Figure scripts – GPR ice thickness of Glaciar Perito Moreno, Upsala and Viedma

Python scripts for the figures and numbers produced in Python for
Koch et al., *Earth System Science Data* (essd-2025-678). All scripts run on the
published dataset: https://doi.org/10.5281/zenodo.21099957

| Script | Output |
|---|---|
| `scripts/fig_scatter_density_4panel.py` | Measured vs. modelled bedrock elevation (Fig. 8); `--mode ice_thickness` for ice thickness |
| `scripts/volume_uncertainty.py` | Ice volume and volume uncertainty per glacier (831 ± 155 km³ in total) |
| `scripts/fig_3d_bedrock_lake_level.py` | 3-D views of the reconstructed ice and of the bed below lake level |

## Data

1. Download from Zenodo into `data/`:
   `01_GPR_measurement_data.csv`, `03_Bedrock_Elevation_20x20_res.tif`,
   `04_Ice_Thickness_Fields_100x100_res.tif`, `05_Ice_thickness_Fields_uncertainty.tif`.
2. Only for `fig_3d_bedrock_lake_level.py`: a NASADEM mosaic of the study area
   (NASA JPL, 2020, https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001),
   reprojected to EPSG:32718, saved as `data/nasadem_2000.tif`.

Glacier outlines (RGI v7.0, year 2000; RGI Consortium, 2023, CC BY 4.0) are included in `auxiliary/`.
Other locations: set `ESSD_GPR_DATA`, `ESSD_GPR_NASADEM` and `ESSD_GPR_OUT`.

## Run

```
pip install -r requirements.txt
python scripts/fig_scatter_density_4panel.py
python scripts/volume_uncertainty.py
python scripts/fig_3d_bedrock_lake_level.py
```

Outputs are written to `output/`.

## Reference systems

All data: WGS 84 / UTM zone 18S (EPSG:32718). Bedrock raster and lake levels (179 m, 252 m) are
orthometric heights above EGM96 (m a.s.l.); GPR point elevations in the CSV are ellipsoidal heights
(WGS 84). See the README of the Zenodo record.

## License

MIT (code). Data: see the Zenodo record and the RGI licence.
