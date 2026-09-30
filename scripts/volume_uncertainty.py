"""
Ice volume and volume uncertainty per glacier (Sect. 4.3: 831 +- 155 km3 in total).

Inputs (Zenodo): 04_Ice_Thickness_Fields_100x100_res.tif (H),
                 05_Ice_thickness_Fields_uncertainty.tif (sigma_H), RGI v7.0 outlines (2000).

Per glacier (pixels whose centre falls inside the outline):
  V              = sum(H_i) * A_px
  sigma_V        = sum(sigma_H_i) * A_px             errors fully correlated (reported value)
  sigma_V_uncorr = sqrt(sum(sigma_H_i^2)) * A_px     errors independent (lower bound, for reference)

Thickness errors of a reconstructed bed are strongly spatially correlated, so the fully
correlated sum is the value to report.

  python scripts/volume_uncertainty.py
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

import config


def read_band(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        if src.nodata is not None and not np.isnan(src.nodata):
            a[a == src.nodata] = np.nan
        return a, src.transform, src.crs, src.shape


def read_on_grid(path, transform, crs, shape):
    # nearest neighbour: both rasters share the 100 m pixel lattice, so this is an exact crop/pad
    with rasterio.open(path) as src:
        dst = np.full(shape, np.nan)
        reproject(rasterio.band(src, 1), dst, dst_transform=transform, dst_crs=crs,
                  src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.nearest)
    return dst


def main():
    H, transform, crs, shape = read_band(config.THK_TIF)
    S = read_on_grid(config.UNC_TIF, transform, crs, shape)
    px_area = abs(transform.a * transform.e)  # m^2

    gdf = gpd.read_file(config.OUTLINES).to_crs(crs.to_wkt())

    rows = []
    for _, g in gdf.iterrows():
        inside = geometry_mask([g.geometry], out_shape=shape, transform=transform, invert=True)
        h_ok = inside & np.isfinite(H)
        h, s = H[h_ok], S[h_ok]
        # a few pixels have H but no sigma: give them the glacier's mean relative uncertainty
        s_missing = ~np.isfinite(s)
        if s_missing.any():
            s = np.where(s_missing, np.nansum(s) / np.sum(h[~s_missing]) * h, s)

        V, sV = h.sum() * px_area, s.sum() * px_area
        rows.append({
            "glacier": g["glacier"],
            "rgi_id": g["rgi_id"],
            "area_with_thickness_km2": h_ok.sum() * px_area / 1e6,
            "mean_H_m": h.mean(),
            "max_H_m": h.max(),
            "volume_km3": V / 1e9,
            "sigma_V_km3": sV / 1e9,
            "sigma_V_pct": 100 * sV / V,
            "sigma_V_uncorrelated_km3": np.sqrt(np.sum(s ** 2)) * px_area / 1e9,
        })

    df = pd.DataFrame(rows)
    tot = {
        "glacier": "Total",
        "area_with_thickness_km2": df.area_with_thickness_km2.sum(),
        "volume_km3": df.volume_km3.sum(),
        "sigma_V_km3": df.sigma_V_km3.sum(),          # fully correlated: adds linearly
        "sigma_V_uncorrelated_km3": np.sqrt((df.sigma_V_uncorrelated_km3 ** 2).sum()),
    }
    tot["sigma_V_pct"] = 100 * tot["sigma_V_km3"] / tot["volume_km3"]
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)

    out = config.OUT / "volume_uncertainty_per_glacier.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    for _, r in df.iterrows():
        print(f"{r.glacier:<14} V = {r.volume_km3:7.1f} +- {r.sigma_V_km3:5.1f} km3 ({r.sigma_V_pct:4.1f} %)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
