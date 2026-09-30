"""
4-panel scatter-density plot of modelled vs. measured bedrock elevation (Fig. 8)
or ice thickness: (a) Perito Moreno, (b) Viedma, (c) Upsala, (d) all glaciers.

Measured values: 01_GPR_measurement_data.csv. Modelled values: the Zenodo raster
sampled at each GPR point (03 bedrock elevation or 04 ice thickness).

  python scripts/fig_scatter_density_4panel.py                        # bedrock, all points
  python scripts/fig_scatter_density_4panel.py --mode ice_thickness
  python scripts/fig_scatter_density_4panel.py --every50              # every 50th point (~100 m)
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import config
from config import fmt_num

# ============================================================
# 0. SETTINGS
# ============================================================

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["bedrock", "ice_thickness"], default="bedrock")
ap.add_argument("--every50", action="store_true", help="use every 50th point per glacier")
args = ap.parse_args()

COMPARE_MODE = args.mode
USE_EVERY_50TH_POINT = args.every50
SUBSAMPLING_STEP = 50

csv_path = config.GPR_CSV
ice_thickness_tif_path = config.THK_TIF
bedrock_tif_path = config.BED_TIF
out_dir = config.OUT

GLACIERS = ["Perito Moreno", "Viedma", "Upsala"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

HEX_GRIDSIZE = 60

int_formatter = FuncFormatter(lambda x, pos: fmt_num(x, 0))


# ============================================================
# 1. MODE
# ============================================================

if COMPARE_MODE == "bedrock":
    measured_col, modelled_col = "Bedrock_el", "Bedrock_el_modelled"
    tif_path = bedrock_tif_path
    x_label = "Measured bedrock elevation (m a.s.l.)"
    y_label = "Modelled bedrock elevation (m a.s.l.)"
    footer = "Bedrock elevation comparison"
    output_label = "bedrock_elevation"
elif COMPARE_MODE == "ice_thickness":
    measured_col, modelled_col = "IceThickn", "IceThickn_modelled"
    tif_path = ice_thickness_tif_path
    x_label = "Measured ice thickness (m)"
    y_label = "Modelled ice thickness (m)"
    footer = "Ice thickness comparison"
    output_label = "ice_thickness"
else:
    raise ValueError("COMPARE_MODE must be either 'bedrock' or 'ice_thickness'.")

if USE_EVERY_50TH_POINT:
    sampling_label = f"every_{SUBSAMPLING_STEP}th_point"
    sampling_text = f"every {SUBSAMPLING_STEP}th point"
else:
    sampling_label = "all_points"
    sampling_text = "all points"

stem = f"scatter_density_4panel_modelled_vs_measured_{output_label}_{sampling_label}"
out_png = out_dir / f"{stem}.png"
out_pdf = out_dir / f"{stem}.pdf"
out_csv = out_dir / f"modelled_vs_measured_{output_label}_comparison_{sampling_label}.csv"


# ============================================================
# 2. READ CSV
# ============================================================

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df["Glacier"] = df["Glacier"].astype(str).str.strip()

for col in ["Easting", "Northing", "Surface_el", "Traveltime(ns)",
            "IceThickn", "Bedrock_el", "Uncertainty"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Easting", "Northing", "Glacier", measured_col])
df["original_index"] = np.arange(len(df))

print("Points per glacier in CSV:")
print(df["Glacier"].value_counts())

if USE_EVERY_50TH_POINT:
    df = (
        df.sort_values(["Glacier", "original_index"])
          .groupby("Glacier", group_keys=False)
          .nth[::SUBSAMPLING_STEP]
          .copy()
    )


# ============================================================
# 3. SAMPLE RASTER AT POINT LOCATIONS
# ============================================================

with rasterio.open(tif_path) as src:
    coords = list(zip(df["Easting"], df["Northing"]))
    df[modelled_col] = [v[0] for v in src.sample(coords)]
    if src.nodata is not None:
        df.loc[df[modelled_col] == src.nodata, modelled_col] = np.nan

df = df.dropna(subset=[measured_col, modelled_col])

if COMPARE_MODE == "ice_thickness":
    df = df[(df[measured_col] > 0) & (df[modelled_col] > 0)]
    df = df[(df[measured_col] <= 1000) & (df[modelled_col] <= 1000)]
else:
    df = df[(df[measured_col] > -3000) & (df[modelled_col] > -3000)]
    df = df[(df[measured_col] < 4000) & (df[modelled_col] < 4000)]

df["signed_error"] = df[modelled_col] - df[measured_col]
df["abs_error"] = df["signed_error"].abs()

print("\nValid comparison points per glacier:")
print(df["Glacier"].value_counts())


def stats(d):
    e = d["signed_error"]
    return {
        "SD": e.std(),
        "MAE": d["abs_error"].mean(),
        "Bias": e.mean(),
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "n": len(d),
    }


panels = [(g, df[df["Glacier"] == g]) for g in GLACIERS] + [("All glaciers", df)]


# ============================================================
# 4. AXIS LIMITS (shared by all panels)
# ============================================================

measured = df[measured_col].values
modelled = df[modelled_col].values

if COMPARE_MODE == "ice_thickness":
    axis_min = 0
    axis_max = int(np.ceil(max(measured.max(), modelled.max()) / 100.0) * 100)
else:
    padding = 50
    axis_min = int(np.floor((min(measured.min(), modelled.min()) - padding) / 100.0) * 100)
    axis_max = int(np.ceil((max(measured.max(), modelled.max()) + padding) / 100.0) * 100)


# ============================================================
# 5. PLOT
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
axes = axes.ravel()

hexbins = []
for ax, label, (name, d) in zip(axes, PANEL_LABELS, panels):
    hb = ax.hexbin(
        d[measured_col].values,
        d[modelled_col].values,
        gridsize=HEX_GRIDSIZE,
        extent=(axis_min, axis_max, axis_min, axis_max),
        mincnt=1,
        cmap="viridis",
    )
    hexbins.append(hb)

    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linewidth=1.0)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(name)

    ax.xaxis.set_major_formatter(int_formatter)
    ax.yaxis.set_major_formatter(int_formatter)
    ax.tick_params(labelbottom=True, labelleft=True)

    ax.text(0.04, 0.95, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=15, fontweight="bold")

    s = stats(d)
    ax.text(
        0.04, 0.85,
        (
            f"SD = {fmt_num(s['SD'], 1)} m\n"
            f"MAE = {fmt_num(s['MAE'], 1)} m\n"
            f"Bias = {fmt_num(s['Bias'], 1)} m\n"
            f"RMSE = {fmt_num(s['RMSE'], 1)} m\n"
            f"n = {fmt_num(s['n'], 0)}"
        ),
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
    )

    print(f"{name:14s} SD {s['SD']:6.1f}  MAE {s['MAE']:6.1f}  Bias {s['Bias']:6.1f}  "
          f"RMSE {s['RMSE']:6.1f}  n {s['n']:,}")

# Tick labels only on the outer axes, as in the original figure
for ax in axes[:2]:
    ax.tick_params(labelbottom=False)
for ax in axes[1::2]:
    ax.tick_params(labelleft=False)
for ax in axes[2:]:
    ax.set_xlabel(x_label)
for ax in axes[::2]:
    ax.set_ylabel(y_label)

# One colour scale for all four panels
vmax = max(hb.get_array().max() for hb in hexbins)
norm = LogNorm(vmin=1, vmax=vmax)
for hb in hexbins:
    hb.set_norm(norm)

fig.subplots_adjust(left=0.08, right=0.86, bottom=0.10, top=0.95, wspace=0.12, hspace=0.18)
cax = fig.add_axes([0.89, 0.10, 0.025, 0.85])
cbar = fig.colorbar(hexbins[-1], cax=cax)
cbar.set_label("N points (log scale)")
cbar.ax.yaxis.set_major_locator(LogLocator(base=10))
cbar.ax.yaxis.set_major_formatter(int_formatter)
cbar.ax.yaxis.set_minor_formatter(NullFormatter())

fig.text(0.47, 0.03, f"{footer}; Sampling: {sampling_text}", ha="center", fontsize=11)

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved PNG to: {out_png}")
print(f"Saved PDF to: {out_pdf}")

df.to_csv(out_csv, index=False)
print(f"Saved comparison table to: {out_csv}")
