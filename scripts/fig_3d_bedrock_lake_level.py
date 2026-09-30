"""
Paper figure: 3-D views of the three survey glaciers.

  (a) overview: reconstructed ice (bed + ice thickness, coloured by thickness)
      with the surrounding terrain and the lakes in front
  (b)-(d) Perito Moreno, Upsala, Viedma without the ice: glacier bed coloured
      by elevation with contour lines, the lake-level contour, and the part of
      the bed below lake level under water

Inputs (Zenodo 10.5281/zenodo.21099957):
  03_Bedrock_Elevation_20x20_res.tif, 04_Ice_Thickness_Fields_100x100_res.tif
Terrain around the glaciers: NASADEM 2000 (see README). Lakes are nodata there;
they get a flat floor under the water, and small DEM artefacts in the lakes
(e.g. in front of the calving fronts) are lowered to the surrounding values.
Area statistics use only the bedrock raster and the RGI outlines.

  python scripts/fig_3d_bedrock_lake_level.py
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import rasterio
import rasterio.enums
import vtk
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from rasterio.features import geometry_mask
from rasterio.fill import fillnodata
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage

import config
from config import fmt_num

CSV = config.GPR_CSV
BED_TIF = config.BED_TIF
THK_TIF = config.THK_TIF
DEM_TIF = config.NASADEM
OUTLINES = config.OUTLINES
OUT = config.OUT / "glaciers_3d_bed_below_lake_level"

LAKES = config.LAKE_LEVELS
GLACIERS = config.GLACIERS
ORDER = ["Perito Moreno", "Upsala", "Viedma"]    # south to north = left to right in (a)

# scene extents of (b)-(d): the map frames of the bedrock figure in the manuscript,
# read off its graticule (lon_min, lat_min, lon_max, lat_max)
REGIONS = {
    "Perito Moreno": (-73.230, -50.628, -73.004, -50.458),
    "Upsala": (-73.428, -49.996, -73.139, -49.752),
    "Viedma": (-73.245, -49.586, -72.980, -49.364),
}

VE = 2.0
LAKE_FLOOR_DEPTH = 60.0
CONTOUR_STEP = 100.0          # m, bed contour interval
BED_CLIM = (-500.0, 1500.0)
THK_CLIM = (0.0, 1000.0)

# light theme for the paper
BG = "white"
TERRAIN = "#b9bfbd"
WATER = "#3f8fc9"
LAKE_LINE = "#0b4f8a"
BED_CMAP = LinearSegmentedColormap.from_list(
    "bed", ["#2a3440", "#5b5470", "#9a6a6a", "#c98a62", "#e6c49a", "#f4eee2"])
THK_CMAP = LinearSegmentedColormap.from_list(
    "thk", ["#cfe6f2", "#a9d0e3", "#5aa1c8", "#2a6fa6", "#123f73", "#0a1f40"])


# ---------------------------------------------------------------------------
# rasters
# ---------------------------------------------------------------------------
def read(path, bounds, shape, resampling, nodata_values=()):
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, src.transform)
        a = src.read(1, window=win, out_shape=shape, resampling=resampling,
                     masked=True).astype("float64").filled(np.nan)
    for v in nodata_values:
        a[a == v] = np.nan
    return a


def prepare(bounds, stride, polys):
    """Bed, ice thickness, cleaned terrain and lake map on one grid.

    stride: decimation of the 20 x 30 m bedrock grid.
    """
    with rasterio.open(BED_TIF) as src:
        win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        full_tr = src.window_transform(win)
        shape = (int(win.height) // stride, int(win.width) // stride)
        bounds = rasterio.windows.bounds(win, src.transform)
    tr = full_tr * rasterio.Affine.scale(win.width / shape[1] / 1.0, win.height / shape[0] / 1.0)

    resamp = rasterio.enums.Resampling.average if stride > 1 else rasterio.enums.Resampling.nearest
    bed = read(BED_TIF, bounds, shape, resamp)
    thk = read(THK_TIF, bounds, shape, rasterio.enums.Resampling.bilinear, nodata_values=(0.0,))
    dem = read(DEM_TIF, bounds, shape, rasterio.enums.Resampling.bilinear)

    gid = np.zeros(shape, dtype=int)                       # 1.. = index in ORDER + 1
    for i, name in enumerate(ORDER):
        m = geometry_mask([polys[name]], out_shape=shape, transform=tr, invert=True)
        gid[m & np.isfinite(bed)] = i + 1
    glac = gid > 0

    # DEM voids: interpolate, then decide per connected void which lake it is
    void = ~np.isfinite(dem)
    filled = fillnodata(np.where(void, 0.0, dem).astype("float32"), mask=(~void).astype("uint8"),
                        max_search_distance=max(40, 400 // stride),
                        smoothing_iterations=2).astype("float64")
    lab, nlab = ndimage.label(void)
    lake_of = np.full(shape, np.nan)
    if nlab:
        low = ndimage.minimum(filled, lab, index=np.arange(1, nlab + 1))
        comp_level = np.full(nlab + 1, np.nan)
        for lvl in sorted(LAKES.values()):
            sel = np.isnan(comp_level[1:]) & (low < lvl + 40.0)
            comp_level[1:][sel] = lvl
        lake_of = comp_level[lab]
        lake_of[lab == 0] = np.nan
    dem = np.where(np.isfinite(lake_of), lake_of - LAKE_FLOOR_DEPTH, filled)

    # noise in the lakes (e.g. in front of calving fronts): lower small bumps
    # to the values around them, only near the lakes and below lake level + 80 m
    k = max(3, int(round(15 / stride)) | 1)
    it = max(2, int(round(25 / stride)))
    opened = ndimage.grey_opening(np.where(glac, 1e4, dem), size=(k, k))
    for lvl in LAKES.values():
        near = ndimage.binary_dilation(lake_of == lvl, iterations=it)
        zone = near & ~glac & (dem < lvl + 80.0)
        dem = np.where(zone, np.minimum(dem, opened), dem)
        sunk = zone & (dem < lvl) & ~np.isfinite(lake_of)
        lake_of[sunk] = lvl

    ice_surf = np.where(glac & np.isfinite(thk), bed + thk, dem)
    thk_show = np.where(glac & np.isfinite(thk), thk, np.nan)
    return dict(tr=tr, shape=shape, bed=bed, dem=dem, glac=glac, gid=gid,
                lake_of=lake_of, surf=ice_surf, thk=thk_show)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
class Frame:
    """Local km coordinates with vertical exaggeration."""

    def __init__(self, x0, y0):
        self.x0, self.y0 = x0, y0

    def grid(self, d, z):
        rows, cols = np.mgrid[0:d["shape"][0], 0:d["shape"][1]]
        tr = d["tr"]
        E = tr.c + (cols + 0.5) * tr.a
        N = tr.f + (rows + 0.5) * tr.e
        return pv.StructuredGrid((E - self.x0) / 1000.0, (N - self.y0) / 1000.0,
                                 np.asarray(z) / 1000.0 * VE)


def cells(grid, mask, invert=False):
    g = grid.copy()
    g["m"] = mask.astype(float).ravel(order="F")
    g = g.point_data_to_cell_data(pass_point_data=True)
    out = g.threshold(0.999, scalars="m", preference="cell", invert=invert)
    return out.extract_surface(algorithm="dataset_surface").compute_normals(split_vertices=False)


def water_meshes(frame, d, glacier_levels):
    """Flat water at each lake level: the lake itself plus glacier bed below that level."""
    meshes = []
    for lvl in LAKES.values():
        m = (d["lake_of"] == lvl)
        for name, gl in glacier_levels.items():
            if gl == lvl:
                i = ORDER.index(name) + 1
                m |= (d["gid"] == i) & (d["bed"] < lvl)
        if m.sum() < 4:
            continue
        g = frame.grid(d, np.full(d["shape"], lvl))
        meshes.append(cells(g, m))
    return meshes


def world_to_px(pl, p):
    c = vtk.vtkCoordinate()
    c.SetCoordinateSystemToWorld()
    c.SetValue(*p)
    x, y = c.GetComputedDoubleDisplayValue(pl.renderer)
    return x, pl.window_size[1] - y


def aim(pl, bounds, az, el, zoom):
    """Look from azimuth az (deg, from +x towards +y) and elevation el, fit to bounds."""
    cx, cy, cz = (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2
    a, e = np.radians(az), np.radians(el)
    pl.camera.focal_point = (cx, cy, cz)
    pl.camera.position = (cx + np.cos(e) * np.cos(a), cy + np.cos(e) * np.sin(a), cz + np.sin(e))
    pl.camera.up = (0, 0, 1)
    pl.camera.view_angle = 25.0
    pl.reset_camera(bounds=bounds)
    pl.camera.zoom(zoom)
    pl.reset_camera_clipping_range()


def trim(img, labels=None, pad=12):
    """Crop the white background around a render; shift label pixel positions to match."""
    rgb = img[..., :3].astype(int)
    content = np.abs(rgb - 255).sum(axis=2) > 12
    rows, cols = np.nonzero(content)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, img.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, img.shape[1])
    if labels is not None:
        labels = {k: (x - c0, y - r0) for k, (x, y) in labels.items()}
    return img[r0:r1, c0:c1], labels


def plotter(size):
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(BG)
    pl.enable_depth_peeling(number_of_peels=6, occlusion_ratio=0.0)
    pl.enable_anti_aliasing("fxaa")
    return pl


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def overview(polys, size=(3600, 1800)):
    b = [np.inf, np.inf, -np.inf, -np.inf]
    for p in polys.values():
        pb = p.bounds
        b = [min(b[0], pb[0]), min(b[1], pb[1]), max(b[2], pb[2]), max(b[3], pb[3])]
    bounds = (b[0] - 3000, b[1] - 3000, b[2] + 9000, b[3] + 3000)    # more room on the lake side (east)
    d = prepare(bounds, stride=6, polys=polys)
    fr = Frame((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

    pl = plotter(size)
    surf = fr.grid(d, d["surf"])
    surf["thk"] = np.nan_to_num(d["thk"], nan=-1).ravel(order="F")
    ice = cells(surf, d["glac"])
    terr = cells(surf, d["glac"], invert=True)
    pl.add_mesh(terr, color=TERRAIN, smooth_shading=True, ambient=0.25, diffuse=0.8)
    pl.add_mesh(ice, scalars="thk", preference="point", cmap=THK_CMAP, clim=THK_CLIM,
                smooth_shading=True, show_scalar_bar=False, ambient=0.3, specular=0.15)
    for w in water_meshes(fr, d, {}):
        pl.add_mesh(w, color=WATER, opacity=0.85, ambient=0.4, specular=0.3)

    aim(pl, surf.bounds, az=-18.0, el=30.0, zoom=2.1)
    pl.show(auto_close=False, interactive=False)
    img = pl.screenshot(return_img=True)

    # label positions: above each glacier's lower part, and on the lakes
    labels = {}
    for name in ORDER:
        g = d["gid"] == ORDER.index(name) + 1
        low = g & (d["bed"] < np.nanpercentile(d["bed"][g], 25))
        r, c = np.nonzero(low)
        e = d["tr"].c + (c.mean() + 0.5) * d["tr"].a
        n = d["tr"].f + (r.mean() + 0.5) * d["tr"].e
        labels[name] = world_to_px(pl, ((e - fr.x0) / 1000, (n - fr.y0) / 1000, 2.2 * VE))
    for lake, (e, n) in {"Lago Argentino": (645000, 4440000), "Lago Viedma": (645000, 4513000)}.items():
        labels[lake] = world_to_px(pl, ((e - fr.x0) / 1000, (n - fr.y0) / 1000, LAKES[lake] / 1000 * VE))
    pl.close()
    return trim(img, labels)


def glacier_panel(name, polys, gpr, size=(1300, 1150)):
    rgi, lake_name = GLACIERS[name]
    lvl = LAKES[lake_name]
    poly = polys[name]

    # whole-glacier statistics at full resolution
    with rasterio.open(BED_TIF) as src:
        win = from_bounds(*poly.bounds, src.transform).round_offsets().round_lengths()
        bed_all = src.read(1, window=win, masked=True).astype("float64").filled(np.nan)
        tr_all = src.window_transform(win)
    ins = geometry_mask([poly], out_shape=bed_all.shape, transform=tr_all, invert=True) & np.isfinite(bed_all)
    px = abs(tr_all.a * tr_all.e)
    area, below = ins.sum() * px / 1e6, (ins & (bed_all < lvl)).sum() * px / 1e6
    rows, cols = np.nonzero(ins & (bed_all < lvl))
    be = tr_all.c + (cols + 0.5) * tr_all.a
    bn = tr_all.f + (rows + 0.5) * tr_all.e

    # scene: the map frame of the manuscript's bedrock figure
    g = gpr[gpr["Glacier"] == name]
    lon0, lat0, lon1, lat1 = REGIONS[name]
    bounds = transform_bounds("EPSG:4326", "EPSG:32718", lon0, lat0, lon1, lat1)
    d = prepare(bounds, stride=2, polys=polys)
    # camera target: the part of the bed below lake level inside the frame
    inside = (be >= bounds[0]) & (be <= bounds[2]) & (bn >= bounds[1]) & (bn <= bounds[3])
    be, bn = be[inside], bn[inside]
    fr = Frame((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

    pl = plotter(size)
    z = np.where(d["glac"], d["bed"], d["dem"])
    grid = fr.grid(d, z)
    grid["elev"] = z.ravel(order="F")
    bedm = cells(grid, d["glac"])
    terr = cells(grid, d["glac"], invert=True)
    pl.add_mesh(terr, color=TERRAIN, smooth_shading=True, ambient=0.25, diffuse=0.8)
    pl.add_mesh(bedm, scalars="elev", preference="point", cmap=BED_CMAP, clim=BED_CLIM,
                smooth_shading=True, show_scalar_bar=False, ambient=0.3, specular=0.1)

    levels = np.arange(np.floor(np.nanmin(z[d["glac"]]) / CONTOUR_STEP) * CONTOUR_STEP,
                       np.nanmax(z[d["glac"]]), CONTOUR_STEP)
    iso = bedm.contour(list(levels), scalars="elev", preference="point")
    if iso.n_points:
        iso.points[:, 2] += 0.004
        pl.add_mesh(iso, color="#2b2b2b", line_width=1.0, opacity=0.55)
    shore = bedm.contour([lvl], scalars="elev", preference="point")
    if shore.n_points:
        shore.points[:, 2] += 0.006
        pl.add_mesh(shore, color=LAKE_LINE, line_width=3.5)

    for w in water_meshes(fr, d, {name: lvl}):
        pl.add_mesh(w, color=WATER, opacity=0.45, ambient=0.4, specular=0.3)

    # look up-glacier from the lake: from the submerged part towards the glacier centre
    c = poly.centroid
    fx, fy = be.mean() if be.size else g.Easting.mean(), bn.mean() if bn.size else g.Northing.mean()
    az = np.degrees(np.arctan2(fy - c.y, fx - c.x))
    aim(pl, grid.bounds, az=az, el=38.0, zoom=1.25)
    pl.show(auto_close=False, interactive=False)
    img = pl.screenshot(return_img=True)
    pl.close()
    print(f"{name}: bed below {lvl:.0f} m a.s.l.: {below:.1f} of {area:.1f} km2 ({100 * below / area:.1f} %)")
    return trim(img)[0], dict(level=lvl, lake=lake_name, below=below, area=area)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def main():
    import pandas as pd

    outl = gpd.read_file(OUTLINES).to_crs(32718)
    polys = {}
    for name, (rgi, _) in GLACIERS.items():
        p = outl.loc[outl["rgi_id"] == rgi, "geometry"].iloc[0]
        polys[name] = p if p.geom_type == "Polygon" else max(p.geoms, key=lambda q: q.area)

    gpr = pd.read_csv(CSV, usecols=["Easting", "Northing", "Glacier"])
    gpr["Glacier"] = gpr["Glacier"].astype(str).str.strip()

    ov, ov_labels = overview(polys)
    panels = {name: glacier_panel(name, polys, gpr) for name in ORDER}

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "mathtext.fontset": "stix"})
    # row heights follow the cropped renders, so there is no empty space
    W = 7.1                                    # in, double-column width
    h_top = W * ov.shape[0] / ov.shape[1]
    h_row = max(W / 3 * img.shape[0] / img.shape[1] for img, _ in panels.values())
    title_h, text_h, cbar_h = 0.22, 0.34, 0.55
    H = h_top + title_h + h_row + text_h + cbar_h
    fig = plt.figure(figsize=(W, H))

    def box(x, y_top, w, h):              # inches from the top-left -> figure fraction
        return [x / W, 1 - (y_top + h) / H, w / W, h / H]

    ax = fig.add_axes(box(0, 0, W, h_top))
    ax.imshow(ov)
    ax.set_axis_off()
    ax.text(0.005, 0.98, "(a)", transform=ax.transAxes, va="top", fontsize=11, fontweight="bold")
    for name, (x, y) in ov_labels.items():
        lake = name.startswith("Lago")
        ax.text(x, y, name if lake else f"Glaciar {name}", ha="center", va="center",
                fontsize=8, style="italic" if lake else "normal",
                color="#0b3d66" if lake else "black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    cax = ax.inset_axes([0.07, 0.86, 0.24, 0.035])
    cb = fig.colorbar(ScalarMappable(Normalize(*THK_CLIM), THK_CMAP), cax=cax, orientation="horizontal",
                      extend="max")
    cb.set_label("Ice thickness (m)", fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=7)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_num(v)))

    y = h_top + title_h
    for i, name in enumerate(ORDER):
        img, s = panels[name]
        fig.text((i + 0.02) / 3, 1 - (h_top + 0.04) / H, f"({'bcd'[i]}) Glaciar {name}",
                 va="top", fontsize=9, fontweight="bold")
        ax = fig.add_axes(box(i * W / 3, y, W / 3, h_row))
        ax.imshow(img)
        ax.set_axis_off()
        fig.text((i + 0.02) / 3, 1 - (y + h_row + 0.03) / H,
                 f"{fmt_num(s['below'], 1)} km\u00b2 of {fmt_num(s['area'], 0)} km\u00b2 "
                 f"({fmt_num(100 * s['below'] / s['area'], 0)} %) below "
                 f"{fmt_num(s['level'])} m a.s.l.\n({s['lake']})",
                 va="top", fontsize=7.5)

    cax = fig.add_axes(box(W * 0.2, y + h_row + text_h + 0.05, W * 0.6, 0.09))
    cb = fig.colorbar(ScalarMappable(Normalize(*BED_CLIM), BED_CMAP), cax=cax, orientation="horizontal",
                      extend="both")
    cb.set_label(f"Bedrock elevation (m a.s.l.); contours every {fmt_num(CONTOUR_STEP)} m, "
                 f"bold blue line and water: lake level", fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=7)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_num(v)))
    fig.text(0.995, 0.005, f"vertical exaggeration {VE:.0f}\u00d7", ha="right", va="bottom",
             fontsize=7, color="#555555")

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300)
    print("wrote", OUT.with_suffix(".png"), "and .pdf")


if __name__ == "__main__":
    main()
