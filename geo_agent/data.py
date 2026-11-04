"""Generate the small synthetic study area used by the demo.

The data imitate a field station near Gazipur, Bangladesh: farm plots, three
village boundaries, roads, an irrigation canal, soil sampling points and a
10 m NDVI raster. Everything is made up and generated from a fixed random seed,
so the benchmark answers are always the same and the demo runs offline.

Some layers are stored in different coordinate reference systems on purpose,
and one plot has a self-intersecting (invalid) polygon. These are the traps the
agent's checks are meant to catch.

Run:  python -m geo_agent.data
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon, box

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UTM = "EPSG:32646"  # WGS 84 / UTM zone 46N, metres
WGS84 = "EPSG:4326"  # longitude/latitude, degrees

# South-west corner of the study area in UTM 46N (roughly 24.03 N, 90.40 E).
X0, Y0 = 236_000.0, 2_660_000.0
CROPS = ["rice", "wheat", "maize", "rice", "wheat", "rice"]
N_RATES = [0, 60, 120, 180]
NDVI_BY_CROP = {"rice": 0.62, "wheat": 0.55, "maize": 0.70, "fallow": 0.18}


def make_plots(rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Return 24 farm plots (6 columns x 4 rows) in UTM 46N."""
    rows = []
    pid = 1
    for r in range(4):
        for c in range(6):
            w = 60 + rng.uniform(-8, 8)
            h = 40 + rng.uniform(-6, 6)
            x = X0 + 100 + c * 90
            y = Y0 + 100 + r * 70
            crop = CROPS[(c + r) % len(CROPS)]
            rows.append(
                {
                    "plot_id": f"P{pid:02d}",
                    "crop": crop,
                    "n_rate": N_RATES[(c + 2 * r) % len(N_RATES)],
                    "geometry": box(x, y, x + w, y + h),
                }
            )
            pid += 1
    # The last plot is fallow and drawn as a "bow-tie": an invalid polygon
    # whose edges cross. Shapely reports its area as ~0 until it is repaired.
    x, y = X0 + 100 + 5 * 90, Y0 + 100 + 3 * 70
    rows[-1]["crop"] = "fallow"
    rows[-1]["n_rate"] = 0
    rows[-1]["geometry"] = Polygon([(x, y), (x + 60, y + 40), (x + 60, y), (x, y + 40)])
    return gpd.GeoDataFrame(rows, crs=UTM)


def make_villages() -> gpd.GeoDataFrame:
    """Three village boundaries that split the plot grid west to east."""
    ext_y0, ext_y1 = Y0, Y0 + 450
    rows = [
        {"village": "Salna", "geometry": box(X0, ext_y0, X0 + 175, ext_y1)},
        {"village": "Bhawal", "geometry": box(X0 + 175, ext_y0, X0 + 445, ext_y1)},
        {"village": "Kaliakoir", "geometry": box(X0 + 445, ext_y0, X0 + 700, ext_y1)},
    ]
    return gpd.GeoDataFrame(rows, crs=UTM)


def make_roads() -> gpd.GeoDataFrame:
    rows = [
        {"road_id": "R1", "type": "paved", "geometry": LineString([(X0, Y0 + 60), (X0 + 700, Y0 + 75)])},
        {"road_id": "R2", "type": "earthen", "geometry": LineString([(X0 + 320, Y0), (X0 + 330, Y0 + 450)])},
        {"road_id": "R3", "type": "earthen", "geometry": LineString([(X0 + 50, Y0 + 400), (X0 + 380, Y0 + 430), (X0 + 690, Y0 + 410)])},
    ]
    return gpd.GeoDataFrame(rows, crs=UTM)


def make_canal() -> gpd.GeoDataFrame:
    line = LineString([(X0 + 20, Y0 + 445), (X0 + 260, Y0 + 300), (X0 + 520, Y0 + 290), (X0 + 700, Y0 + 180)])
    return gpd.GeoDataFrame([{"name": "Canal A", "geometry": line}], crs=UTM)


def make_samples(rng: np.random.Generator) -> gpd.GeoDataFrame:
    xs = rng.uniform(X0 + 80, X0 + 660, 40)
    ys = rng.uniform(Y0 + 80, Y0 + 420, 40)
    rows = [
        {"sample_id": f"S{i + 1:02d}", "soc_pct": round(float(rng.uniform(0.6, 1.8)), 2), "geometry": Point(x, y)}
        for i, (x, y) in enumerate(zip(xs, ys))
    ]
    return gpd.GeoDataFrame(rows, crs=UTM)


def make_ndvi(plots: gpd.GeoDataFrame, rng: np.random.Generator) -> tuple[np.ndarray, object]:
    """10 m NDVI raster: background ~0.30, each plot near its crop's mean."""
    res = 10.0
    width, height = 70, 45  # 700 m x 450 m
    transform = from_origin(X0, Y0 + height * res, res, res)
    arr = 0.30 + rng.normal(0, 0.02, (height, width))
    rows_idx, cols_idx = np.mgrid[0:height, 0:width]
    xs = X0 + (cols_idx + 0.5) * res
    ys = Y0 + height * res - (rows_idx + 0.5) * res
    for _, p in plots.iterrows():
        geom = p.geometry if p.geometry.is_valid else p.geometry.buffer(0)
        minx, miny, maxx, maxy = geom.bounds
        inside = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        bump = NDVI_BY_CROP[p.crop] + p.n_rate / 180 * 0.08
        arr[inside] = bump + rng.normal(0, 0.02, inside.sum())
    return np.clip(arr, -1, 1).astype("float32"), transform


def generate(out_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Write all layers to ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    plots = make_plots(rng)
    villages = make_villages()
    roads = make_roads()
    canal = make_canal()
    samples = make_samples(rng)
    ndvi, transform = make_ndvi(plots, rng)

    paths = {
        # Plots, roads and samples are stored in lon/lat (degrees): measuring
        # area or distance on them directly gives meaningless numbers.
        "plots": out_dir / "farm_plots.geojson",
        "roads": out_dir / "roads.geojson",
        "samples": out_dir / "soil_samples.geojson",
        # Villages and canal stay in UTM 46N (metres), so joins with the
        # lon/lat layers need reprojection first.
        "villages": out_dir / "villages.gpkg",
        "canal": out_dir / "canal.gpkg",
        "ndvi": out_dir / "ndvi_10m.tif",
    }
    plots.to_crs(WGS84).to_file(paths["plots"], driver="GeoJSON")
    roads.to_crs(WGS84).to_file(paths["roads"], driver="GeoJSON")
    samples.to_crs(WGS84).to_file(paths["samples"], driver="GeoJSON")
    villages.to_file(paths["villages"], driver="GPKG")
    canal.to_file(paths["canal"], driver="GPKG")
    with rasterio.open(
        paths["ndvi"], "w", driver="GTiff", height=ndvi.shape[0], width=ndvi.shape[1],
        count=1, dtype="float32", crs=UTM, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(ndvi, 1)
    return paths


if __name__ == "__main__":
    for name, path in generate().items():
        print(f"{name:9s} -> {path}")
