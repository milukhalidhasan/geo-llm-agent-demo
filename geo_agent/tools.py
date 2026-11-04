"""GIS tools the agent can call.

Each tool is a plain Python function that works on named layers held in a
``Workspace``. Tools return small JSON-serialisable dictionaries (never whole
datasets) so that the result can be passed back to the language model.

The tools deliberately behave like naive GeoPandas code: they do *not* fix the
user's mistakes. For example, ``measure`` on a layer in longitude/latitude
returns a number in squared degrees, converted as if it were square metres.
Catching such mistakes is the job of ``checks.py``, so the two can be
evaluated separately.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask

from .data import DATA_DIR

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

DEFAULT_LAYERS = {
    "plots": "farm_plots.geojson",
    "villages": "villages.gpkg",
    "roads": "roads.geojson",
    "canal": "canal.gpkg",
    "samples": "soil_samples.geojson",
}
DEFAULT_RASTERS = {"ndvi": "ndvi_10m.tif"}

AREA_UNITS = {"m2": 1.0, "ha": 1e4, "km2": 1e6}
LENGTH_UNITS = {"m": 1.0, "km": 1e3}


class ToolError(Exception):
    """Raised when a tool is called with bad arguments (unknown layer etc.)."""


@dataclass
class Workspace:
    """Named vector layers and raster paths available to the agent."""

    layers: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    rasters: dict[str, Path] = field(default_factory=dict)
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR

    @classmethod
    def default(cls, data_dir: Path = DATA_DIR) -> "Workspace":
        ws = cls(data_dir=data_dir)
        for name, fname in DEFAULT_LAYERS.items():
            ws.layers[name] = gpd.read_file(data_dir / fname)
        for name, fname in DEFAULT_RASTERS.items():
            ws.rasters[name] = data_dir / fname
        return ws

    def get(self, name: str) -> gpd.GeoDataFrame:
        if name not in self.layers:
            raise ToolError(f"No layer called '{name}'. Available layers: {sorted(self.layers)}")
        return self.layers[name]

    def put(self, name: str, gdf: gpd.GeoDataFrame) -> None:
        if not name or not str(name).strip():
            raise ToolError("out_name must be a non-empty string")
        self.layers[str(name)] = gdf


def _crs_label(gdf: gpd.GeoDataFrame) -> str:
    if gdf.crs is None:
        return "none"
    epsg = gdf.crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else gdf.crs.name


def _crs_info(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    if gdf.crs is None:
        return {"crs": "none", "geographic": None, "units": None}
    units = gdf.crs.axis_info[0].unit_name if gdf.crs.axis_info else None
    return {"crs": _crs_label(gdf), "geographic": bool(gdf.crs.is_geographic), "units": units}


def _layer_summary(name: str, gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    return {"layer": name, "n_features": int(len(gdf)), **_crs_info(gdf)}


def _py(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python for JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


# ---------------------------------------------------------------- tools


def list_layers(ws: Workspace) -> dict[str, Any]:
    """List the vector layers and rasters currently in the workspace."""
    return {
        "layers": [_layer_summary(n, g) for n, g in ws.layers.items()],
        "rasters": sorted(ws.rasters),
    }


def load_layer(ws: Workspace, filename: str, out_name: str) -> dict[str, Any]:
    """Load a vector file from the data folder into the workspace."""
    path = (ws.data_dir / filename).resolve()
    if ws.data_dir.resolve() not in path.parents or not path.exists():
        raise ToolError(f"File '{filename}' not found in the data folder")
    gdf = gpd.read_file(path)
    ws.put(out_name, gdf)
    return _layer_summary(out_name, gdf)


def inspect_layer(ws: Workspace, layer: str) -> dict[str, Any]:
    """Describe a layer: CRS, geometry types, columns, invalid geometries, a few rows."""
    gdf = ws.get(layer)
    cols = [c for c in gdf.columns if c != gdf.geometry.name]
    sample = gdf[cols].head(3).to_dict(orient="records")
    values = {}
    for c in cols:
        if not pd.api.types.is_numeric_dtype(gdf[c]) and gdf[c].nunique() <= 10:
            values[c] = sorted(map(str, gdf[c].dropna().unique()))
    return {
        **_layer_summary(layer, gdf),
        "geometry_types": sorted(gdf.geom_type.dropna().unique().tolist()),
        "columns": {c: str(gdf[c].dtype) for c in cols},
        "categorical_values": values,
        "n_invalid_geometries": int((~gdf.is_valid).sum()),
        "bounds": [round(float(b), 6) for b in gdf.total_bounds],
        "first_rows": [{k: _py(v) for k, v in r.items()} for r in sample],
    }


def reproject(ws: Workspace, layer: str, epsg: int, out_name: str) -> dict[str, Any]:
    """Reproject a layer to another CRS given by its EPSG code (e.g. 32646 for UTM 46N)."""
    gdf = ws.get(layer)
    try:
        out = gdf.to_crs(epsg=int(epsg))
    except Exception as exc:  # pyproj errors are not very friendly
        raise ToolError(f"Could not reproject to EPSG:{epsg}: {exc}") from exc
    ws.put(out_name, out)
    return _layer_summary(out_name, out)


def fix_geometries(ws: Workspace, layer: str, out_name: str) -> dict[str, Any]:
    """Repair invalid geometries with shapely's make_valid."""
    gdf = ws.get(layer)
    n_bad = int((~gdf.is_valid).sum())
    out = gdf.copy()
    out["geometry"] = out.geometry.make_valid()
    ws.put(out_name, out)
    return {**_layer_summary(out_name, out), "n_repaired": n_bad}


_OPS: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
}


def filter_features(ws: Workspace, layer: str, column: str, op: str, value: Any, out_name: str) -> dict[str, Any]:
    """Keep the features where `column op value` is true (op is one of == != > >= < <=)."""
    gdf = ws.get(layer)
    if column not in gdf.columns:
        raise ToolError(f"Column '{column}' not in layer '{layer}'. Columns: {list(gdf.columns)}")
    if op not in _OPS:
        raise ToolError(f"op must be one of {list(_OPS)}")
    series = gdf[column]
    if pd.api.types.is_numeric_dtype(series):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"Column '{column}' is numeric; value '{value}' is not a number") from exc
    else:
        series = series.astype(str).str.lower()
        value = str(value).lower()
    out = gdf[_OPS[op](series, value)].copy()
    ws.put(out_name, out)
    return _layer_summary(out_name, out)


def buffer(ws: Workspace, layer: str, distance: float, out_name: str) -> dict[str, Any]:
    """Buffer every feature by `distance`, given in the units of the layer's CRS."""
    gdf = ws.get(layer)
    import warnings

    out = gdf.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["geometry"] = gdf.geometry.buffer(float(distance))
    ws.put(out_name, out)
    return {**_layer_summary(out_name, out), "distance": float(distance)}


def spatial_join(ws: Workspace, left: str, right: str, predicate: str, out_name: str) -> dict[str, Any]:
    """Attach attributes of `right` to features of `left` where the predicate holds.

    predicate: intersects, within, contains. Only matching left features are kept
    (inner join); a left feature that matches two right features appears twice.
    """
    lg, rg = ws.get(left), ws.get(right)
    if predicate not in {"intersects", "within", "contains"}:
        raise ToolError("predicate must be intersects, within or contains")
    import warnings

    with warnings.catch_warnings():
        # GeoPandas warns about CRS mismatch; checks.py reports it instead.
        warnings.simplefilter("ignore")
        out = gpd.sjoin(lg, rg, how="inner", predicate=predicate)
    out = out.drop(columns=[c for c in out.columns if c.startswith("index_")])
    ws.put(out_name, out)
    return _layer_summary(out_name, out)


def clip(ws: Workspace, layer: str, mask_layer: str, out_name: str) -> dict[str, Any]:
    """Cut `layer` to the outline of `mask_layer`."""
    gdf, msk = ws.get(layer), ws.get(mask_layer)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = gpd.clip(gdf, msk)
    ws.put(out_name, out)
    return _layer_summary(out_name, out)


def measure(ws: Workspace, layer: str, quantity: str, unit: str, aggregate: str = "total") -> dict[str, Any]:
    """Measure area or length of features in the layer's current CRS.

    quantity: "area" (unit m2, ha or km2) or "length" (unit m or km).
    aggregate: "total", "mean" or "per_feature" (first 30 features, with an id column).
    """
    gdf = ws.get(layer)
    import warnings

    if quantity not in {"area", "length"}:
        raise ToolError("quantity must be 'area' or 'length'")
    factors = AREA_UNITS if quantity == "area" else LENGTH_UNITS
    if unit not in factors:
        raise ToolError(f"unit for {quantity} must be one of {list(factors)}")
    with warnings.catch_warnings():
        # GeoPandas warns about geographic CRS; checks.py reports it instead.
        warnings.simplefilter("ignore")
        raw = gdf.geometry.area if quantity == "area" else gdf.geometry.length
    values = raw / factors[unit]
    result: dict[str, Any] = {"layer": layer, "quantity": quantity, "unit": unit, "crs": _crs_label(gdf),
                              "n_features": int(len(gdf))}
    if aggregate == "total":
        result["value"] = float(values.sum())
    elif aggregate == "mean":
        result["value"] = float(values.mean()) if len(values) else None
    elif aggregate == "per_feature":
        id_col = next((c for c in gdf.columns if c.endswith("_id") or c in {"village", "name"}), None)
        ids = gdf[id_col].astype(str).tolist() if id_col else [str(i) for i in gdf.index]
        result["values"] = {i: round(float(v), 6) for i, v in list(zip(ids, values))[:30]}
    else:
        raise ToolError("aggregate must be total, mean or per_feature")
    return result


def count_features(ws: Workspace, layer: str) -> dict[str, Any]:
    """Count the features in a layer."""
    gdf = ws.get(layer)
    return {"layer": layer, "count": int(len(gdf))}


def summarise(ws: Workspace, layer: str, column: str, stat: str, group_by: str | None = None) -> dict[str, Any]:
    """Summarise an attribute column: stat is count, sum, mean, min, max or nunique.

    With `group_by`, return the statistic per group, sorted from largest to smallest.
    """
    gdf = ws.get(layer)
    for c in [column] + ([group_by] if group_by else []):
        if c not in gdf.columns:
            raise ToolError(f"Column '{c}' not in layer '{layer}'. Columns: {list(gdf.columns)}")
    if stat not in {"count", "sum", "mean", "min", "max", "nunique"}:
        raise ToolError("stat must be count, sum, mean, min, max or nunique")
    if group_by:
        res = gdf.groupby(group_by)[column].agg(stat).sort_values(ascending=False)
        return {"layer": layer, "column": column, "stat": stat, "group_by": group_by,
                "values": {str(k): _py(v) for k, v in res.items()}}
    return {"layer": layer, "column": column, "stat": stat, "value": _py(gdf[column].agg(stat))}


def zonal_stats(ws: Workspace, layer: str, raster: str, stat: str, out_name: str) -> dict[str, Any]:
    """Compute a raster statistic (mean, min, max, median) inside each polygon.

    The layer is reprojected to the raster's CRS internally. The result layer gets
    a new column named `<raster>_<stat>`.
    """
    gdf = ws.get(layer)
    if raster not in ws.rasters:
        raise ToolError(f"No raster called '{raster}'. Available: {sorted(ws.rasters)}")
    funcs = {"mean": np.mean, "min": np.min, "max": np.max, "median": np.median}
    if stat not in funcs:
        raise ToolError(f"stat must be one of {list(funcs)}")
    col = f"{raster}_{stat}"
    out = gdf.copy()
    vals = []
    with rasterio.open(ws.rasters[raster]) as src:
        geoms = gdf.to_crs(src.crs).geometry
        for geom in geoms:
            try:
                data, _ = rio_mask(src, [geom], crop=True, filled=False)
                arr = data.compressed()
                vals.append(float(funcs[stat](arr)) if arr.size else None)
            except ValueError:  # polygon outside the raster
                vals.append(None)
    out[col] = vals
    ws.put(out_name, out)
    id_col = next((c for c in gdf.columns if c.endswith("_id")), None)
    preview = dict(zip(out[id_col].astype(str), [None if v is None else round(v, 4) for v in vals])) if id_col else {}
    return {**_layer_summary(out_name, out), "new_column": col, "values": dict(list(preview.items())[:30]),
            "n_missing": int(sum(v is None for v in vals))}


def save_layer(ws: Workspace, layer: str, filename: str) -> dict[str, Any]:
    """Save a layer as GeoJSON in the outputs folder."""
    gdf = ws.get(layer)
    ws.output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name
    if not name.endswith(".geojson"):
        name += ".geojson"
    path = ws.output_dir / name
    gdf.to_file(path, driver="GeoJSON")
    return {"saved": str(path), "n_features": int(len(gdf))}


def plot_layer(ws: Workspace, layer: str, filename: str, column: str | None = None) -> dict[str, Any]:
    """Draw a layer (optionally coloured by a column) to a PNG in the outputs folder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gdf = ws.get(layer)
    ws.output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name
    if not name.endswith(".png"):
        name += ".png"
    fig, ax = plt.subplots(figsize=(6, 4))
    gdf.plot(ax=ax, column=column if column in gdf.columns else None, legend=bool(column), edgecolor="black")
    ax.set_title(f"{layer} ({_crs_label(gdf)})")
    fig.tight_layout()
    path = ws.output_dir / name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"saved": str(path)}


# ---------------------------------------------------------------- registry

_S = {"type": "string"}
_N = {"type": "number"}


def _schema(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "list_layers": list_layers,
    "load_layer": load_layer,
    "inspect_layer": inspect_layer,
    "reproject": reproject,
    "fix_geometries": fix_geometries,
    "filter_features": filter_features,
    "buffer": buffer,
    "spatial_join": spatial_join,
    "clip": clip,
    "measure": measure,
    "count_features": count_features,
    "summarise": summarise,
    "zonal_stats": zonal_stats,
    "save_layer": save_layer,
    "plot_layer": plot_layer,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema("list_layers", "List the vector layers and rasters available.", {}, []),
    _schema("load_layer", "Load a vector file from the data folder.",
            {"filename": _S, "out_name": _S}, ["filename", "out_name"]),
    _schema("inspect_layer", "Show a layer's CRS (and whether it is geographic), columns, category values, "
            "number of invalid geometries and first rows.", {"layer": _S}, ["layer"]),
    _schema("reproject", "Reproject a layer to a CRS given as an EPSG code (32646 = UTM zone 46N, metres).",
            {"layer": _S, "epsg": {"type": "integer"}, "out_name": _S}, ["layer", "epsg", "out_name"]),
    _schema("fix_geometries", "Repair invalid geometries.", {"layer": _S, "out_name": _S}, ["layer", "out_name"]),
    _schema("filter_features", "Keep features where `column op value` is true.",
            {"layer": _S, "column": _S, "op": {"type": "string", "enum": list(_OPS)},
             "value": {"type": ["string", "number"]}, "out_name": _S},
            ["layer", "column", "op", "value", "out_name"]),
    _schema("buffer", "Buffer features by a distance in the units of the layer's CRS.",
            {"layer": _S, "distance": _N, "out_name": _S}, ["layer", "distance", "out_name"]),
    _schema("spatial_join", "Inner spatial join: keep features of `left` matching `right`, with right's attributes.",
            {"left": _S, "right": _S, "predicate": {"type": "string", "enum": ["intersects", "within", "contains"]},
             "out_name": _S}, ["left", "right", "predicate", "out_name"]),
    _schema("clip", "Clip a layer to the outline of another layer.",
            {"layer": _S, "mask_layer": _S, "out_name": _S}, ["layer", "mask_layer", "out_name"]),
    _schema("measure", "Measure area (m2, ha, km2) or length (m, km) in the layer's current CRS.",
            {"layer": _S, "quantity": {"type": "string", "enum": ["area", "length"]},
             "unit": {"type": "string", "enum": ["m2", "ha", "km2", "m", "km"]},
             "aggregate": {"type": "string", "enum": ["total", "mean", "per_feature"]}},
            ["layer", "quantity", "unit"]),
    _schema("count_features", "Count the features in a layer.", {"layer": _S}, ["layer"]),
    _schema("summarise", "Statistic (count, sum, mean, min, max, nunique) of a column, optionally per group.",
            {"layer": _S, "column": _S, "stat": {"type": "string", "enum": ["count", "sum", "mean", "min", "max", "nunique"]},
             "group_by": _S}, ["layer", "column", "stat"]),
    _schema("zonal_stats", "Raster statistic (mean, min, max, median) inside each polygon; adds column <raster>_<stat>.",
            {"layer": _S, "raster": _S, "stat": {"type": "string", "enum": ["mean", "min", "max", "median"]},
             "out_name": _S}, ["layer", "raster", "stat", "out_name"]),
    _schema("save_layer", "Save a layer as GeoJSON in the outputs folder.",
            {"layer": _S, "filename": _S}, ["layer", "filename"]),
    _schema("plot_layer", "Draw a layer to a PNG in the outputs folder.",
            {"layer": _S, "filename": _S, "column": _S}, ["layer", "filename"]),
]

assert [s["function"]["name"] for s in TOOL_SCHEMAS] == list(TOOLS), "schemas and tools out of sync"


def call_tool(ws: Workspace, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name. Raises ToolError for unknown tools or bad arguments."""
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'. Available tools: {list(TOOLS)}")
    allowed = set(TOOL_SCHEMAS[list(TOOLS).index(name)]["function"]["parameters"]["properties"])
    extra = set(args) - allowed
    if extra:
        raise ToolError(f"Unexpected argument(s) for {name}: {sorted(extra)}. Allowed: {sorted(allowed)}")
    try:
        return TOOLS[name](ws, **args)
    except ToolError:
        raise
    except TypeError as exc:
        raise ToolError(f"Bad arguments for {name}: {exc}") from exc
    except Exception as exc:
        raise ToolError(f"{type(exc).__name__} in {name}: {exc}") from exc


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
