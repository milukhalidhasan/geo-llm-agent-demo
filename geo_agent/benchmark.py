"""Benchmark questions and their reference answers.

Reference answers are computed here with plain GeoPandas, written by hand and
independent of the agent's tools, always in UTM 46N (metres) and after
repairing invalid geometries. Each question also lists which trap it tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from .data import DATA_DIR, UTM


@dataclass
class Question:
    qid: str
    text: str
    kind: str  # "number" or "name"
    reference: Callable[[dict[str, Any]], Any]
    rel_tol: float = 0.0  # relative tolerance for numbers (0 = exact)
    tests: str = ""  # what the question is designed to test


@lru_cache(maxsize=1)
def _layers() -> dict[str, Any]:
    d = {
        "plots": gpd.read_file(DATA_DIR / "farm_plots.geojson"),
        "villages": gpd.read_file(DATA_DIR / "villages.gpkg"),
        "roads": gpd.read_file(DATA_DIR / "roads.geojson"),
        "canal": gpd.read_file(DATA_DIR / "canal.gpkg"),
        "samples": gpd.read_file(DATA_DIR / "soil_samples.geojson"),
    }
    utm = {k: v.to_crs(UTM) for k, v in d.items()}
    utm["plots"] = utm["plots"].assign(geometry=utm["plots"].make_valid())
    utm["raw"] = d
    with rasterio.open(DATA_DIR / "ndvi_10m.tif") as src:
        means = []
        for g in utm["plots"].to_crs(src.crs).geometry:
            arr, _ = rio_mask(src, [g], crop=True, filled=False)
            means.append(float(np.mean(arr.compressed())))
    utm["plots"]["ndvi_mean"] = means
    return utm


def _ha(gdf) -> float:
    return float(gdf.area.sum() / 1e4)


QUESTIONS: list[Question] = [
    Question("Q01", "How many farm plots are in the dataset?", "number",
             lambda L: len(L["plots"]), tests="basic count"),
    Question("Q02", "What is the EPSG code of the coordinate reference system of the plots layer?", "number",
             lambda L: L["raw"]["plots"].crs.to_epsg(), tests="CRS inspection"),
    Question("Q03", "How many farm plots have an invalid geometry?", "number",
             lambda L: int((~L["raw"]["plots"].is_valid).sum()), tests="geometry validity"),
    Question("Q04", "What is the total area of rice plots in hectares?", "number",
             lambda L: _ha(L["plots"][L["plots"].crop == "rice"]), rel_tol=0.02,
             tests="filter + area; plots stored in degrees"),
    Question("Q05", "What is the total length of all roads in kilometres?", "number",
             lambda L: float(L["roads"].length.sum() / 1e3), rel_tol=0.02,
             tests="length; roads stored in degrees"),
    Question("Q06", "How many farm plots lie within 50 metres of the canal (touching the 50 m buffer counts)?",
             "number", lambda L: int(L["plots"].intersects(L["canal"].buffer(50).union_all()).sum()),
             tests="buffer in metres + overlay; canal and plots in different CRSs"),
    Question("Q07", "Which village contains the most farm plots?", "name",
             lambda L: gpd.sjoin(L["plots"], L["villages"], predicate="within").village.value_counts().idxmax(),
             tests="spatial join + group count; CRS mismatch"),
    Question("Q08", "What is the total area in hectares of the farm plots in Kaliakoir village?", "number",
             lambda L: _ha(gpd.sjoin(L["plots"], L["villages"][L["villages"].village == "Kaliakoir"],
                                     predicate="within")), rel_tol=0.02,
             tests="filter + join + area; includes the invalid plot"),
    Question("Q09", "How many soil samples fall inside wheat plots?", "number",
             lambda L: len(gpd.sjoin(L["samples"], L["plots"][L["plots"].crop == "wheat"], predicate="within")),
             tests="point-in-polygon"),
    Question("Q10", "What is the mean area in hectares of the plots that received at least 120 kg N/ha?", "number",
             lambda L: float(L["plots"][L["plots"].n_rate >= 120].area.mean() / 1e4), rel_tol=0.02,
             tests="numeric filter + mean area"),
    Question("Q11", "What is the average of the mean NDVI values of the wheat plots?", "number",
             lambda L: float(L["plots"][L["plots"].crop == "wheat"].ndvi_mean.mean()), rel_tol=0.02,
             tests="zonal statistics + aggregation"),
    Question("Q12", "Which plot (plot_id) has the highest mean NDVI?", "name",
             lambda L: str(L["plots"].loc[L["plots"].ndvi_mean.idxmax(), "plot_id"]),
             tests="zonal statistics + argmax"),
]


def reference_answers() -> dict[str, Any]:
    L = _layers()
    return {q.qid: q.reference(L) for q in QUESTIONS}


_NUM = re.compile(r"-?\d+(?:[.,]\d+)*(?:[eE]-?\d+)?")


def parse_number(text: str | None) -> float | None:
    """First number in the answer text ('1,234.5 ha' -> 1234.5)."""
    if not text:
        return None
    m = _NUM.search(text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def is_correct(q: Question, answer_text: str | None, reference: Any) -> bool | None:
    """True/False, or None when no answer could be parsed."""
    if answer_text is None:
        return None
    if q.kind == "name":
        return re.search(rf"\b{re.escape(str(reference))}\b", answer_text, re.I) is not None
    val = parse_number(answer_text)
    if val is None:
        return None
    ref = float(reference)
    if q.rel_tol == 0:
        return abs(val - ref) < 1e-9
    return abs(val - ref) <= q.rel_tol * abs(ref)
