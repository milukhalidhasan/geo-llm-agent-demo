"""Checks that run before and after every tool call.

They look for common mistakes in geospatial workflows:

* GEOGRAPHIC_CRS   - measuring area/length or buffering a layer in degrees
* CRS_MISMATCH     - joining or clipping two layers stored in different CRSs
* INVALID_GEOMETRY - input layer has invalid (e.g. self-intersecting) polygons
* EMPTY_RESULT     - an operation returned no features, or a zero measurement
* UNIT_SANITY      - a value that is implausible for a field-scale study area

Modes (set per agent run):

* ``off``   - checks still run and are written to the trace, but the model never
              sees them (used as a baseline in the evaluation)
* ``warn``  - warnings are added to the tool result the model reads
* ``block`` - like ``warn``, but a pre-check with severity "error" stops the
              tool from running, so the model has to fix the step
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .tools import Workspace

MODES = ("off", "warn", "block")

# Plausible limits for this demo's study area (a few hundred metres across).
MAX_PLAUSIBLE_AREA_HA = 1_000.0
MAX_PLAUSIBLE_LENGTH_KM = 100.0
MAX_PLAUSIBLE_BUFFER_M = 50_000.0


@dataclass
class Finding:
    code: str
    severity: str  # "warning" or "error"
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _layer(ws: Workspace, name: Any):
    return ws.layers.get(name) if isinstance(name, str) else None


def _is_geographic(gdf) -> bool:
    return gdf is not None and gdf.crs is not None and gdf.crs.is_geographic


def pre_checks(ws: Workspace, tool: str, args: dict[str, Any]) -> list[Finding]:
    """Checks on the inputs, before the tool runs."""
    out: list[Finding] = []
    if tool in {"measure", "buffer"}:
        gdf = _layer(ws, args.get("layer"))
        if _is_geographic(gdf):
            what = "Buffer distance" if tool == "buffer" else f"{args.get('quantity', 'measurement')}"
            out.append(Finding(
                "GEOGRAPHIC_CRS", "error",
                f"Layer '{args.get('layer')}' is in a geographic CRS ({gdf.crs.to_string()}, units: degrees). "
                f"{what} would be computed in degrees, not metres. Reproject to a projected CRS first "
                f"(e.g. EPSG:32646, UTM zone 46N).",
            ))
    if tool in {"spatial_join", "clip"}:
        a = _layer(ws, args.get("left") or args.get("layer"))
        b = _layer(ws, args.get("right") or args.get("mask_layer"))
        if a is not None and b is not None and a.crs != b.crs:
            out.append(Finding(
                "CRS_MISMATCH", "error",
                f"Input layers have different CRSs ({a.crs.to_string() if a.crs else 'none'} vs "
                f"{b.crs.to_string() if b.crs else 'none'}). Reproject one so both match before this operation.",
            ))
    if tool in {"measure", "buffer", "spatial_join", "clip", "zonal_stats"}:
        for key in ("layer", "left", "right", "mask_layer"):
            gdf = _layer(ws, args.get(key))
            if gdf is not None and len(gdf):
                n_bad = int((~gdf.is_valid).sum())
                if n_bad:
                    out.append(Finding(
                        "INVALID_GEOMETRY", "warning",
                        f"Layer '{args[key]}' has {n_bad} invalid geometr{'y' if n_bad == 1 else 'ies'}; "
                        f"areas and overlays may be wrong. Consider fix_geometries.",
                    ))
    if tool == "buffer":
        try:
            d = float(args.get("distance"))
        except (TypeError, ValueError):
            d = None
        gdf = _layer(ws, args.get("layer"))
        if d is not None and not _is_geographic(gdf):
            if d <= 0:
                out.append(Finding("UNIT_SANITY", "warning", f"Buffer distance {d} is not positive."))
            elif d > MAX_PLAUSIBLE_BUFFER_M:
                out.append(Finding("UNIT_SANITY", "warning", f"Buffer distance {d} m is larger than 50 km; check units."))
    return out


def post_checks(ws: Workspace, tool: str, args: dict[str, Any], result: dict[str, Any]) -> list[Finding]:
    """Checks on the result, after the tool ran."""
    out: list[Finding] = []
    n = result.get("n_features")
    if tool in {"filter_features", "spatial_join", "clip", "buffer", "reproject"} and n == 0:
        out.append(Finding(
            "EMPTY_RESULT", "warning",
            f"{tool} returned 0 features. Check the filter value, the predicate, or whether the layers "
            f"overlap (same CRS?).",
        ))
    if tool == "measure":
        val = result.get("value")
        unit = result.get("unit")
        if val is not None:
            if val == 0:
                out.append(Finding("EMPTY_RESULT", "warning", "Measured value is 0."))
            elif result.get("quantity") == "area":
                ha = val * {"m2": 1e-4, "ha": 1.0, "km2": 100.0}.get(unit, 1.0)
                if ha > MAX_PLAUSIBLE_AREA_HA:
                    out.append(Finding("UNIT_SANITY", "warning", f"Area {ha:.3g} ha is implausibly large for this study area."))
                elif ha < 1e-6 * max(result.get("n_features", 1), 1):
                    out.append(Finding("UNIT_SANITY", "warning",
                                       f"Area {val:.3g} {unit} is implausibly small; was it measured in degrees?"))
            elif result.get("quantity") == "length":
                km = val * {"m": 1e-3, "km": 1.0}.get(unit, 1.0)
                if km > MAX_PLAUSIBLE_LENGTH_KM:
                    out.append(Finding("UNIT_SANITY", "warning", f"Length {km:.3g} km is implausibly large."))
                elif km < 1e-6:
                    out.append(Finding("UNIT_SANITY", "warning",
                                       f"Length {val:.3g} {unit} is implausibly small; was it measured in degrees?"))
    if tool == "zonal_stats" and result.get("n_missing"):
        out.append(Finding("EMPTY_RESULT", "warning",
                           f"{result['n_missing']} polygon(s) got no raster value (outside the raster?)."))
    return out


def findings_to_dicts(findings: list[Finding]) -> list[dict[str, str]]:
    return [f.to_dict() for f in findings]
