#!/usr/bin/env python3
"""Plot official NM OCD oil- and gas-well context around one HISUI candidate.

The script queries the New Mexico Energy, Minerals and Natural Resources
Department Oil Conservation Division (OCD) ArcGIS REST Oil Wells (0) and Gas
Wells (1) layers, saves the raw layer/query responses, and makes a projected
distance map.  Candidate
coordinates must use either EPSG:32613 (WGS 84 / UTM zone 13N; default) or
EPSG:26913 (NAD83 / UTM zone 13N).  Both use metre coordinates, but they are
different datums: the ArcGIS server reprojects the layer's native EPSG:26913
geometry to the requested output CRS before this script computes distances.

Well locations in the OCD layers are approximate.  Spatial proximity is useful
source-context information, not evidence that a well emitted methane at the
HISUI acquisition time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np


ANALYSIS_VERSION = "2026-07-28-v2"
SERVICE_URL = (
    "https://mercator.env.nm.gov/server/rest/services/"
    "emnrd/ocd_wells/MapServer"
)
OIL_LAYER_URL = f"{SERVICE_URL}/0"
GAS_LAYER_URL = f"{SERVICE_URL}/1"
POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
LAYER_SPECS = (
    (0, "Oil Wells", OIL_LAYER_URL),
    (1, "Gas Wells", GAS_LAYER_URL),
)
# Backward-compatible names used by small unit-level helpers and tests.
LAYER_URL = OIL_LAYER_URL
QUERY_URL = f"{OIL_LAYER_URL}/query"
SERVICE_NATIVE_EPSG = 26913
DATUM_TRANSFORMATION = 108190
SUPPORTED_MAP_EPSGS = (32613, 26913)
HISUI_PIXEL_SIZE_M = 20.0
OUT_FIELDS = (
    "OBJECTID",
    "API",
    "wellname",
    "well_type",
    "status",
    "methane_category",
    "ogrid_name",
    "dir_status",
    "ogrid",
    "latitude",
    "longitude",
    "URL",
    "spud_date",
    "plug_date",
    "eff_date",
    "apr_date",
)
LOCATION_NOTE = (
    "NM OCD well locations are approximate. Proximity identifies possible "
    "source context; it does not confirm methane emissions."
)
TEMPORAL_NOTE = (
    "This is a live/current OCD inventory snapshot, not historical ground "
    "truth for the HISUI acquisition date; current status does not establish "
    "2022 operating status."
)
HISUI_NOTE = (
    "The candidate cross marks the centre of one nominal 20 m HISUI pixel; "
    "the outlined square shows that nominal pixel footprint."
)
WIND_NOTE = (
    "The wind arrow shows direction only. Hourly reanalysis wind is spatially "
    "coarse and is not a local instantaneous wind observation."
)


class WellContextError(RuntimeError):
    """Raised when service data cannot support a valid context map."""


@dataclass(frozen=True)
class HTTPJSONResponse:
    url: str
    raw: bytes
    payload: dict[str, object]


@dataclass(frozen=True)
class WellRecord:
    source_layer_id: int
    source_layer_name: str
    object_id: int | None
    api: str | None
    name: str
    well_type: str
    status_code: str
    methane_category: str
    status_category: str
    type_category: str
    operator: str | None
    directional_status: str | None
    latitude: float | None
    longitude: float | None
    record_url: str | None
    spud_date: str | None
    plug_date: str | None
    effective_date: str | None
    approval_date: str | None
    easting_m: float
    northing_m: float
    distance_m: float


@dataclass(frozen=True)
class WindContext:
    wind_from_deg: float
    downwind_deg: float
    speed_mps: float | None
    source_label: str
    arrow_is_direction_only: bool = True
    limitation: str = WIND_NOTE


HTTPGetter = Callable[[str, Mapping[str, object], float], HTTPJSONResponse]


def euclidean_distance_m(
    first_easting: float,
    first_northing: float,
    second_easting: float,
    second_northing: float,
) -> float:
    """Return planar distance for coordinates expressed in the same metre CRS."""

    return math.hypot(second_easting - first_easting, second_northing - first_northing)


def _finite_float(value: object, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise WellContextError(f"{name} must be numeric") from error
    if not math.isfinite(numeric):
        raise WellContextError(f"{name} must be finite")
    return numeric


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _arcgis_error(payload: Mapping[str, object], url: str) -> None:
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return
    code = error.get("code")
    message = error.get("message")
    details = error.get("details")
    detail_text = ""
    if isinstance(details, list) and details:
        detail_text = "; " + "; ".join(str(value) for value in details)
    raise WellContextError(
        f"ArcGIS service error at {url}: code={code}, message={message}{detail_text}"
    )


def fetch_arcgis_json(
    url: str,
    params: Mapping[str, object],
    timeout_seconds: float,
) -> HTTPJSONResponse:
    """GET one ArcGIS JSON response with a clear, bounded failure mode."""

    encoded = urlencode({key: str(value) for key, value in params.items()})
    request_url = f"{url}?{encoded}"
    request = Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hisui-methane-iterative-mf/2026-07-28",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            final_url = response.geturl()
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise WellContextError(
            f"NM OCD ArcGIS HTTP {error.code} for {request_url}: {body[:500]}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WellContextError(
            f"NM OCD ArcGIS request failed for {request_url}: {error}"
        ) from error
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise WellContextError(
            f"NM OCD ArcGIS returned invalid JSON for {request_url}"
        ) from error
    if not isinstance(payload, dict):
        raise WellContextError(
            f"NM OCD ArcGIS returned a non-object JSON response for {request_url}"
        )
    _arcgis_error(payload, final_url)
    return HTTPJSONResponse(url=final_url, raw=raw, payload=payload)


def fetch_power_json(
    url: str,
    params: Mapping[str, object],
    timeout_seconds: float,
) -> HTTPJSONResponse:
    """GET one NASA POWER JSON response with bounded, explicit failures."""

    encoded = urlencode({key: str(value) for key, value in params.items()})
    request_url = f"{url}?{encoded}"
    request = Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hisui-methane-iterative-mf/2026-07-28",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            final_url = response.geturl()
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise WellContextError(
            f"NASA POWER HTTP {error.code} for {request_url}: {body[:500]}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WellContextError(
            f"NASA POWER request failed for {request_url}: {error}"
        ) from error
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise WellContextError(
            f"NASA POWER returned invalid JSON for {request_url}"
        ) from error
    if not isinstance(payload, dict):
        raise WellContextError(
            f"NASA POWER returned a non-object JSON response for {request_url}"
        )
    return HTTPJSONResponse(url=final_url, raw=raw, payload=payload)


def _response_wkid(payload: Mapping[str, object]) -> int | None:
    reference = payload.get("spatialReference")
    if not isinstance(reference, Mapping):
        return None
    for key in ("latestWkid", "wkid"):
        value = reference.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _classify_status(attributes: Mapping[str, object]) -> str:
    methane_category = str(attributes.get("methane_category") or "").strip().lower()
    status = str(attributes.get("status") or "").strip().lower()
    combined = f"{methane_category} {status}"
    if any(
        token in combined
        for token in ("non-active", "nonactive", "inactive", "plug", "abandon")
    ):
        return "non_active"
    if methane_category.endswith(", active") or status.startswith(("act", "new")):
        return "active"
    return "unknown"


def _classify_type(attributes: Mapping[str, object]) -> str:
    value = str(attributes.get("well_type") or "").strip().lower()
    methane_category = str(attributes.get("methane_category") or "").strip().lower()
    combined = f"{value} {methane_category}"
    if "gas" in combined or value == "g":
        return "gas"
    if "co2" in combined:
        return "co2"
    if "inject" in combined or value in {"i", "inj"}:
        return "injection"
    if "water" in combined or value in {"w", "wat"}:
        return "water"
    if "oil" in combined or value == "o":
        return "oil"
    return value or "unknown"


def parse_query_response(
    payload: Mapping[str, object],
    *,
    candidate_easting_m: float,
    candidate_northing_m: float,
    radius_m: float,
    output_epsg: int,
    source_layer_id: int = 0,
    source_layer_name: str = "Oil Wells",
) -> tuple[list[WellRecord], int]:
    """Validate, classify, distance-filter, and sort ArcGIS point features."""

    _arcgis_error(payload, QUERY_URL)
    if payload.get("exceededTransferLimit") is True:
        raise WellContextError(
            "NM OCD query exceeded the ArcGIS transfer limit; reduce --radius-m "
            "instead of accepting a truncated well set"
        )
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise WellContextError("ArcGIS query response lacks a features array")
    response_wkid = _response_wkid(payload)
    if raw_features and response_wkid != output_epsg:
        raise WellContextError(
            f"ArcGIS query returned EPSG:{response_wkid}, expected EPSG:{output_epsg}"
        )

    wells: list[WellRecord] = []
    for index, feature in enumerate(raw_features):
        if not isinstance(feature, Mapping):
            raise WellContextError(f"ArcGIS feature {index} is not an object")
        geometry = feature.get("geometry")
        attributes = feature.get("attributes")
        if not isinstance(geometry, Mapping) or not isinstance(attributes, Mapping):
            raise WellContextError(
                f"ArcGIS feature {index} lacks point geometry or attributes"
            )
        x = _finite_float(geometry.get("x"), f"feature {index} x")
        y = _finite_float(geometry.get("y"), f"feature {index} y")
        distance = euclidean_distance_m(
            candidate_easting_m, candidate_northing_m, x, y
        )
        if distance > radius_m + 1e-6:
            continue
        object_id_value = attributes.get("OBJECTID")
        try:
            object_id = int(object_id_value) if object_id_value is not None else None
        except (TypeError, ValueError):
            object_id = None
        name = _optional_text(attributes.get("wellname")) or "Unnamed well"
        well_type = _optional_text(attributes.get("well_type")) or "Unknown"
        methane_category = _optional_text(attributes.get("methane_category"))
        status_code = _optional_text(attributes.get("status")) or "Unknown"
        wells.append(
            WellRecord(
                source_layer_id=int(source_layer_id),
                source_layer_name=str(source_layer_name),
                object_id=object_id,
                api=_optional_text(attributes.get("API")),
                name=name,
                well_type=well_type,
                status_code=status_code,
                methane_category=methane_category or "Unknown",
                status_category=_classify_status(attributes),
                type_category=_classify_type(attributes),
                operator=_optional_text(attributes.get("ogrid_name")),
                directional_status=_optional_text(attributes.get("dir_status")),
                latitude=_optional_float(attributes.get("latitude")),
                longitude=_optional_float(attributes.get("longitude")),
                record_url=_optional_text(attributes.get("URL")),
                spud_date=_optional_text(attributes.get("spud_date")),
                plug_date=_optional_text(attributes.get("plug_date")),
                effective_date=_optional_text(attributes.get("eff_date")),
                approval_date=_optional_text(attributes.get("apr_date")),
                easting_m=x,
                northing_m=y,
                distance_m=distance,
            )
        )
    wells.sort(key=lambda well: (well.distance_m, well.object_id or -1))
    return wells, len(raw_features)


def _metadata_native_epsg(metadata: Mapping[str, object]) -> int | None:
    extent = metadata.get("extent")
    if isinstance(extent, Mapping):
        wkid = _response_wkid(extent)
        if wkid is not None:
            return wkid
    return _response_wkid(metadata)


def deduplicate_wells_by_api(
    wells: Sequence[WellRecord],
) -> tuple[list[WellRecord], list[dict[str, object]]]:
    """Deduplicate cross-layer API records with a deterministic preference."""

    ordered = sorted(
        wells,
        key=lambda well: (
            well.distance_m,
            well.source_layer_id,
            well.object_id if well.object_id is not None else -1,
        ),
    )
    kept: list[WellRecord] = []
    by_api: dict[str, WellRecord] = {}
    duplicates: list[dict[str, object]] = []
    for well in ordered:
        normalized_api = (well.api or "").strip().upper()
        if not normalized_api:
            kept.append(well)
            continue
        prior = by_api.get(normalized_api)
        if prior is None:
            by_api[normalized_api] = well
            kept.append(well)
            continue
        duplicates.append(
            {
                "api": normalized_api,
                "kept_source_layer_id": prior.source_layer_id,
                "kept_object_id": prior.object_id,
                "kept_distance_m": prior.distance_m,
                "dropped_source_layer_id": well.source_layer_id,
                "dropped_object_id": well.object_id,
                "dropped_distance_m": well.distance_m,
            }
        )
    return kept, duplicates


def _well_query_params(
    *,
    candidate_easting_m: float,
    candidate_northing_m: float,
    epsg: int,
    radius_m: float,
    max_record_count: int,
) -> dict[str, object]:
    params: dict[str, object] = {
        "where": "1=1",
        "geometry": json.dumps(
            {
                "x": candidate_easting_m,
                "y": candidate_northing_m,
                "spatialReference": {"wkid": epsg},
            },
            separators=(",", ":"),
        ),
        "geometryType": "esriGeometryPoint",
        "inSR": epsg,
        "spatialRel": "esriSpatialRelIntersects",
        "distance": f"{radius_m:.6f}",
        "units": "esriSRUnit_Meter",
        "outFields": ",".join(OUT_FIELDS),
        "returnGeometry": "true",
        "returnZ": "false",
        "returnM": "false",
        "outSR": epsg,
        "orderByFields": "OBJECTID ASC",
        "geometryPrecision": 3,
        "resultRecordCount": max_record_count,
        "f": "json",
    }
    if epsg != SERVICE_NATIVE_EPSG:
        params["datumTransformation"] = DATUM_TRANSFORMATION
    return params


def query_official_wells(
    *,
    candidate_easting_m: float,
    candidate_northing_m: float,
    epsg: int,
    radius_m: float,
    timeout_seconds: float,
    http_get: HTTPGetter = fetch_arcgis_json,
) -> tuple[
    list[HTTPJSONResponse],
    list[HTTPJSONResponse],
    list[WellRecord],
    dict[str, int],
    list[dict[str, object]],
]:
    if epsg not in SUPPORTED_MAP_EPSGS:
        raise WellContextError(
            "--epsg must be 32613 (WGS84 UTM 13N) or 26913 "
            "(NAD83 UTM 13N), so plotted Euclidean distances are in metres"
        )
    candidate_easting_m = _finite_float(candidate_easting_m, "candidate easting")
    candidate_northing_m = _finite_float(candidate_northing_m, "candidate northing")
    radius_m = _finite_float(radius_m, "radius")
    timeout_seconds = _finite_float(timeout_seconds, "timeout")
    if radius_m <= 0:
        raise WellContextError("radius must be positive")
    if timeout_seconds <= 0:
        raise WellContextError("timeout must be positive")

    metadata_responses: list[HTTPJSONResponse] = []
    query_responses: list[HTTPJSONResponse] = []
    all_wells: list[WellRecord] = []
    returned_counts: dict[str, int] = {}
    for layer_id, layer_name, layer_url in LAYER_SPECS:
        metadata_response = http_get(layer_url, {"f": "json"}, timeout_seconds)
        metadata = metadata_response.payload
        _arcgis_error(metadata, metadata_response.url)
        if metadata.get("geometryType") != "esriGeometryPoint":
            raise WellContextError(
                f"NM OCD {layer_name} layer is no longer an ArcGIS point layer"
            )
        native_epsg = _metadata_native_epsg(metadata)
        if native_epsg != SERVICE_NATIVE_EPSG:
            raise WellContextError(
                f"NM OCD {layer_name} native CRS changed from "
                f"EPSG:{SERVICE_NATIVE_EPSG} to EPSG:{native_epsg}; review "
                "datum handling before continuing"
            )
        query_params = _well_query_params(
            candidate_easting_m=candidate_easting_m,
            candidate_northing_m=candidate_northing_m,
            epsg=epsg,
            radius_m=radius_m,
            max_record_count=int(metadata.get("maxRecordCount") or 2000),
        )
        query_response = http_get(
            f"{layer_url}/query", query_params, timeout_seconds
        )
        layer_wells, returned_count = parse_query_response(
            query_response.payload,
            candidate_easting_m=candidate_easting_m,
            candidate_northing_m=candidate_northing_m,
            radius_m=radius_m,
            output_epsg=epsg,
            source_layer_id=layer_id,
            source_layer_name=layer_name,
        )
        metadata_responses.append(metadata_response)
        query_responses.append(query_response)
        all_wells.extend(layer_wells)
        returned_counts[str(layer_id)] = returned_count
    wells, duplicates = deduplicate_wells_by_api(all_wells)
    return metadata_responses, query_responses, wells, returned_counts, duplicates


def make_wind_context(
    wind_from_deg: float | None,
    wind_speed_mps: float | None,
    source_label: str,
) -> WindContext | None:
    if wind_from_deg is None:
        if wind_speed_mps is not None:
            raise WellContextError("--wind-speed-mps requires --wind-from-deg")
        return None
    direction = _finite_float(wind_from_deg, "wind-from direction") % 360.0
    speed = (
        None
        if wind_speed_mps is None
        else _finite_float(wind_speed_mps, "wind speed")
    )
    if speed is not None and speed < 0:
        raise WellContextError("wind speed must be non-negative")
    label = source_label.strip()
    if not label:
        raise WellContextError("wind source label must not be blank")
    return WindContext(
        wind_from_deg=direction,
        downwind_deg=(direction + 180.0) % 360.0,
        speed_mps=speed,
        source_label=label,
    )


def fetch_power_hourly_wind(
    *,
    latitude: float,
    longitude: float,
    date_utc: str,
    hour_utc: int,
    timeout_seconds: float = 30.0,
    http_get: HTTPGetter = fetch_power_json,
) -> tuple[WindContext, HTTPJSONResponse, dict[str, object]]:
    """Fetch one reproducible UTC hourly WS10M/WD10M value from POWER."""

    latitude = _finite_float(latitude, "POWER latitude")
    longitude = _finite_float(longitude, "POWER longitude")
    if not -90.0 <= latitude <= 90.0:
        raise WellContextError("POWER latitude must be within [-90, 90]")
    if not -180.0 <= longitude <= 180.0:
        raise WellContextError("POWER longitude must be within [-180, 180]")
    try:
        parsed_date = datetime.strptime(date_utc, "%Y-%m-%d")
    except ValueError as error:
        raise WellContextError("POWER date must use YYYY-MM-DD") from error
    if not 0 <= int(hour_utc) <= 23:
        raise WellContextError("POWER UTC hour must be within [0, 23]")
    compact_date = parsed_date.strftime("%Y%m%d")
    timestamp_key = f"{compact_date}{int(hour_utc):02d}"
    params: dict[str, object] = {
        "parameters": "WS10M,WD10M",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": compact_date,
        "end": compact_date,
        "format": "JSON",
        "time-standard": "UTC",
    }
    response = http_get(POWER_HOURLY_URL, params, timeout_seconds)
    header = response.payload.get("header")
    if not isinstance(header, Mapping) or str(header.get("time_standard")) != "UTC":
        raise WellContextError("NASA POWER response does not declare UTC time")
    properties = response.payload.get("properties")
    parameters = properties.get("parameter") if isinstance(properties, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise WellContextError("NASA POWER response is missing properties.parameter")
    try:
        speed = _finite_float(parameters["WS10M"][timestamp_key], "POWER WS10M")
        direction = _finite_float(parameters["WD10M"][timestamp_key], "POWER WD10M")
    except (KeyError, TypeError) as error:
        raise WellContextError(
            f"NASA POWER response is missing WS10M/WD10M at {timestamp_key} UTC"
        ) from error
    fill_value = _optional_float(header.get("fill_value"))
    if fill_value is not None and (speed == fill_value or direction == fill_value):
        raise WellContextError(f"NASA POWER returned fill data at {timestamp_key} UTC")
    wind = make_wind_context(
        direction,
        speed,
        f"NASA POWER MERRA-2 hourly, {date_utc} {int(hour_utc):02d}:00 UTC",
    )
    assert wind is not None
    return wind, response, params


STATUS_COLORS = {
    "active": "#1b9e77",
    "non_active": "#7570b3",
    "unknown": "#d95f02",
}
TYPE_MARKERS = {
    "oil": "o",
    "gas": "^",
    "co2": "P",
    "injection": "s",
    "water": "D",
    "unknown": "h",
}


def _display_category(value: str) -> str:
    return value.replace("_", " ").title()


def _well_label(well: WellRecord, rank: int) -> str:
    api = well.api or "API unavailable"
    return (
        f"{rank}. {well.name}\n"
        f"   {api} | {well.distance_m:.1f} m | "
        f"{_display_category(well.status_category)} {well.well_type} "
        f"({well.status_code})"
    )


def save_well_context_figure(
    path: Path,
    wells: Sequence[WellRecord],
    *,
    candidate_easting_m: float,
    candidate_northing_m: float,
    epsg: int,
    radius_m: float,
    label_count: int,
    wind: WindContext | None,
) -> None:
    if label_count < 0:
        raise WellContextError("label-count must not be negative")
    oil_count = sum(well.type_category == "oil" for well in wells)
    gas_count = sum(well.type_category == "gas" for well in wells)
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(
        2, 2, width_ratios=(3.25, 1.25), height_ratios=(18.0, 1.15)
    )
    axis = fig.add_subplot(grid[0, 0])
    sidebar = fig.add_subplot(grid[0, 1])
    footer_axis = fig.add_subplot(grid[1, :])
    sidebar.set_axis_off()
    footer_axis.set_axis_off()

    for fraction in (1.0 / 3.0, 2.0 / 3.0, 1.0):
        ring_radius = radius_m * fraction
        axis.add_patch(
            Circle(
                (candidate_easting_m, candidate_northing_m),
                ring_radius,
                fill=False,
                edgecolor="#8c8c8c" if fraction < 1 else "#333333",
                linewidth=0.8 if fraction < 1 else 1.4,
                linestyle=":" if fraction < 1 else "--",
                zorder=1,
            )
        )
        axis.text(
            candidate_easting_m,
            candidate_northing_m + ring_radius,
            f" {ring_radius:.0f} m",
            ha="left",
            va="bottom",
            color="#555555",
            fontsize=8,
        )

    combined_categories = sorted(
        {(well.type_category, well.status_category) for well in wells}
    )
    legend_handles: list[Line2D] = []
    for type_category, status_category in combined_categories:
        selected = [
            well
            for well in wells
            if well.type_category == type_category
            and well.status_category == status_category
        ]
        marker = TYPE_MARKERS.get(type_category, TYPE_MARKERS["unknown"])
        color = STATUS_COLORS.get(status_category, STATUS_COLORS["unknown"])
        axis.scatter(
            [well.easting_m for well in selected],
            [well.northing_m for well in selected],
            marker=marker,
            s=72,
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
                label=(
                    f"Current {_display_category(status_category)} "
                    f"{_display_category(type_category)}"
                ),
            )
        )

    pixel_half = HISUI_PIXEL_SIZE_M / 2.0
    axis.add_patch(
        Rectangle(
            (candidate_easting_m - pixel_half, candidate_northing_m - pixel_half),
            HISUI_PIXEL_SIZE_M,
            HISUI_PIXEL_SIZE_M,
            facecolor="#e7298a",
            edgecolor="#b30059",
            linewidth=1.2,
            alpha=0.18,
            zorder=6,
        )
    )
    axis.scatter(
        [candidate_easting_m],
        [candidate_northing_m],
        marker="+",
        s=260,
        color="#e7298a",
        linewidth=3.0,
        zorder=8,
        label="HISUI candidate (20 m pixel centre)",
    )
    axis.scatter(
        [candidate_easting_m],
        [candidate_northing_m],
        marker="x",
        s=180,
        color="black",
        linewidth=1.2,
        zorder=7,
    )

    nearest = list(wells[:label_count])
    rank_offsets = (
        (0, 0),
        (0, 13),
        (13, 7),
        (13, -7),
        (0, -13),
        (-13, -7),
        (-13, 7),
    )
    for rank, well in enumerate(nearest, start=1):
        offset = rank_offsets[(rank - 1) % len(rank_offsets)]
        axis.annotate(
            str(rank),
            xy=(well.easting_m, well.northing_m),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            bbox={
                "boxstyle": "circle,pad=0.12",
                "facecolor": "black",
                "edgecolor": "white",
                "linewidth": 0.4,
                "alpha": 0.8,
            },
            arrowprops=(
                None
                if offset == (0, 0)
                else {"arrowstyle": "-", "color": "#555555", "linewidth": 0.6}
            ),
            zorder=9,
        )
    if wells:
        nearest_well = wells[0]
        axis.plot(
            [candidate_easting_m, nearest_well.easting_m],
            [candidate_northing_m, nearest_well.northing_m],
            color="#b30059",
            linewidth=1.2,
            linestyle="--",
            zorder=3,
        )
        midpoint_x = (candidate_easting_m + nearest_well.easting_m) / 2.0
        midpoint_y = (candidate_northing_m + nearest_well.northing_m) / 2.0
        distance_label_angle = math.degrees(
            math.atan2(
                nearest_well.northing_m - candidate_northing_m,
                nearest_well.easting_m - candidate_easting_m,
            )
        )
        if distance_label_angle > 90.0:
            distance_label_angle -= 180.0
        elif distance_label_angle < -90.0:
            distance_label_angle += 180.0
        axis.text(
            midpoint_x,
            midpoint_y,
            f" {nearest_well.distance_m:.1f} m ",
            fontsize=9,
            color="#b30059",
            ha="center",
            va="bottom",
            rotation=distance_label_angle,
            rotation_mode="anchor",
            zorder=10,
        )

    if wind is not None:
        bearing_rad = math.radians(wind.downwind_deg)
        arrow_length = radius_m * 0.42
        delta_x = math.sin(bearing_rad) * arrow_length
        delta_y = math.cos(bearing_rad) * arrow_length
        axis.annotate(
            "",
            xy=(candidate_easting_m + delta_x, candidate_northing_m + delta_y),
            xytext=(candidate_easting_m, candidate_northing_m),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#377eb8",
                "linewidth": 2.4,
                "mutation_scale": 16,
            },
            zorder=7,
        )
        speed_label = (
            "" if wind.speed_mps is None else f", {wind.speed_mps:.2f} m/s"
        )
        axis.text(
            candidate_easting_m + delta_x,
            candidate_northing_m + delta_y,
            f" downwind {wind.downwind_deg:.1f}°{speed_label}",
            color="#245b8a",
            fontsize=9,
            ha="left" if delta_x >= 0 else "right",
            va="bottom" if delta_y >= 0 else "top",
            zorder=8,
        )

    padding = radius_m * 1.08
    axis.set_xlim(candidate_easting_m - padding, candidate_easting_m + padding)
    axis.set_ylim(candidate_northing_m - padding, candidate_northing_m + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d0d0d0", linewidth=0.5, zorder=0)
    axis.set_xlabel(f"Easting (m; EPSG:{epsg})")
    axis.set_ylabel(f"Northing (m; EPSG:{epsg})")
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(formatter)
    axis.text(
        0.03,
        0.97,
        "N",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    axis.annotate(
        "",
        xy=(0.03, 0.94),
        xytext=(0.03, 0.87),
        xycoords=axis.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.5},
    )
    scale_length = radius_m / 3.0
    scale_x0 = candidate_easting_m - radius_m * 0.92
    scale_y0 = candidate_northing_m - radius_m * 0.92
    axis.plot(
        [scale_x0, scale_x0 + scale_length],
        [scale_y0, scale_y0],
        color="black",
        linewidth=3,
        solid_capstyle="butt",
        zorder=10,
    )
    axis.text(
        scale_x0 + scale_length / 2.0,
        scale_y0 + radius_m * 0.025,
        f"{scale_length:.0f} m",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    candidate_handle = Line2D(
        [0],
        [0],
        marker="+",
        color="#e7298a",
        linestyle="none",
        markersize=12,
        markeredgewidth=2.5,
        label="HISUI candidate",
    )
    radius_handle = Line2D(
        [0],
        [0],
        color="#333333",
        linestyle="--",
        label=f"Query radius ({radius_m:.0f} m)",
    )
    if wind is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#377eb8",
                linewidth=2.4,
                label="Schematic downwind direction",
            )
        )
    axis.legend(
        handles=[candidate_handle, radius_handle, *legend_handles],
        loc="lower right",
        framealpha=0.92,
        fontsize=8,
    )

    sidebar.text(
        0.0,
        1.0,
        f"Nearest wells ({min(label_count, len(wells))} labeled / {len(wells)} total)\n"
        f"{oil_count} oil + {gas_count} gas",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=sidebar.transAxes,
    )
    label_y = 0.94
    for rank, well in enumerate(nearest, start=1):
        sidebar.text(
            0.0,
            label_y,
            _well_label(well, rank),
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
            transform=sidebar.transAxes,
            wrap=True,
        )
        label_y -= 0.13
    if wind is not None:
        sidebar.text(
            0.0,
            max(label_y - 0.02, 0.17),
            "Wind context\n"
            f"from {wind.wind_from_deg:.1f}° → downwind {wind.downwind_deg:.1f}°\n"
            f"{wind.source_label}",
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.3,
            transform=sidebar.transAxes,
            wrap=True,
        )
    sidebar.text(
        0.0,
        0.02,
        "Official live source\nNM EMNRD Oil Conservation Division\n"
        "Oil Wells (0) + Gas Wells (1)\n"
        f"Native EPSG:{SERVICE_NATIVE_EPSG}; plotted EPSG:{epsg}\n"
        "Current inventory; not 2022 status",
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.3,
        transform=sidebar.transAxes,
    )

    fig.suptitle(
        "Current official NM OCD oil + gas well context for HISUI methane candidate\n"
        f"candidate E={candidate_easting_m:.1f} m, N={candidate_northing_m:.1f} m "
        f"| radius={radius_m:.0f} m | {len(wells)} wells",
        fontsize=14,
    )
    footer = (
        "HISUI cross = centre of a nominal 20 m pixel; square = nominal footprint. "
        "OCD well positions are approximate; proximity does not confirm emissions.\n"
        "Current OCD inventory, not 2022 operating status."
    )
    if wind is not None:
        footer += (
            " Wind arrow shows direction only; hourly reanalysis is not local "
            "instantaneous wind."
        )
    footer_axis.text(
        0.5,
        0.5,
        footer,
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
        transform=footer_axis.transAxes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_candidate_well_context(
    *,
    candidate_easting_m: float,
    candidate_northing_m: float,
    epsg: int,
    radius_m: float,
    output_dir: str | Path,
    timeout_seconds: float = 30.0,
    label_count: int = 5,
    wind_from_deg: float | None = None,
    wind_speed_mps: float | None = None,
    wind_source_label: str = "User-provided wind",
    power_latitude: float | None = None,
    power_longitude: float | None = None,
    power_date_utc: str | None = None,
    power_hour_utc: int | None = None,
    http_get: HTTPGetter = fetch_arcgis_json,
    power_http_get: HTTPGetter = fetch_power_json,
) -> dict[str, object]:
    candidate_easting_m = _finite_float(candidate_easting_m, "candidate easting")
    candidate_northing_m = _finite_float(candidate_northing_m, "candidate northing")
    radius_m = _finite_float(radius_m, "radius")
    if int(label_count) < 0:
        raise WellContextError("label-count must not be negative")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_names = [
        "candidate_well_context.png",
        "candidate_well_context.json",
        "nasa_power_hourly_raw.json",
    ]
    for _, layer_name, _ in LAYER_SPECS:
        layer_slug = layer_name.lower().replace(" ", "_")
        artifact_names.extend(
            (
                f"ocd_{layer_slug}_layer_raw.json",
                f"ocd_{layer_slug}_query_raw.json",
            )
        )
    # Invalidate results from an earlier successful query before doing any
    # network work.  A failed refresh must never leave a stale map/summary that
    # can be mistaken for the current run.
    for artifact_name in artifact_names:
        (destination / artifact_name).unlink(missing_ok=True)
    power_values = (
        power_latitude,
        power_longitude,
        power_date_utc,
        power_hour_utc,
    )
    power_requested = any(value is not None for value in power_values)
    if power_requested and not all(value is not None for value in power_values):
        raise WellContextError(
            "POWER provenance requires latitude, longitude, date, and UTC hour"
        )
    if power_requested and (wind_from_deg is not None or wind_speed_mps is not None):
        raise WellContextError(
            "Use either POWER query inputs or manual wind values, not both"
        )
    power_response: HTTPJSONResponse | None = None
    power_query: dict[str, object] | None = None
    if power_requested:
        wind, power_response, power_query = fetch_power_hourly_wind(
            latitude=float(power_latitude),
            longitude=float(power_longitude),
            date_utc=str(power_date_utc),
            hour_utc=int(power_hour_utc),
            timeout_seconds=timeout_seconds,
            http_get=power_http_get,
        )
    else:
        wind = make_wind_context(
            wind_from_deg, wind_speed_mps, wind_source_label
        )
    (
        metadata_responses,
        query_responses,
        wells,
        returned_counts,
        duplicate_api_records,
    ) = query_official_wells(
        candidate_easting_m=candidate_easting_m,
        candidate_northing_m=candidate_northing_m,
        epsg=int(epsg),
        radius_m=radius_m,
        timeout_seconds=timeout_seconds,
        http_get=http_get,
    )

    figure_path = destination / "candidate_well_context.png"
    summary_path = destination / "candidate_well_context.json"
    power_raw_path = destination / "nasa_power_hourly_raw.json"
    raw_paths: dict[int, tuple[Path, Path]] = {}
    for (layer_id, layer_name, _), metadata_response, query_response in zip(
        LAYER_SPECS, metadata_responses, query_responses
    ):
        layer_slug = layer_name.lower().replace(" ", "_")
        metadata_raw_path = destination / f"ocd_{layer_slug}_layer_raw.json"
        query_raw_path = destination / f"ocd_{layer_slug}_query_raw.json"
        metadata_raw_path.write_bytes(metadata_response.raw)
        query_raw_path.write_bytes(query_response.raw)
        raw_paths[layer_id] = (metadata_raw_path, query_raw_path)
    if power_response is not None:
        power_raw_path.write_bytes(power_response.raw)

    save_well_context_figure(
        figure_path,
        wells,
        candidate_easting_m=candidate_easting_m,
        candidate_northing_m=candidate_northing_m,
        epsg=int(epsg),
        radius_m=radius_m,
        label_count=int(label_count),
        wind=wind,
    )

    status_counts = Counter(well.status_category for well in wells)
    type_counts = Counter(well.type_category for well in wells)
    nearest = wells[0] if wells else None
    if int(epsg) == SERVICE_NATIVE_EPSG:
        datum_note = (
            "Candidate and returned well geometries use the layer's native "
            "EPSG:26913 (NAD83 / UTM zone 13N); no datum transformation was needed."
        )
    else:
        datum_note = (
            "EPSG:32613 and EPSG:26913 use different datums and must not be "
            "treated as identical numeric grids. ArcGIS reprojected official "
            "well geometries from native EPSG:26913 to EPSG:32613 using explicit "
            f"datumTransformation WKID {DATUM_TRANSFORMATION}; planar distances "
            "were then computed in metres."
        )
    summary: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "easting_m": candidate_easting_m,
            "northing_m": candidate_northing_m,
            "epsg": int(epsg),
            "nominal_hisui_pixel_size_m": HISUI_PIXEL_SIZE_M,
            "note": HISUI_NOTE,
        },
        "query": {
            "radius_m": radius_m,
            "service_url": SERVICE_URL,
            "layers": [
                {"layer_id": layer_id, "name": layer_name, "url": layer_url}
                for layer_id, layer_name, layer_url in LAYER_SPECS
            ],
            "layer_native_epsg": SERVICE_NATIVE_EPSG,
            "datum_transformation_wkid": (
                DATUM_TRANSFORMATION if int(epsg) != SERVICE_NATIVE_EPSG else None
            ),
            "output_epsg": int(epsg),
            "arcgis_returned_feature_count_by_layer": returned_counts,
            "arcgis_returned_feature_count_total": int(sum(returned_counts.values())),
            "within_planar_radius_before_api_deduplication": (
                len(wells) + len(duplicate_api_records)
            ),
            "within_planar_radius_after_api_deduplication": len(wells),
            "duplicate_api_records_removed": duplicate_api_records,
            "status_counts": dict(sorted(status_counts.items())),
            "type_counts": dict(sorted(type_counts.items())),
        },
        "coordinate_handling": {
            "candidate_crs": f"EPSG:{int(epsg)}",
            "service_native_crs": "EPSG:26913 (NAD83 / UTM zone 13N)",
            "default_candidate_crs": "EPSG:32613 (WGS 84 / UTM zone 13N)",
            "datum_note": datum_note,
        },
        "nearest_well": asdict(nearest) if nearest is not None else None,
        "wells": [asdict(well) for well in wells],
        "wind": asdict(wind) if wind is not None else None,
        "wind_provenance": (
            {
                "mode": "nasa_power_hourly_api",
                "endpoint": POWER_HOURLY_URL,
                "request_url": power_response.url,
                "query_parameters": power_query,
                "raw_response": {
                    "path": str(power_raw_path),
                    "sha256": _sha256(power_response.raw),
                },
            }
            if power_response is not None
            else {
                "mode": "manual_or_absent",
                "note": (
                    "Manual wind values have no independently saved API response."
                    if wind is not None
                    else "No wind input was used."
                ),
            }
        ),
        "limitations": {
            "well_location": LOCATION_NOTE,
            "methane_attribution": (
                "A nearby permitted well is a plausible source candidate only. "
                "This spatial join is not an emission detection or attribution."
            ),
            "wind": WIND_NOTE if wind is not None else None,
            "temporal_inventory": TEMPORAL_NOTE,
        },
        "raw_response_provenance": {
            str(layer_id): {
                "layer_id": layer_id,
                "layer_name": layer_name,
                "layer_metadata": {
                    "path": str(raw_paths[layer_id][0]),
                    "request_url": metadata_response.url,
                    "sha256": _sha256(metadata_response.raw),
                },
                "well_query": {
                    "path": str(raw_paths[layer_id][1]),
                    "request_url": query_response.url,
                    "sha256": _sha256(query_response.raw),
                },
            }
            for (layer_id, layer_name, _), metadata_response, query_response in zip(
                LAYER_SPECS, metadata_responses, query_responses
            )
        },
        "outputs": {
            "figure": str(figure_path),
            "summary": str(summary_path),
            "raw_oil_layer_metadata": str(raw_paths[0][0]),
            "raw_oil_query_response": str(raw_paths[0][1]),
            "raw_gas_layer_metadata": str(raw_paths[1][0]),
            "raw_gas_query_response": str(raw_paths[1][1]),
            **(
                {"raw_nasa_power_hourly_response": str(power_raw_path)}
                if power_response is not None
                else {}
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-easting", type=float, required=True)
    parser.add_argument("--candidate-northing", type=float, required=True)
    parser.add_argument(
        "--epsg",
        type=int,
        choices=SUPPORTED_MAP_EPSGS,
        default=32613,
        help=(
            "Candidate/output projected CRS. The official layer is native "
            "EPSG:26913; default EPSG:32613 uses a different datum."
        ),
    )
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-count", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--wind-from-deg",
        type=float,
        help="Meteorological wind-from direction, clockwise from north",
    )
    parser.add_argument("--wind-speed-mps", type=float)
    parser.add_argument(
        "--wind-source-label",
        default="User-provided wind",
        help="Visible provenance label for optional wind input",
    )
    parser.add_argument("--power-latitude", type=float)
    parser.add_argument("--power-longitude", type=float)
    parser.add_argument(
        "--power-date-utc",
        help="UTC date for a reproducible NASA POWER hourly query (YYYY-MM-DD)",
    )
    parser.add_argument("--power-hour-utc", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = make_candidate_well_context(
            candidate_easting_m=args.candidate_easting,
            candidate_northing_m=args.candidate_northing,
            epsg=args.epsg,
            radius_m=args.radius_m,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout_seconds,
            label_count=args.label_count,
            wind_from_deg=args.wind_from_deg,
            wind_speed_mps=args.wind_speed_mps,
            wind_source_label=args.wind_source_label,
            power_latitude=args.power_latitude,
            power_longitude=args.power_longitude,
            power_date_utc=args.power_date_utc,
            power_hour_utc=args.power_hour_utc,
        )
    except WellContextError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
