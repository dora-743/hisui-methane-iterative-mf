from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_candidate_well_context import (  # noqa: E402
    DATUM_TRANSFORMATION,
    GAS_LAYER_URL,
    HTTPJSONResponse,
    LAYER_URL,
    POWER_HOURLY_URL,
    QUERY_URL,
    WellContextError,
    WellRecord,
    euclidean_distance_m,
    make_candidate_well_context,
    make_wind_context,
    parse_query_response,
    save_well_context_figure,
)


def _feature(
    object_id: int,
    x: float,
    y: float,
    *,
    category: str = "Oil, Active",
    status: str = "Acti",
    name: str = "TEST WELL #001",
    well_type: str = "Oil",
) -> dict[str, object]:
    return {
        "attributes": {
            "OBJECTID": object_id,
            "API": f"30-025-{object_id:05d}",
            "wellname": name,
            "well_type": well_type,
            "status": status,
            "methane_category": category,
            "ogrid_name": "Test Operator",
            "dir_status": "H",
            "latitude": 32.0,
            "longitude": -103.0,
            "URL": "https://example.invalid/well",
            "spud_date": None,
            "plug_date": "99991231065958",
            "eff_date": None,
            "apr_date": None,
        },
        "geometry": {"x": x, "y": y},
    }


def _metadata_payload() -> dict[str, object]:
    return {
        "name": "Oil Wells",
        "geometryType": "esriGeometryPoint",
        "maxRecordCount": 2000,
        "extent": {"spatialReference": {"wkid": 26913}},
        "sourceSpatialReference": {"wkid": 4269},
    }


class PlotCandidateWellContextTests(unittest.TestCase):
    def test_euclidean_distance_uses_common_projected_metres(self) -> None:
        self.assertEqual(euclidean_distance_m(100.0, 200.0, 103.0, 204.0), 5.0)

    def test_query_parsing_classifies_sorts_and_radius_filters(self) -> None:
        payload = {
            "spatialReference": {"wkid": 32613},
            "features": [
                _feature(
                    2,
                    120.0,
                    100.0,
                    category="Oil, Non-active",
                    status="Plug",
                ),
                _feature(1, 103.0, 104.0),
                _feature(3, 106.0, 108.0, status="New"),
            ],
        }
        wells, returned = parse_query_response(
            payload,
            candidate_easting_m=100.0,
            candidate_northing_m=100.0,
            radius_m=10.0,
            output_epsg=32613,
        )
        self.assertEqual(returned, 3)
        self.assertEqual([well.object_id for well in wells], [1, 3])
        self.assertEqual(wells[0].distance_m, 5.0)
        self.assertEqual(wells[0].status_category, "active")
        self.assertEqual(wells[0].type_category, "oil")
        self.assertEqual(wells[0].status_code, "Acti")
        self.assertEqual(wells[0].plug_date, "99991231065958")

    def test_query_parser_rejects_service_error_and_truncation(self) -> None:
        with self.assertRaisesRegex(WellContextError, "code=400"):
            parse_query_response(
                {"error": {"code": 400, "message": "Bad geometry"}},
                candidate_easting_m=0.0,
                candidate_northing_m=0.0,
                radius_m=100.0,
                output_epsg=32613,
            )
        with self.assertRaisesRegex(WellContextError, "transfer limit"):
            parse_query_response(
                {
                    "spatialReference": {"wkid": 32613},
                    "features": [],
                    "exceededTransferLimit": True,
                },
                candidate_easting_m=0.0,
                candidate_northing_m=0.0,
                radius_m=100.0,
                output_epsg=32613,
            )

    def test_mocked_network_workflow_writes_raw_provenance_and_figure(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        metadata = _metadata_payload()
        oil_query = {
            "spatialReference": {"wkid": 32613},
            "features": [_feature(44606, 653732.4, 3566443.8)],
        }
        gas_query = {
            "spatialReference": {"wkid": 32613},
            "features": [
                _feature(
                    44606,
                    653732.4,
                    3566443.8,
                    category="Gas, Active",
                    well_type="Gas",
                ),
                _feature(
                    20756,
                    654140.0,
                    3566440.0,
                    category="Gas, Non-active",
                    status="Plug",
                    well_type="Gas",
                ),
            ],
        }

        def fake_get(
            url: str, params: dict[str, object], timeout: float
        ) -> HTTPJSONResponse:
            self.assertEqual(timeout, 12.0)
            calls.append((url, dict(params)))
            if not url.endswith("/query"):
                payload = metadata
            elif url == f"{LAYER_URL}/query":
                payload = oil_query
            elif url == f"{GAS_LAYER_URL}/query":
                payload = gas_query
            else:
                raise AssertionError(f"Unexpected URL: {url}")
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            return HTTPJSONResponse(
                url=f"{url}?mock=true", raw=raw, payload=payload
            )

        power_payload = {
            "header": {"time_standard": "UTC", "fill_value": -999.0},
            "properties": {
                "parameter": {
                    "WS10M": {"2022103016": 2.79},
                    "WD10M": {"2022103016": 239.2},
                }
            },
        }
        power_raw = json.dumps(power_payload, separators=(",", ":")).encode("utf-8")

        def fake_power_get(
            url: str, params: dict[str, object], timeout: float
        ) -> HTTPJSONResponse:
            self.assertEqual(url, POWER_HOURLY_URL)
            self.assertEqual(timeout, 12.0)
            self.assertEqual(params["time-standard"], "UTC")
            self.assertEqual(params["start"], "20221030")
            return HTTPJSONResponse(
                url=f"{url}?mock=true", raw=power_raw, payload=power_payload
            )

        with tempfile.TemporaryDirectory() as temporary:
            result = make_candidate_well_context(
                candidate_easting_m=653840.0,
                candidate_northing_m=3566440.0,
                epsg=32613,
                radius_m=750.0,
                output_dir=temporary,
                timeout_seconds=12.0,
                label_count=1,
                power_latitude=32.2241,
                power_longitude=-103.3674,
                power_date_utc="2022-10-30",
                power_hour_utc=16,
                http_get=fake_get,
                power_http_get=fake_power_get,
            )
            self.assertEqual(
                [call[0] for call in calls],
                [LAYER_URL, QUERY_URL, GAS_LAYER_URL, f"{GAS_LAYER_URL}/query"],
            )
            query_params = calls[1][1]
            self.assertEqual(query_params["inSR"], 32613)
            self.assertEqual(query_params["outSR"], 32613)
            self.assertEqual(
                query_params["datumTransformation"], DATUM_TRANSFORMATION
            )
            geometry = json.loads(str(query_params["geometry"]))
            self.assertEqual(geometry["spatialReference"]["wkid"], 32613)
            self.assertAlmostEqual(result["wind"]["downwind_deg"], 59.2)
            self.assertIn("current OCD inventory", result["limitations"]["temporal_inventory"])
            self.assertIn("different datums", result["coordinate_handling"]["datum_note"])
            self.assertEqual(result["query"]["type_counts"], {"gas": 1, "oil": 1})
            self.assertEqual(len(result["query"]["duplicate_api_records_removed"]), 1)
            for output in result["outputs"].values():
                self.assertTrue(Path(output).is_file())
            figure = Path(result["outputs"]["figure"])
            self.assertGreater(figure.stat().st_size, 1000)
            stored_query = Path(
                result["outputs"]["raw_oil_query_response"]
            ).read_bytes()
            self.assertEqual(
                stored_query,
                json.dumps(oil_query, separators=(",", ":")).encode("utf-8"),
            )
            self.assertEqual(
                Path(result["outputs"]["raw_nasa_power_hourly_response"]).read_bytes(),
                power_raw,
            )
            self.assertEqual(
                result["wind_provenance"]["query_parameters"]["time-standard"],
                "UTC",
            )

    def test_failed_refresh_removes_outputs_from_an_earlier_run(self) -> None:
        def failed_get(
            _url: str, _params: dict[str, object], _timeout: float
        ) -> HTTPJSONResponse:
            raise WellContextError("mock network failure")

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            stale_names = (
                "candidate_well_context.png",
                "candidate_well_context.json",
                "ocd_oil_wells_layer_raw.json",
                "ocd_oil_wells_query_raw.json",
                "ocd_gas_wells_layer_raw.json",
                "ocd_gas_wells_query_raw.json",
                "nasa_power_hourly_raw.json",
            )
            for name in stale_names:
                (destination / name).write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(WellContextError, "mock network failure"):
                make_candidate_well_context(
                    candidate_easting_m=653840.0,
                    candidate_northing_m=3566440.0,
                    epsg=32613,
                    radius_m=750.0,
                    output_dir=destination,
                    http_get=failed_get,
                )

            self.assertFalse(any((destination / name).exists() for name in stale_names))

    def test_figure_smoke_handles_active_and_non_active_types(self) -> None:
        common = {
            "source_layer_id": 0,
            "source_layer_name": "Oil Wells",
            "operator": None,
            "directional_status": None,
            "latitude": None,
            "longitude": None,
            "record_url": None,
            "spud_date": None,
            "plug_date": None,
            "effective_date": None,
            "approval_date": None,
        }
        wells = [
            WellRecord(
                object_id=1,
                api="30-025-00001",
                name="ACTIVE OIL",
                well_type="Oil",
                status_code="Acti",
                methane_category="Oil, Active",
                status_category="active",
                type_category="oil",
                easting_m=100.0,
                northing_m=40.0,
                distance_m=107.7,
                **common,
            ),
            WellRecord(
                object_id=2,
                api="30-025-00002",
                name="PLUGGED OIL",
                well_type="Gas",
                status_code="Plug",
                methane_category="Gas, Non-active",
                status_category="non_active",
                type_category="gas",
                source_layer_id=1,
                source_layer_name="Gas Wells",
                easting_m=-180.0,
                northing_m=-100.0,
                distance_m=205.9,
                **{key: value for key, value in common.items() if not key.startswith("source_layer")},
            ),
        ]
        wind = make_wind_context(239.2, 2.79, "Synthetic hourly reanalysis")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "well-context.png"
            save_well_context_figure(
                output,
                wells,
                candidate_easting_m=0.0,
                candidate_northing_m=0.0,
                epsg=32613,
                radius_m=750.0,
                label_count=2,
                wind=wind,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
