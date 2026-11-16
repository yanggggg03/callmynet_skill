import base64
import copy
import io
import math
import sys
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from core import MODEL_IDS, analyze_waveforms, export_bundle, file_rows, validate_records
from demo import make_demo
from excel_writer import write_workbook


def traces(powers=None):
    rows = []
    for model in MODEL_IDS:
        for n in range(101):
            t = n / 100
            value = (powers or {}).get(model, lambda t: 1000)(t)
            rows.append({"run_id": "run-test", "sample_id": "1", "scenario_id": "same-case",
                         "model_id": model, "time_s": t, "power_w": value,
                         "target_w": 1000, "source": "measured"})
    return rows


class StabilityTests(unittest.TestCase):
    def test_pulse_off_segments_excluded_only_by_explicit_window(self):
        demo = make_demo()
        analysis = demo["waveform_analysis"]
        self.assertTrue(all(m["stable"] for m in analysis["metrics"]))
        self.assertTrue(any(r["time_s"] > 6 and r["power_w"] == 0 for r in analysis["waveforms"]))
        self.assertEqual(analysis["metrics"][0]["evaluation_end_s"], 5.95)
        result = analyze_waveforms({"rows": analysis["waveforms"], "source": "demo"})
        self.assertIsNone(result["winner"])

    def test_optional_power_channels_are_physical_and_preserved(self):
        demo = make_demo()["waveform_analysis"]
        self.assertEqual(len(demo["waveforms"]), 2103)
        for row in demo["waveforms"]:
            self.assertAlmostEqual(row["power_w"], row["incident_power_w"] - row["reflected_power_w"])
        bad = copy.deepcopy(demo["waveforms"])
        bad[150]["reflected_power_w"] = bad[150]["incident_power_w"] + 1
        with self.assertRaisesRegex(ValueError, "反射功率"):
            analyze_waveforms({"rows": bad, "source": "demo"})

    def test_invalid_evaluation_window_rejected(self):
        with self.assertRaisesRegex(ValueError, "评估窗口"):
            analyze_waveforms({"rows": traces(), "options": {"evaluation_start_s": 2, "evaluation_end_s": 3}})

    def test_zero_power_is_not_stable_despite_zero_variance(self):
        result = analyze_waveforms({"rows": traces({"residual_mlp": lambda t: 0}), "source": "measured"})
        bad = next(m for m in result["metrics"] if m["model_id"] == "residual_mlp")
        self.assertFalse(bad["stable"])
        self.assertEqual(bad["error_pct"], 100)
        self.assertNotEqual(result["winner"], "residual_mlp")

    def test_none_stable_does_not_claim_winner(self):
        result = analyze_waveforms({"rows": traces(dict.fromkeys(MODEL_IDS, lambda t: 600)), "source": "measured"})
        self.assertIsNone(result["winner"])
        self.assertEqual(result["selection_status"], "none_stable")

    def test_oscillation_rejected_and_good_waveform_wins(self):
        result = analyze_waveforms({"rows": traces({"residual_mlp": lambda t: 1000 + 160 * math.sin(120 * t),
                                                              "fourier_resmlp": lambda t: 1000 + 4 * math.sin(120 * t)}), "source": "measured"})
        self.assertEqual(result["winner"], "conditional_flow")
        self.assertFalse(next(m for m in result["metrics"] if m["model_id"] == "residual_mlp")["stable"])

    def test_grid_mismatch_rejected(self):
        rows = traces(); rows[-1]["time_s"] += .01
        with self.assertRaisesRegex(ValueError, "时间网格"):
            analyze_waveforms({"rows": rows})

    def test_stale_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "run_id"):
            analyze_waveforms({"rows": traces(), "expected_run_id": "different-run"})

    def test_mixed_conditions_rejected(self):
        rows = traces(); rows[0]["scenario_id"] = "another-case"
        with self.assertRaisesRegex(ValueError, "同一"):
            analyze_waveforms({"rows": rows})

    def test_mixed_sources_rejected(self):
        rows = traces(); rows[0]["source"] = "simulation"
        with self.assertRaisesRegex(ValueError, "source"):
            analyze_waveforms({"rows": rows})

    def test_sparse_tail_rejected(self):
        rows = [r for r in traces() if r["time_s"] <= .5 or r["time_s"] == 1]
        with self.assertRaisesRegex(ValueError, "采样点"):
            analyze_waveforms({"rows": rows})

    def test_demo_cannot_become_measurement(self):
        demo = make_demo()
        with self.assertRaises(ValueError):
            analyze_waveforms({"rows": demo["waveform_analysis"]["waveforms"], "source": "measured"})
        demo["mode"] = "inference"
        with self.assertRaises(ValueError):
            export_bundle(demo)


class InterchangeTests(unittest.TestCase):
    def test_uploaded_lowercase_and_requested_phase_alias(self):
        for headers in (("mag0", "meg0"), ("mag0", "deg0"), (" Mag0 ", " Deg0 ")):
            memory = io.BytesIO()
            write_workbook(memory, [("input", headers, [[.7, -120]])])
            rows = file_rows({"filename": "input.xlsx", "content_base64": base64.b64encode(memory.getvalue()).decode()})
            self.assertEqual(validate_records(rows), [{"sample_id": "1", "mag": .7, "deg": -120.0}])
        with self.assertRaisesRegex(ValueError, "缺少相位列"):
            validate_records([{"mag0": .7, "wrong": 10}])
        with self.assertRaisesRegex(ValueError, "不一致"):
            validate_records([{"mag0": .7, "deg0": 10, "meg0": 20}])

    def test_returned_reflection_coefficient_has_magnitude_phase_and_components(self):
        demo = make_demo()
        for row in demo["predictions"]:
            real, imag = row["estimated_matched_real"], row["estimated_matched_imag"]
            self.assertAlmostEqual(math.hypot(real, imag), row["estimated_matched_mag"])
            self.assertAlmostEqual(math.degrees(math.atan2(imag, real)), row["estimated_matched_deg"])
            self.assertAlmostEqual(real**2+imag**2, row["estimated_reflected_power_ratio"])
        payload = export_bundle(demo)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
            for field in ("estimated_matched_mag", "estimated_matched_deg", "estimated_matched_real", "estimated_matched_imag"):
                self.assertIn(field, xml)

    def test_xlsx_input_roundtrip(self):
        memory = io.BytesIO()
        write_workbook(memory, [("input", ["Mag0", "Deg0"], [[.4, 60], [.7, -179]])])
        records = file_rows({"filename": "input.xlsx", "content_base64": base64.b64encode(memory.getvalue()).decode()})
        validated = validate_records(records)
        self.assertEqual(validated[1], {"sample_id": "2", "mag": .7, "deg": -179.0})

    def test_export_preserves_every_waveform_sample(self):
        demo = make_demo()
        payload = export_bundle(demo)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                if name.endswith('.xml'):
                    ET.fromstring(archive.read(name))
            wave = ET.fromstring(archive.read('xl/worksheets/sheet4.xml'))
            ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            self.assertEqual(len(wave.findall('x:sheetData/x:row', ns)) - 1, len(demo['waveform_analysis']['waveforms']))

    def test_invalid_inputs_rejected(self):
        for rows in ([{"mag": float('nan'), "deg": 0}], [{"mag": 1.1, "deg": 0}], []):
            with self.assertRaises(ValueError):
                validate_records(rows)


if __name__ == '__main__':
    unittest.main()
