"""Synthetic UI fixtures. No trained-model or experimental-performance claims."""
import math

from core import LABELS, MODEL_IDS, analyze_waveforms


def make_demo():
    run_id = "demo-ui-only"
    records = [{"sample_id": "1", "mag": 0.4, "deg": 60.0}, {"sample_id": "2", "mag": 0.6, "deg": -45.0}]
    predictions = []
    for record in records:
        for index, model_id in enumerate(MODEL_IDS):
            amplitude = [0.068, 0.051, 0.032][index] + (int(record["sample_id"]) - 1) * 0.009
            predictions.append({**record, "model_id": model_id, "model_label": LABELS[model_id],
                                "L1": 82.4 + index * 1.7 + int(record["sample_id"]),
                                "L2": 104.8 - index * 2.3, "length_unit": "deg",
                                "estimated_matched_mag": amplitude,
                                "estimated_reflected_power_ratio": amplitude ** 2,
                                "estimated_return_loss_db": -20 * math.log10(amplitude),
                                "evaluator": "residual_mlp", "outside_training_input_box": False,
                                "training_support_distance": 0.2, "performance_source": "synthetic_demo"})
    rows = []
    for index, model_id in enumerate(MODEL_IDS):
        for step in range(401):
            t = step / 100
            noise = [16, 9, 2.2][index] * math.sin(t * 34) + [5, 3, 0.8][index] * math.sin(t * 57)
            transient = 130 * math.exp(-t * [2, 3, 5][index]) * math.cos(t * 10)
            rows.append({"run_id": run_id, "sample_id": "1", "scenario_id": "example-1000W",
                         "model_id": model_id, "time_s": t, "power_w": 1000 + transient + noise,
                         "target_w": 1000, "source": "demo"})
    analysis = analyze_waveforms({"rows": rows, "source": "demo"})
    return {"mode": "demo", "run_id": run_id, "magnitude_unit": "linear", "records": records,
            "predictions": predictions, "estimated_recommendations": [next(r for r in predictions if r["sample_id"] == row["sample_id"] and r["model_id"] == "conditional_flow") for row in records],
            "metadata": [], "evaluation_model": "residual_mlp", "waveform_analysis": analysis,
            "notice": "合成示例仅用于演示面板、波形评估与Excel导出，不代表三个模型的真实表现。"}
