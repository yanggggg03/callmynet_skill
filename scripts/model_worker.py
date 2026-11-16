"""Isolated model process: the original projects use identical module names."""
import json
import sys
from pathlib import Path

from training_identity import checkpoint_identity


def main():
    request = json.load(sys.stdin)
    root = Path(__file__).resolve().parents[1]
    model_id = request["model_id"]
    sys.path.insert(0, request.get('runtime_dir') or str(root / "models" / model_id))
    import numpy as np
    import torch
    from data import to_complex
    from engine import Predictor, initialize, tensor

    device = initialize(request["seed"], request["threads"], request["device"])
    runner = Predictor(request["checkpoint"], device)
    ck = runner.ck
    config = ck["metadata"]["config"]
    info = {
        "id": model_id,
        "data_sha256": ck["data_sha256"],
        "best_epoch": ck["best_epoch"],
        "length_unit": config["length_unit"],
        "magnitude_unit": config["magnitude_unit"],
        "bounds": ck["bounds"].tolist(),
        "bounds_source": ck["metadata"]["bounds_source"],
        "training_source": ck["metadata"].get("source", ""),
        "training_identity": checkpoint_identity(ck, request["checkpoint"], request.get("prepared_data")),
    }
    records = request["records"]
    # The panel always supplies explicit units; all model inputs are converted to complex Gamma.
    raw = to_complex([r["mag"] for r in records], [r["deg"] for r in records], request["magnitude_unit"])
    x = tensor(raw, device)
    if request.get("mode", "generate") == "generate":
        with torch.inference_mode():
            _, _, candidates, _, _, _ = runner.candidates(x, request["candidates"], request["seed"])
        payload = {"metadata": info, "candidates": candidates.cpu().tolist()}
    else:
        # Evaluate EVERY scheme and ALL Flow candidates with this SAME response model.
        batches = request["candidate_sets"]
        output = []
        with torch.inference_mode():
            for index, per_model in enumerate(batches):
                context = (x[index:index + 1] - runner.mean) / runner.std
                per_row = {}
                for source_id, pairs in per_model.items():
                    values = tensor(pairs, device)
                    normalized = (values - runner.low) / runner.span
                    if not torch.isfinite(values).all() or (normalized < -1e-5).any() or (normalized > 1 + 1e-5).any():
                        raise ValueError("Candidate outside common evaluator bounds")
                    contexts = context.expand(len(values), -1)
                    gamma = runner.forward(contexts, normalized)
                    powers = gamma.square().sum(-1)
                    selected = int(powers.argmin())
                    amp = float(gamma[selected].norm())
                    distance = runner.support_distance(context, normalized[selected:selected + 1])
                    per_row[source_id] = {
                        "L1": float(values[selected, 0]), "L2": float(values[selected, 1]),
                        "length_unit": config["length_unit"],
                        "estimated_matched_mag": amp,
                        "estimated_matched_real": float(gamma[selected, 0]),
                        "estimated_matched_imag": float(gamma[selected, 1]),
                        "estimated_matched_deg": float(torch.rad2deg(torch.atan2(gamma[selected, 1], gamma[selected, 0]))),
                        "estimated_reflected_power_ratio": float(powers[selected]),
                        "estimated_return_loss_db": float(-20 * np.log10(max(amp, 1e-8))),
                        "candidate_count": len(pairs), "selected_candidate": selected,
                        "training_support_distance": float(distance[0]),
                        "outside_training_input_box": bool(((context < runner.context_min) | (context > runner.context_max)).any()),
                        "evaluator": model_id,
                        "performance_source": "shared_response_surrogate_estimate",
                    }
                output.append(per_row)
        payload = {"metadata": info, "evaluated": output}
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
