"""Training and inference for measured magnitude/phase -> two electrical lengths."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import SCHEMA, fingerprint, load, local_path, read_records, to_complex
from model import MODEL_NAME
from networks import ResponseSurrogate, build


def initialize(seed, threads, device):
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu")
    return torch.device(device)


def tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def weighted(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(args):
    if (
        min(
            args.epochs,
            args.surrogate_epochs,
            args.width,
            args.surrogate_width,
            args.blocks,
            args.layers,
            args.batch_size,
            args.patience,
            args.candidates,
            args.fourier_features,
        )
        < 1
    ):
        raise ValueError("Epochs, dimensions, counts and patience must be positive")
    if args.lr <= 0 or args.response_weight < 0 or args.fourier_scale <= 0:
        raise ValueError("Invalid learning rate, response weight, or Fourier scale")
    device = initialize(args.seed, args.threads, args.device)
    path = local_path(args.data)
    if not path.is_file():
        raise ValueError(
            f"Measured data not found: {path}. Put your six-column Excel in DATA and run common.py"
        )
    d, meta = load(path)
    out = local_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ti, vi = d["train_idx"], d["val_idx"]
    mean = d["x"][ti].mean(0)
    std = np.maximum(d["x"][ti].std(0), 1e-6)
    bounds = np.asarray(meta["length_bounds"], dtype=np.float32)
    low, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    context = tensor((d["x"] - mean) / std, device)
    lengths = tensor((d["y"] - low) / span, device)
    after = tensor(d["after"], device)
    xtrain, ytrain = context[ti], lengths[ti]
    xval, yval = context[vi], lengths[vi]
    generator = torch.Generator().manual_seed(args.seed)
    # Surrogate learns from ALL training actions including poor matching records.
    forward = ResponseSurrogate(args.surrogate_width, 3).to(device)
    opt = torch.optim.AdamW(forward.parameters(), lr=args.lr, weight_decay=1e-5)
    dl = DataLoader(
        TensorDataset(xtrain, ytrain, after[ti]),
        args.batch_size,
        shuffle=True,
        generator=generator,
    )
    history = {"surrogate": [], "inverse": []}
    best, stale, best_forward = float("inf"), 0, None
    print(
        f"Model={MODEL_NAME}; data={path}; splits={[len(d[k+'_idx']) for k in ('train','val','test')]}; length_unit={meta['config']['length_unit']}",
        flush=True,
    )
    for epoch in range(1, args.surrogate_epochs + 1):
        forward.train()
        total = 0.0
        for x, y, target in dl:
            loss = (forward(x, y) - target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite forward loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forward.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(x)
        forward.eval()
        with torch.no_grad():
            val = float((forward(xval, yval) - after[vi]).square().mean())
        history["surrogate"].append(
            {"epoch": epoch, "train_mse": total / len(ti), "val_complex_mse": val}
        )
        print(f"forward epoch={epoch} val_complex_mse={val:.6f}", flush=True)
        if val < best:
            best, stale, best_forward = val, 0, state_cpu(forward)
        else:
            stale += 1
        if stale >= args.patience:
            break
    forward.load_state_dict(best_forward)
    forward.eval()
    forward.requires_grad_(False)
    config = {
        "width": args.width,
        "blocks": args.blocks,
        "layers": args.layers,
        "fourier_features": args.fourier_features,
        "fourier_scale": args.fourier_scale,
    }
    inverse = build(MODEL_NAME, config).to(device)
    quality = np.sum(d["after"] ** 2, axis=1)
    cfg = meta["config"]
    weights = np.maximum(
        cfg["quality_weight_floor"], np.exp(-quality / cfg["quality_power_scale"])
    )
    mask = np.ones(len(quality), dtype=bool)
    if cfg["inverse_max_matched_mag"] is not None:
        mask = quality <= cfg["inverse_max_matched_mag"] ** 2
    itr, iva = ti[mask[ti]], vi[mask[vi]]
    if not len(itr) or not len(iva):
        raise ValueError(
            "No accepted inverse train/validation rows; relax inverse_max_matched_mag"
        )
    w = tensor(weights, device)
    train_loader = DataLoader(
        TensorDataset(context[itr], lengths[itr], w[itr]),
        args.batch_size,
        shuffle=True,
        generator=generator,
    )
    opt = torch.optim.AdamW(inverse.parameters(), lr=args.lr, weight_decay=1e-5)
    best, stale, best_inverse, best_epoch = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        inverse.train()
        total = 0.0
        for x, y, weight in train_loader:
            if inverse.is_flow:
                supervision = weighted(-inverse.log_prob(x, y), weight)
                samples = inverse.sample(x, 8)
                xx = x[:, None, :].expand(-1, 8, -1).reshape(-1, 2)
                power = (
                    forward(xx, samples.reshape(-1, 2))
                    .square()
                    .sum(-1)
                    .reshape(-1, 8)
                    .mean(-1)
                )
            else:
                predicted, sigma = inverse(x)
                error = ((predicted - y) / sigma).square() * 0.5 + torch.log(sigma)
                supervision = weighted(error.mean(-1), weight)
                power = forward(x, predicted).square().sum(-1)
            # Learned response regularization, NOT ideal physics or measured performance.
            loss = supervision + args.response_weight * weighted(power, weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite inverse loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inverse.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(x)
        inverse.eval()
        with torch.no_grad():
            if inverse.is_flow:
                val = float(
                    weighted(-inverse.log_prob(context[iva], lengths[iva]), w[iva])
                )
            else:
                pred, _ = inverse(context[iva])
                val = float(weighted((pred - lengths[iva]).square().mean(-1), w[iva]))
        history["inverse"].append(
            {
                "epoch": epoch,
                "train_loss": total / len(itr),
                "val_supervised_score": val,
            }
        )
        print(
            f"inverse epoch={epoch} loss={total/len(itr):.6f} val={val:.6f}", flush=True
        )
        if val < best:
            best, stale, best_inverse, best_epoch = val, 0, state_cpu(inverse), epoch
        else:
            stale += 1
        if stale >= args.patience:
            break
    ck = {
        "schema": SCHEMA,
        "model_name": MODEL_NAME,
        "config": config,
        "state": best_inverse,
        "forward_config": {"width": args.surrogate_width, "blocks": 3},
        "forward_state": best_forward,
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "bounds": torch.from_numpy(bounds),
        "metadata": meta,
        "data_sha256": fingerprint(path),
        "training_args": vars(args),
        "best_epoch": best_epoch,
        "inverse_accepted_train_rows": len(itr),
        "candidates": args.candidates,
        "validation_surrogate_complex_mse": history["surrogate"][
            np.argmin([v["val_complex_mse"] for v in history["surrogate"]])
        ]["val_complex_mse"],
        "support_x": context[ti].cpu(),
        "support_y": lengths[ti].cpu(),
        "quality_weighting": "training labels weighted by measured matched reflection power",
    }
    # New runs avoid overwriting any old nine-input checkpoint.
    torch.save(ck, out / "best.pt")
    save_json(out / "history.json", history)
    save_json(
        out / "training_summary.json",
        {
            "schema": SCHEMA,
            "model": MODEL_NAME,
            "data_sha256": ck["data_sha256"],
            "best_epoch": best_epoch,
            "length_unit": cfg["length_unit"],
            "metadata": meta,
            "parameters": sum(p.numel() for p in inverse.parameters()),
        },
    )
    print(f"Saved measured checkpoint: {out/'best.pt'}", flush=True)


class Predictor:
    def __init__(self, path, device="cpu"):
        path = local_path(path)
        if not path.is_file():
            raise ValueError(
                f"No measured checkpoint at {path}. Run train.py with your measured table first"
            )
        # Trusted local checkpoints only; torch checkpoints may contain pickle payloads.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(ck, dict)
            or ck.get("schema") != SCHEMA
            or ck.get("model_name") != MODEL_NAME
        ):
            raise ValueError(
                "Wrong model or old nine-input checkpoint. Retrain this project's measured model"
            )
        self.ck, self.device = ck, torch.device(device)
        self.inverse = build(MODEL_NAME, ck["config"]).to(self.device)
        self.inverse.load_state_dict(ck["state"])
        self.inverse.eval()
        self.forward = ResponseSurrogate(**ck["forward_config"]).to(self.device)
        self.forward.load_state_dict(ck["forward_state"])
        self.forward.eval()
        self.mean = ck["mean"].to(self.device)
        self.std = ck["std"].to(self.device)
        bounds = ck["bounds"].to(self.device)
        self.low, self.span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
        self.context_min = ck["support_x"].min(0).values.to(self.device)
        self.context_max = ck["support_x"].max(0).values.to(self.device)

    @torch.inference_mode()
    def candidates(self, raw, count=32, seed=42):
        context = (raw - self.mean) / self.std
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if self.inverse.is_flow:
            choices = self.inverse.sample(context, count, generator)
        else:
            choices = self.inverse(context)[0][:, None, :]
        xx = context[:, None, :].expand(-1, choices.shape[1], -1)
        gamma = self.forward(xx.reshape(-1, 2), choices.reshape(-1, 2)).reshape(
            *choices.shape[:2], 2
        )
        power = gamma.square().sum(-1)
        best = power.argmin(-1)
        ar = torch.arange(len(raw), device=self.device)
        selected = choices[ar, best]
        return (
            selected * self.span + self.low,
            gamma[ar, best],
            choices * self.span + self.low,
            best,
            context,
            selected,
        )

    @torch.inference_mode()
    def support_distance(self, context, selected):
        query = torch.cat([context, selected], -1)
        best = torch.full((len(query),), float("inf"), device=self.device)
        sx, sy = self.ck["support_x"], self.ck["support_y"]
        for i in range(0, len(sx), 2048):
            reference = torch.cat([sx[i : i + 2048], sy[i : i + 2048]], -1).to(
                self.device
            )
            best = torch.minimum(best, torch.cdist(query, reference).min(-1).values)
        return best


def evaluate(args):
    device = initialize(args.seed, args.threads, args.device)
    path = local_path(args.data)
    d, meta = load(path)
    runner = Predictor(args.checkpoint, device)
    if fingerprint(path) != runner.ck["data_sha256"]:
        raise ValueError(
            "Evaluation requires the exact prepared NPZ used for training, including its fixed split"
        )
    ids = d["test_idx"]
    predictions, estimates, recorded_estimates, distances = [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(ids), args.batch_size):
            idx = ids[i : i + args.batch_size]
            raw = tensor(d["x"][idx], device)
            target = tensor(d["y"][idx], device)
            pred, gamma, _, _, context, normalized = runner.candidates(
                raw, args.candidates, args.seed + i
            )
            predictions.append(pred.cpu().numpy())
            estimates.append(gamma.cpu().numpy())
            recorded_estimates.append(
                runner.forward(context, (target - runner.low) / runner.span)
                .cpu()
                .numpy()
            )
            distances.append(runner.support_distance(context, normalized).cpu().numpy())
    pred, estimated = np.concatenate(predictions), np.concatenate(estimates)
    forward_on_recorded = np.concatenate(recorded_estimates)
    target = d["y"][ids]
    observed = d["after"][ids]
    metrics = {
        "schema": SCHEMA,
        "model": MODEL_NAME,
        "test_rows": len(ids),
        "length_unit": meta["config"]["length_unit"],
        "candidates": args.candidates if runner.inverse.is_flow else 1,
        "length_mae_per_column": np.abs(pred - target).mean(0).tolist(),
        "length_rmse_per_column": np.sqrt(((pred - target) ** 2).mean(0)).tolist(),
        "surrogate_test_complex_mse_at_recorded_lengths": float(
            ((forward_on_recorded - observed) ** 2).mean()
        ),
        "surrogate_test_amplitude_mae_at_recorded_lengths": float(
            np.abs(
                np.linalg.norm(forward_on_recorded, axis=1)
                - np.linalg.norm(observed, axis=1)
            ).mean()
        ),
        "recorded_test_mean_reflected_power": float((observed**2).sum(-1).mean()),
        "estimated_selected_mean_reflected_power": float((estimated**2).sum(-1).mean()),
        "estimated_selected_success_20db": float(
            ((estimated**2).sum(-1) <= 0.01).mean()
        ),
        "mean_selected_training_support_distance": float(
            np.concatenate(distances).mean()
        ),
        "limitation": "Selected lengths were not applied to hardware. Estimated success is surrogate-only; length error compares one recorded action and cannot prove optimality.",
        "data_sha256": runner.ck["data_sha256"],
        "best_epoch": runner.ck["best_epoch"],
    }
    out = local_path(args.output)
    save_json(out / "metrics.json", metrics)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source_row",
                "L1_recorded",
                "L2_recorded",
                "L1_recommended",
                "L2_recommended",
                "estimated_matched_mag",
                "observed_recorded_matched_mag",
                "length_unit",
            ]
        )
        for j, idx in enumerate(ids):
            w.writerow(
                [
                    int(d["row_ids"][idx]),
                    *target[j],
                    *pred[j],
                    np.linalg.norm(estimated[j]),
                    np.linalg.norm(observed[j]),
                    meta["config"]["length_unit"],
                ]
            )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def predict(args):
    output = local_path(args.output) if args.output else None
    if args.input and output is None:
        output = local_path(f"results/{Path(args.input).stem}_predictions.xlsx")
    if output:
        if output.suffix.lower() not in (".xlsx", ".json", ".csv"):
            raise ValueError("Prediction output must be .xlsx, .csv, or .json")
        if args.input and output.resolve() == local_path(args.input).resolve():
            raise ValueError(
                "Output must differ from input; the source table is preserved"
            )
    device = initialize(args.seed, args.threads, args.device)
    runner = Predictor(args.checkpoint, device)
    if args.input:
        records, row_ids = read_records(
            local_path(args.input), args.sheet, prediction=True
        )
        mags, phases = [r["Mag0"] for r in records], [r["Deg0"] for r in records]
    else:
        if args.mag is None or args.deg is None:
            raise ValueError(
                "Provide --mag and --deg, or --input a table containing Mag0,Deg0"
            )
        mags, phases, row_ids = [args.mag], [args.deg], [None]
    unit = args.magnitude_unit or runner.ck["metadata"]["config"]["magnitude_unit"]
    raw = to_complex(mags, phases, unit)
    results = []
    for i in range(0, len(raw), args.batch_size):
        x = tensor(raw[i : i + args.batch_size], device)
        lengths, gamma, candidates, best, context, normalized = runner.candidates(
            x, args.candidates, args.seed + i
        )
        distance = runner.support_distance(context, normalized)
        for j in range(len(x)):
            amplitude = float(gamma[j].norm())
            bounds = runner.ck["bounds"].numpy()
            item = {
                "mag": float(mags[i + j]),
                "deg": float(phases[i + j]),
                "magnitude_unit": unit,
                "source_row": (
                    int(row_ids[i + j]) if row_ids[i + j] is not None else None
                ),
                "L1": float(lengths[j, 0]),
                "L2": float(lengths[j, 1]),
                "length_unit": runner.ck["metadata"]["config"]["length_unit"],
                "estimated_matched_mag": amplitude,
                "estimated_matched_deg": float(
                    torch.rad2deg(torch.atan2(gamma[j, 1], gamma[j, 0]))
                ),
                "estimated_return_loss_db": float(-20 * np.log10(max(amplitude, 1e-6))),
                "candidate_count": candidates.shape[1],
                "selected_candidate": int(best[j]),
                "training_support_distance": float(distance[j]),
                "outside_training_input_box": bool(
                    (
                        (context[j] < runner.context_min)
                        | (context[j] > runner.context_max)
                    ).any()
                ),
                "bounds": bounds.tolist(),
                "bounds_source": runner.ck["metadata"]["bounds_source"],
                "performance_source": "learned_measured_response_surrogate_not_new_measurement",
                "note": "Recommended candidate, not a certified global optimum. Small-amplitude reflection phase is noise-sensitive.",
            }
            if args.include_candidates:
                item["candidates"] = candidates[j].cpu().tolist()
            results.append(item)
    payload = results[0] if len(results) == 1 else results
    if output and output.suffix.lower() == ".xlsx":
        from excel_io import export_predictions

        export_predictions(
            output,
            results,
            {
                "模型": MODEL_NAME,
                "权重路径": str(local_path(args.checkpoint).resolve()),
                "输入文件": (
                    str(local_path(args.input).resolve())
                    if args.input
                    else "命令行单条输入"
                ),
                "输入幅值单位": unit,
                "输入相位单位": "deg（°）",
                "输出长度单位": runner.ck["metadata"]["config"]["length_unit"],
                "预计匹配幅值单位": "线性反射系数幅值，0～1",
                "预计匹配相位单位": "deg（°）",
                "结果含义": "L1、L2是推荐支节电长度；estimated字段为响应代理预测，尚未经新实测确认，也不保证全局最优。",
                "数据对应": "predictions按输入记录顺序逐行输出，mag、deg保留输入数值。source_row标识来源行。",
                "候选编号": "selected_candidate与candidate_index从0计数；Flow选择评分最低候选，不取多解平均。",
                "训练数据来源": runner.ck["metadata"].get(
                    "provenance", "见权重内metadata与训练数据来源"
                ),
                "范围与距离": "bounds为训练配置范围；training_support_distance不是概率，outside_training_input_box提示输入超出训练范围。",
            },
        )
    elif output and output.suffix.lower() == ".csv":
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = list(results[0])
        with output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            for item in results:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (list, dict))
                            else value
                        )
                        for key, value in item.items()
                    }
                )
    elif output:
        save_json(output, payload)
    if args.input or (output and output.suffix.lower() == ".xlsx"):
        print(
            json.dumps(
                {
                    "saved": str(output.resolve()),
                    "rows": len(results),
                    "model": MODEL_NAME,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
