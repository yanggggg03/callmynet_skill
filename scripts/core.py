"""Model orchestration, strict waveform comparison, and XLSX interchange."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from excel_writer import write_workbook
from training_identity import validate_comparison

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("residual_mlp", "fourier_resmlp", "conditional_flow")
LABELS = dict(zip(MODEL_IDS, ("残差 MLP", "Fourier ResMLP", "条件 Flow")))
LIMIT_ROWS = 1000


def finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} 必须是有限数值") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def load_config(path=None):
    path = Path(path or ROOT / "config/models.json").resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if [m["id"] for m in config["models"]] != list(MODEL_IDS):
        raise ValueError("配置必须按顺序包含全部三个模型")
    if config["evaluation_model"] not in MODEL_IDS:
        raise ValueError("未知的共用响应评估器")
    config["python_executable"] = config.get("python_executable") or sys.executable
    for model in config["models"]:
        checkpoint = Path(model["checkpoint"]).expanduser()
        model["checkpoint"] = str((ROOT / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve())
        if model.get("prepared_data"):
            prepared = Path(model["prepared_data"]).expanduser()
            model["prepared_data"] = str((ROOT / prepared).resolve() if not prepared.is_absolute() else prepared.resolve())
    return config


def status(config):
    return {
        "service": "callmynet-skill", "version": "1.0.2",
        "models": [{"id": m["id"], "label": m.get("label", LABELS[m["id"]]),
                    "checkpoint_ready": Path(m["checkpoint"]).is_file()} for m in config["models"]],
        "evaluator": config["evaluation_model"],
        "capability": "static_matching_with_optional_external_power_waveforms",
    }


def validate_records(records, unit="linear"):
    if unit not in ("linear", "db"):
        raise ValueError("幅值单位只能是 linear 或 db")
    if not isinstance(records, list) or not 1 <= len(records) <= LIMIT_ROWS:
        raise ValueError(f"输入需要 1～{LIMIT_ROWS} 条记录")
    output = []
    for index, row in enumerate(records, 1):
        mag = finite(row.get("mag", row.get("Mag0")), f"第{index}条 mag")
        deg = finite(row.get("deg", row.get("Deg0")), f"第{index}条 deg")
        if (unit == "linear" and not 0 <= mag <= 1) or (unit == "db" and mag > 0):
            raise ValueError("线性反射幅值须在0～1内，dB幅值须≤0")
        output.append({"sample_id": str(index), "mag": mag, "deg": deg})
    return output


def file_rows(payload):
    """Read CSV/XLSX upload; formulas without saved values fail downstream."""
    filename = str(payload.get("filename", "input.csv"))
    content = base64.b64decode(payload["content_base64"], validate=True)
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = content.decode("gb18030")
        try:
            dialect = csv.Sniffer().sniff(decoded[:8192], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(io.StringIO(decoded), dialect))
    elif suffix == ".xlsx":
        import zipfile
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(v.file_size for v in archive.infolist()) > 100_000_000:
                raise ValueError("工作簿解压后过大")
        spec = importlib.util.spec_from_file_location("matching_xlsx_reader", ROOT / "models/residual_mlp/data.py")
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        rows = list(reader._xlsx_rows(io.BytesIO(content), payload.get("sheet") or None))
    else:
        raise ValueError("仅支持 .csv 和 .xlsx")
    if not rows:
        raise ValueError("表格为空")
    headers = [str(v).strip() for v in rows[0]]
    if len(headers) != len(set(headers)):
        raise ValueError("表头重复")
    records = [dict(zip(headers, row)) for row in rows[1:] if any(str(v).strip() for v in row)]
    if not records:
        raise ValueError("表格没有数据行")
    return records


def worker(config, model, request):
    command = [config["python_executable"], "-B", str(ROOT / "scripts/model_worker.py")]
    body = {"model_id": model["id"], "checkpoint": model["checkpoint"],
            "seed": config.get("seed", 42), "threads": config.get("threads", 2),
            "device": config.get("device", "cpu"), "candidates": config.get("flow_candidates", 32), **request}
    if model.get("prepared_data"):
        body["prepared_data"] = model["prepared_data"]
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    result = subprocess.run(command, input=json.dumps(body), text=True, encoding="utf-8",
                            capture_output=True, timeout=config.get("timeout_seconds", 180), **flags)
    if result.returncode:
        raise ValueError(f"{LABELS[model['id']]} 推理失败：{result.stderr.strip()[-1800:]}")
    return json.loads(result.stdout)


def run_models(config, request):
    unit = request.get("magnitude_unit", "linear")
    records = validate_records(request.get("records"), unit)
    missing = [LABELS[m["id"]] for m in config["models"] if not Path(m["checkpoint"]).is_file()]
    if missing:
        raise ValueError("尚未找到正式训练权重：" + "、".join(missing) + "。请完成训练并修改 config/models.json；示例按钮可先预览。")
    generated = []
    for model in config["models"]:
        generated.append(worker(config, model, {"records": records, "magnitude_unit": unit}))
    metadata = [g["metadata"] for g in generated]
    comparison_verification = validate_comparison(metadata) if config.get("require_same_training_data", True) else "disabled_by_configuration"
    if len({m["length_unit"] for m in metadata}) != 1 or any(m["bounds"] != metadata[0]["bounds"] for m in metadata):
        raise ValueError("三个模型的电长度单位或长度范围不一致")
    candidate_sets = [{m["id"]: generated[j]["candidates"][i] for j, m in enumerate(config["models"])} for i in range(len(records))]
    evaluator = next(m for m in config["models"] if m["id"] == config["evaluation_model"])
    scored = worker(config, evaluator, {"mode": "score", "records": records, "magnitude_unit": unit, "candidate_sets": candidate_sets})
    results, provisional = [], []
    for record, per_model in zip(records, scored["evaluated"]):
        for model_id, prediction in per_model.items():
            results.append({**record, "model_id": model_id, "model_label": LABELS[model_id], **prediction})
        best = min(per_model, key=lambda mid: per_model[mid]["estimated_reflected_power_ratio"])
        provisional.append({**record, "model_id": best, **per_model[best]})
    bundle = {"mode": "inference", "created_at": datetime.now(timezone.utc).isoformat(),
              "magnitude_unit": unit, "records": records, "predictions": results,
              "estimated_recommendations": provisional, "metadata": metadata,
              "evaluation_model": config["evaluation_model"],
              "training_comparison_verification": comparison_verification,
              "selection_status": "awaiting_waveforms", "waveforms": [],
              "notice": "预计结果由同一个响应代理评分；动态稳定性须由同工况实测或仿真功率波形验证。"}
    # Bind future waveforms to this exact inference run; no stale result reuse.
    bundle["run_id"] = hashlib.sha256(json.dumps(bundle, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return bundle


def analyze_waveforms(request):
    """Compare one sample and one run using a common time grid and target-relative limits."""
    rows = request.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) > 300_000:
        raise ValueError("功率表需要1～300000条记录")
    source = request.get("source", "measured")
    if source not in ("measured", "simulation", "demo"):
        raise ValueError("波形来源应为 measured、simulation 或 demo")
    required = ("run_id", "sample_id", "scenario_id", "model_id", "time_s", "power_w", "target_w")
    groups = {}
    power_channels = {}
    identities = set()
    sources_in_rows = set()
    for row in rows:
        if any(row.get(key) is None or str(row.get(key)).strip() == "" for key in required):
            raise ValueError("波形必须包含 " + ",".join(required))
        model_id = str(row["model_id"]).strip()
        if model_id not in MODEL_IDS:
            raise ValueError(f"未知 model_id：{model_id}")
        identity = tuple(str(row[k]).strip() for k in ("run_id", "sample_id", "scenario_id"))
        identities.add(identity)
        if str(row.get("source", "")).strip():
            sources_in_rows.add(str(row["source"]).strip())
        time = finite(row["time_s"], "time_s")
        power, target = finite(row["power_w"], "power_w"), finite(row["target_w"], "target_w")
        if time < 0 or target <= 0 or power < 0:
            raise ValueError("time_s与power_w必须≥0，target_w必须>0")
        has_incident = str(row.get("incident_power_w", "")).strip() != ""
        has_reflected = str(row.get("reflected_power_w", "")).strip() != ""
        if has_incident != has_reflected:
            raise ValueError("入射和反射功率必须同时提供")
        if has_incident:
            incident = finite(row["incident_power_w"], "incident_power_w")
            reflected = finite(row["reflected_power_w"], "reflected_power_w")
            if not 0 <= reflected <= incident or not math.isclose(power, incident-reflected, rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError("须满足0≤反射功率≤入射功率，且power_w=入射功率-反射功率")
            power_channels[(model_id, time)] = {"incident_power_w": incident, "reflected_power_w": reflected,
                                               "reflected_power_ratio": reflected / incident if incident > 0 else None}
        groups.setdefault(model_id, []).append((time, power, target))
    if len(identities) != 1:
        raise ValueError("一次比较仅支持同一 run_id、sample_id、scenario_id，请按工况分开导入")
    run_id, sample_id, scenario_id = next(iter(identities))
    if sources_in_rows and sources_in_rows != {source}:
        raise ValueError("表格source与所选来源不一致，且不得混用实测、仿真、示例")
    if ("demo" in sources_in_rows or run_id.startswith("demo")) and source != "demo":
        raise ValueError("示例波形必须标记为 demo，不能用作真实调节结果")
    expected = request.get("expected_run_id")
    if expected and expected != run_id:
        raise ValueError("波形 run_id 与当前推理结果不一致，请使用本次调节的数据")
    options = request.get("options") or {}
    defaults = {"steady_start_fraction": 0.6, "max_cv_pct": 2.0, "max_ripple_pct": 5.0,
                "max_error_pct": 2.0, "max_drift_pct": 1.0, "max_overshoot_pct": 20.0,
                "min_steady_samples": 20, "min_duration_s": 0.1}
    settings = {key: finite(options.get(key, default), key) for key, default in defaults.items()}
    for key in ("evaluation_start_s", "evaluation_end_s"):
        if options.get(key) not in (None, ""):
            settings[key] = finite(options[key], key)
    if not 0 < settings["steady_start_fraction"] < 1 or any(v <= 0 for k, v in settings.items() if k not in ("steady_start_fraction", "evaluation_start_s", "evaluation_end_s")):
        raise ValueError("稳态窗口比例须在0～1内，各限值须为正数")
    if settings["min_steady_samples"] < 3 or not settings["min_steady_samples"].is_integer():
        raise ValueError("min_steady_samples须为≥3的整数")
    metrics, grid, common_target = [], None, None
    normalized_rows = []
    for model_id in MODEL_IDS:
        if model_id not in groups:
            raise ValueError("需要全部三个模型在相同工况下的波形")
        values = sorted(groups[model_id])
        times = [v[0] for v in values]
        if len(times) != len(set(times)):
            raise ValueError("同一模型存在重复时间戳")
        if grid is None:
            grid = times
        elif len(grid) != len(times) or any(not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-9) for a, b in zip(grid, times)):
            raise ValueError("三套波形须使用相同时间网格和观察时长；请在导出端统一采样")
        target = values[0][2]
        if any(not math.isclose(target, v[2], rel_tol=1e-8) for v in values):
            raise ValueError("本版本每次比较使用固定目标功率；阶跃目标请按目标段分开")
        if common_target is None:
            common_target = target
        elif not math.isclose(target, common_target, rel_tol=1e-8):
            raise ValueError("三个模型目标功率必须相同")
        evaluation_start = settings.get("evaluation_start_s", times[0])
        evaluation_end = settings.get("evaluation_end_s", times[-1])
        if not times[0] <= evaluation_start < evaluation_end <= times[-1]:
            raise ValueError("评估窗口须在波形时间范围内，且起点小于终点")
        evaluation_values = [v for v in values if evaluation_start <= v[0] <= evaluation_end]
        if len(evaluation_values) < 3:
            raise ValueError("评估窗口采样点不足")
        evaluation_times = [v[0] for v in evaluation_values]
        duration = evaluation_times[-1] - evaluation_times[0]
        if duration < settings["min_duration_s"]:
            raise ValueError("观察时长不足，请补充数据或按装置时间尺度配置最小时长")
        steady_start = evaluation_times[0] + duration * settings["steady_start_fraction"]
        tail = [v for v in evaluation_values if v[0] >= steady_start]
        if len(tail) < settings["min_steady_samples"]:
            raise ValueError("稳态窗口采样点不足")
        powers = [v[1] for v in tail]
        mean = statistics.fmean(powers)
        std = statistics.pstdev(powers)
        cv = std / max(abs(mean), 1e-12) * 100
        ripple = (max(powers) - min(powers)) / target * 100
        error = abs(mean - target) / target * 100
        center = statistics.fmean([v[0] for v in tail])
        denominator = sum((v[0] - center) ** 2 for v in tail)
        slope = sum((v[0] - center) * (v[1] - mean) for v in tail) / denominator
        drift = abs(slope) * (tail[-1][0] - tail[0][0]) / target * 100
        overshoot = max(0, max(v[1] for v in evaluation_values) - target) / target * 100
        ratios = [cv / settings["max_cv_pct"], ripple / settings["max_ripple_pct"],
                  error / settings["max_error_pct"], drift / settings["max_drift_pct"],
                  overshoot / settings["max_overshoot_pct"]]
        stable = all(r <= 1 for r in ratios)
        settling = None
        last_bad = max((i for i, v in enumerate(evaluation_values) if abs(v[1] - target) / target * 100 > settings["max_error_pct"]), default=-1)
        if last_bad + 1 < len(evaluation_values) and evaluation_times[-1] - evaluation_times[last_bad + 1] >= duration * (1 - settings["steady_start_fraction"]):
            settling = evaluation_times[last_bad + 1] - evaluation_times[0]
        metrics.append({"model_id": model_id, "model_label": LABELS[model_id], "stable": stable,
                        "mean_power_w": mean, "std_power_w": std, "cv_pct": cv, "ripple_pct": ripple,
                        "error_pct": error, "drift_pct": drift, "overshoot_pct": overshoot,
                        "settling_time_s": settling, "score": statistics.fmean(ratios),
                        "steady_samples": len(tail), "duration_s": duration,
                        "evaluation_start_s": evaluation_times[0], "evaluation_end_s": evaluation_times[-1],
                        "steady_start_s": steady_start})
        normalized_rows.extend({"run_id": run_id, "sample_id": sample_id, "scenario_id": scenario_id,
                                "model_id": model_id, "time_s": t, "power_w": p, "target_w": target,
                                "source": source, **power_channels.get((model_id, t), {})} for t, p, _ in values)
    if power_channels and len(power_channels) != len(rows):
        raise ValueError("入射与反射通道需覆盖全部三个模型的全部采样点")
    ranked = sorted(metrics, key=lambda m: (not m["stable"], m["score"], m["model_id"]))
    stable = [m for m in ranked if m["stable"]]
    return {"run_id": run_id, "sample_id": sample_id, "scenario_id": scenario_id, "source": source,
            "settings": settings, "metrics": ranked, "waveforms": normalized_rows,
            "winner": stable[0]["model_id"] if stable else None,
            "lowest_score_model": ranked[0]["model_id"],
            "selection_status": "stable_winner" if stable else "none_stable",
            "interpretation": "有限观察窗口内满足当前阈值，不等于数学意义的全局稳定性证明。"}


def export_bundle(bundle):
    predictions = bundle.get("predictions", [])
    analysis = bundle.get("waveform_analysis") or {}
    if analysis and bundle.get("run_id") and bundle["run_id"] != analysis.get("run_id"):
        raise ValueError("推理与波形批次不一致，不能合并导出")
    if bundle.get("mode") == "inference" and analysis.get("source") == "demo":
        raise ValueError("合成示例不能验证真实推理")
    sheets = []
    headers = ("sample_id", "model_id", "mag", "deg", "L1", "L2", "length_unit", "estimated_matched_mag", "estimated_matched_deg", "estimated_matched_real", "estimated_matched_imag", "estimated_reflected_power_ratio", "estimated_return_loss_db", "evaluator", "outside_training_input_box", "training_support_distance")
    sheets.append(("三模型结果", headers, [[r.get(h) for h in headers] for r in predictions]))
    selected = bundle.get("estimated_recommendations", [])
    sheets.append(("预计推荐_待波形验证", headers, [[r.get(h) for h in headers] for r in selected]))
    metric_headers = ("model_id", "stable", "score", "mean_power_w", "cv_pct", "ripple_pct", "error_pct", "drift_pct", "overshoot_pct", "settling_time_s", "steady_samples", "duration_s")
    sheets.append(("稳定性评估", metric_headers, [[m.get(h) for h in metric_headers] for m in analysis.get("metrics", [])]))
    wave_headers = ("run_id", "sample_id", "scenario_id", "model_id", "time_s", "power_w", "target_w", "source", "incident_power_w", "reflected_power_w", "reflected_power_ratio")
    sheets.append(("功率波形", wave_headers, [[r.get(h) for h in wave_headers] for r in analysis.get("waveforms", [])]))
    winner = analysis.get("winner")
    confirmed = [r for r in predictions if r["model_id"] == winner and str(r["sample_id"]) == str(analysis.get("sample_id")) and bundle.get("run_id") == analysis.get("run_id")]
    sheets.append(("波形选优结果", headers, [[r.get(h) for h in headers] for r in confirmed]))
    notes = {"模式": bundle.get("mode", "waveforms_only"), "推理编号": bundle.get("run_id", analysis.get("run_id")),
             "输入幅值单位": bundle.get("magnitude_unit", ""), "共用响应评估器": bundle.get("evaluation_model", ""),
             "波形来源": analysis.get("source", "未接入"), "波形选择状态": analysis.get("selection_status", "未接入"),
             "波形优胜模型": winner or "没有已验证稳定方案", "阈值": analysis.get("settings", {}),
             "说明": "estimated字段为代理估计；demo为合成示例；稳定性结论仅适用于当前观察窗口和阈值。",
             "模型记录": bundle.get("metadata", [])}
    notes["波形形态参考"] = bundle.get("waveform_reference", {})
    notes["主要返回量"] = "匹配后复反射系数：estimated_matched_mag为线性幅值|Γ|，无量纲、非百分比；estimated_matched_deg为相位°；real/imag为实部/虚部。反射功率比为辅助量|Γ|²。"
    if analysis and bundle.get("mode") == "inference" and (analysis.get("run_id") != bundle.get("run_id") or analysis.get("source") == "demo"):
        raise ValueError("不能用示例波形或其他推理批次的波形验证真实推荐")
    sheets.append(("说明", ("项目", "内容"), list(notes.items())))
    memory = io.BytesIO()
    write_workbook(memory, sheets)
    return memory.getvalue()
