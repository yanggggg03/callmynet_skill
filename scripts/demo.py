"""Paper-shaped synthetic RF pulse; no measured/model-performance claims."""
import math

from core import LABELS, MODEL_IDS, analyze_waveforms


def make_demo():
    run_id = "demo-paper-shaped-20kw"
    records = [{"sample_id": "1", "mag": -3.8, "deg": -127.2},
               {"sample_id": "2", "mag": -2.4, "deg": -117.6}]
    # Preset illustrative actions near the screenshot coverage. These are NOT network inference.
    actions = [(50.10,39.20,.055),(51.30,39.35,.052),(52.70,39.15,.061)]
    predictions = []
    for record in records:
        for index, model_id in enumerate(MODEL_IDS):
            l1,l2,amplitude=actions[index]
            predictions.append({**record, "model_id": model_id, "model_label": LABELS[model_id],
                                "L1": l1+(1.8 if record["sample_id"]=="2" else 0),
                                "L2": l2, "length_unit": "deg",
                                "estimated_matched_mag": amplitude,
                                "estimated_reflected_power_ratio": amplitude ** 2,
                                "estimated_return_loss_db": -20 * math.log10(amplitude),
                                "evaluator": "synthetic_preset_not_network", "outside_training_input_box": False,
                                "training_support_distance": None, "performance_source": "synthetic_demo"})
    rows = []
    for index, model_id in enumerate(MODEL_IDS):
        final_ratio=actions[index][2]**2
        initial_ratio=10**(records[0]["mag"]/10)
        for step in range(701):
            t=round(step/100,2)
            if 1<=t<6:
                elapsed=t-1
                incident=20000*(1+.0015*math.sin(81*t)+.0008*math.sin(143*t))
                progress=max(0,elapsed-.04)
                ratio=final_ratio+(initial_ratio-final_ratio)*math.exp(-progress/[.20,.17,.23][index])
                ratio+=.008*math.exp(-((elapsed-.12)/.06)**2)
                ratio+=.00022*math.sin(93*t)+.00012*math.sin(157*t)
                ratio+=.001*[1,.8,1.2][index]*math.exp(-((t-3.25)/.045)**2)
                reflected=incident*min(.95,max(0,ratio))
            else:
                incident=reflected=0.
            rows.append({"run_id":run_id,"sample_id":"1","scenario_id":"paper_shaped_20kw_synthetic",
                         "model_id":model_id,"time_s":t,"power_w":incident-reflected,"target_w":20000,
                         "source":"demo","incident_power_w":incident,"reflected_power_w":reflected})
    analysis=analyze_waveforms({"rows":rows,"source":"demo","options":{"evaluation_start_s":1.,"evaluation_end_s":5.95}})
    return {"mode":"demo","run_id":run_id,"magnitude_unit":"db","records":records,
            "predictions":predictions,"estimated_recommendations":[min((r for r in predictions if r["sample_id"]==record["sample_id"]),key=lambda r:r["estimated_reflected_power_ratio"]) for record in records],
            "metadata":[],"evaluation_model":"synthetic_preset_not_network","waveform_analysis":analysis,
            "waveform_reference":{"title":"ICRF阻抗匹配反馈控制系统的设计与实现","figures":["3.41","3.42","4.8"],
                                  "paper_shape":"入射平台、反射突升后约0.4～0.75秒衰减、低反射稳态与关断",
                                  "not_digitized":True,"not_model_validation":True,
                                  "final_amplitude_basis":"截图线性匹配后反射幅值范围，仅用于设定示例量级"},
            "notice":"合成示例：参考论文图3.41/3.42的脉冲形态与截图幅值量级；不是论文原始数据，也不是三模型预测或实测。功率判定仅使用1.00～5.95秒有功率窗口。"}
