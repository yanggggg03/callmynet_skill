---
name: rf-matching-skill
description: Compare three trained RF impedance-matching networks, display two-stub electrical-length recommendations and external power waveforms in a local panel, and export Excel. Use for this repository's residual MLP, Fourier ResMLP, and conditional Flow models.
---

# RF Matching Skill

Use the Python runtime and browser panel shipped with this folder. The host model can be DeepSeek, Codex, or another tool-capable assistant; the neural networks run in PyTorch, independently of the host language model.

## Start and inspect

1. Resolve this skill folder from the location of `SKILL.md`. Do not assume the working directory.
2. Read `config/models.json`. The three `checkpoint` paths must refer to the user's trained `measured-double-stub-v1` checkpoints. Each checkpoint already includes architecture configuration, inverse-network weights, response-surrogate weights, input standardization, length bounds and units. Keep these together.
3. Start `python -B scripts/server.py` from this folder, or use `start_panel.ps1` with the user's PyTorch Python environment. Reuse an existing matching service on port 8765 when `/api/health` identifies it as `callmynet-skill`.
4. Open `http://127.0.0.1:8765/`. The first panel contains results, examples and Excel downloads; the second contains power waveforms and stability comparisons.

The distributed configuration initially points to the three original `D:/Projects/.../runs/measured/best.pt` paths. Edit a local config or pass `--config <path>` on other machines. Do not substitute `runs/smoke`, `runs/measured_smoke`, old nine-input checkpoints, or generated samples for trained weights. Missing weights mean the runtime is not yet ready for real inference. The explicit example button works without weights.

## Inference and comparison

- Accept one or more rows containing linear `mag` and phase `deg`, or explicitly selected S11 dB magnitude. Input Excel/CSV needs only those columns; aliases `Mag0,Deg0` are accepted. L1/L2 are outputs, not required prediction inputs.
- Run all three networks on the same inputs. Preserve each recommendation separately; do not average multimodal electrical lengths.
- All candidate sets, including Flow samples, are rescored by the single configured `evaluation_model` response surrogate. Separate self-evaluation scores are not comparable. Keep the training-data fingerprint, electrical-length units and bounds consistent across models.
- Without power waveforms, report an **estimated matching recommendation, awaiting waveform verification**. Estimated reflected-power ratio is |Gamma|²; it is not a time waveform or a measured stability result.
- Power comparison requires independently acquired measured or simulated traces for the three schemes under the same initial conditions, target power, time grid and observation length. Read [references/interface.md](references/interface.md) for the exact import contract.
- Bind traces to the current `run_id`, `sample_id`, and `scenario_id`. One waveform import compares one sample and one scenario. Do not attach stale traces or synthetic examples to a real inference run.
- Thresholds evaluate the last 40% of the common observation window by default: CV ≤2%, peak-to-peak ripple/target ≤5%, mean target error ≤2%, linear drift/target ≤1%; full-trace overshoot ≤20%. These defaults are configurable engineering criteria, not device specifications. At least 20 steady-window samples and 0.1 seconds total duration are required by default.
- A stable candidate must satisfy every limit. Rank passing candidates by the mean of metric/limit ratios. If none pass, report **no stable scheme**; the lowest score is diagnostic only. If scores tie, do not claim a meaningful measured advantage. Finite sampled traces do not prove mathematical or global stability.

## Deliver results

Use the panel's Excel download or `POST /api/export` to save all three results, estimated recommendations, waveform metrics, waveform samples, verified selected actions and provenance. Keep the `estimated` and measured/simulation/demo distinctions in the exported results and explanation.

No hardware actuation is implemented. The runtime recommends electrical lengths and evaluates supplied traces. For GitHub distribution and local service setup, read the root README and [references/interface.md](references/interface.md). A repository URL distributes the skill; it does not execute PyTorch by itself.
