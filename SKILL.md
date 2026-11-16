---
name: rf-matching-skill
description: Train configurable local RF impedance-matching networks, manage best.pt checkpoints, predict reflection coefficients from Excel inputs, compare external power waveforms and export Excel in the CallMyNet workspace.
---

# RF Matching Skill

Use this repository's local browser workspace or HTTP runtime. Neural networks run in the user's Python/PyTorch environment independently of the host language model. Resolve paths from this SKILL.md directory.

## Local workspace

Start `python -B scripts/server.py`, or `start_panel.ps1 -Python <PyTorch Python>`, then open http://127.0.0.1:8765/. Reuse a matching service when its health identifies callmynet-skill and supports version2 training endpoints. GitHub only hosts the frontend; training and inference require a local service.

Let the user choose model count, types and names. Built-ins are residual MLP, Fourier ResMLP and conditional Flow; the same type can appear multiple times. Do not force all three. Local developers can register compatible networks; read [references/training.md](references/training.md) for extension and training API details.

## Training

The panel accepts a measured Excel/CSV with initial magnitude/phase, L1/L2 and matched magnitude/phase. Two-column prediction inputs are insufficient for training. Select initial and matched magnitude units independently, retain phases/electrical lengths in degrees, and support experimental group_id.

Prepare data once with fixed group splits, then train selected models against that same prepared archive. Show asynchronous status, logs and checkpoint paths. Each model gets a separate best.pt, history and summary in a new runs/training job directory. Do not merge weights or overwrite previous jobs. Activate only a complete successful job; cancellation/failure leave existing active models intact.

The UI writes a machine-local active configuration beside config/models.json. Keep data, runs, weights and *.local.json out of public releases and ZIPs. Data generated from a common trajectory is not independent measured validation.

## Prediction and waveform verification

Select one or more trained models plus one common response evaluator from that selection. Rescore all candidates with that evaluator; self-scores are not comparable. Require consistent training arrays/splits, length units and bounds.

Upload mag0/deg0 Excel/CSV (mag0/meg0 and mag/deg aliases accepted), confirm successful validation, filename, row count and preview, then compute. A failed upload must not reuse old data; changing inputs, model selection or evaluator invalidates prior results.

Lead results with linear reflection coefficient magnitude |Gamma| and phase in degrees. API/Excel retain real/imaginary components; reflected power ratio is auxiliary magnitude squared. Estimated coefficients come from a response surrogate, not new measurements.

These networks predict static samples, not dynamic power waveforms. Import independently measured or simulated traces for every selected model under the same initial condition, target, time grid and observation window. Bind them to current run_id/sample_id/scenario_id. Read [references/interface.md](references/interface.md) for format and thresholds. Do not synthesize waveforms and label them model responses. Without traces report estimated recommendations awaiting verification; if none pass every threshold report no stable scheme.

Export predictions, recommendations, provided waveforms and stability metrics. No hardware actuation is implemented. See README.md for setup and [references/validation.md](references/validation.md) for verification limits.
