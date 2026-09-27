# AI-Driven Vision-Based IV Fluid Monitoring System — Master Document

> Give this single file to any coding agent (Cursor, Antigravity, Codex) at the start of every session. Read it fully before writing code. Section 6 is hard constraints — never violate them. Section 10 is the living status — update it before ending any session so the next tool/session doesn't guess or hallucinate.

## 1. One-line pitch
Single-camera AI system that monitors IV fluid level, drip rate, and patient distress together, fusing them into one risk signal — replacing load-cell/microcontroller hardware with vision + ML.

## 2. Problem
Manual IV monitoring is error-prone and understaffed. Existing published/patented solutions handle fluid-level detection OR drip-rate counting OR patient monitoring in isolation — never fused — and don't generalize across bag types/lighting.

## 3. Scope
**IS:** single camera, CNN fluid-level model, YOLO drip counting, patient-distress model, fusion layer, predictive ETA, web/mobile dashboard.
**IS NOT:** load cell, HX711, ESP8266/ESP32-CAM, physical buzzer/LCD, Blynk/ThingSpeak cloud.

## 4. Modules (build in this order)
1. **Fluid-level estimation** — CNN on video frames of the IV bag → level class/%
2. **Drip-rate counting** — YOLO + tracker on drip chamber → drops/min
3. **Patient distress monitoring** — pretrained facial/pain model, minimal fine-tuning
4. **Fusion & risk scoring** — combines 1–3 into one alert tier (not independent alerts)
5. **Predictive ETA** — trained model estimating time-to-empty from noisy vision data (must be learned, not volume÷rate arithmetic)
6. **Dashboard/alerting** — web UI + notifications, no physical hardware

## 5. System diagram

```
                ┌─────────────────────┐
                │   Camera(s) Input    │
                │ (IV bag + patient)   │
                └──────────┬───────────┘
                           │ raw video frames
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌────────────┐ ┌──────────────┐
     │ Fluid-Level│ │ Drip-Rate  │ │   Patient    │
     │  CNN (M1)  │ │ YOLO+Track │ │  Distress    │
     │            │ │    (M2)    │ │  Model (M3)  │
     └──────┬─────┘ └──────┬─────┘ └──────┬───────┘
            │ level%        │ drops/min     │ distress score
            └───────┬───────┴───────┬───────┘
                     ▼               ▼
              ┌───────────────────────────┐
              │   Fusion & Risk Scoring    │
              │           (M4)             │
              └────────────┬──────────────┘
                             │
                     ┌───────┴────────┐
                     ▼                ▼
           ┌──────────────────┐ ┌───────────────┐
           │ Predictive ETA(M5)│ │ Dashboard(M6) │
           └──────────────────┘ └───────────────┘
```

## 6. Non-negotiable constraints
- NO load cell, HX711, ESP8266/ESP32-CAM, physical buzzer, or Blynk/ThingSpeak cloud.
- Patient monitoring and IV monitoring must feed one fusion layer, not separate dashboards.
- ETA prediction must be a trained model, not a plain formula.
- If a decision isn't resolved in Section 9, ask — don't assume and proceed.

## 7. Tech stack (locked)
Python, OpenCV, PyTorch (pick one framework and don't switch mid-build), YOLOv8/v5, FastAPI or Streamlit for the dashboard. Prototype target: laptop webcam, no embedded hardware.

**Locked this repo:** PyTorch (when training starts), YOLOv8, FastAPI dashboard.

## 8. Dataset plan
| Module | Data source | Status |
|---|---|---|
| M1 Fluid-level | **Downloaded starter:** Mendeley SESOVERA saline bottles (CC BY 4.0) — 25 imgs/class at 640px in `data/raw/fluid_level/public_mendeley/{0,50,80,100}`. Full set is ~4,217 original 3456px JPEGs (~13 GB) via `python -m tools.download_m1_public --per-class 0`. Roboflow Universe exports need a free API key. IEEE DataPort IV container set needs a login. | Starter public data on disk. Still supplement with self-recorded bag frames before final demo (this public set is bottles, not bags). |
| M2 Drip-rate | No anonymous public dump. IEEE DataPort “IV Drop Detection” (~3,458 imgs) requires login. Roboflow drip-chamber sets are tiny and need a Roboflow account. | Self-record on the rig remains required. |
| M3 Patient distress | None IV-specific — use pretrained model | Use existing pretrained facial-expression/pain-detection model (e.g., FER2013/AffectNet-trained), fine-tune lightly only if needed. Do not build a custom dataset for this. |

## 9. Open decisions (resolve here, don't leave silently assumed)
- Distress-detection approach: **Phase-I default — pretrained facial-expression model (FER-style).** Landmarks/optical-flow not used. Weights not loaded until M3 session; inference wrapper returns `unavailable` until then.
- Fusion logic: **Phase-I default — rule-based thresholds** in `models/fusion/risk.py`. A learned classifier is deferred until M1–M3 produce labeled fused outputs. Rules are a prototype, not a clinical protocol.
- ETA model architecture: **Phase-I default — sequence model (LSTM) trained later** on logged time series. Inference returns `model_not_trained` until weights exist. **Do not substitute volume÷rate arithmetic.**

## 10. Current status (update every session)
- [x] Datasets downloaded/combined (M1) — starter 100 images (25×4 classes) from Mendeley SESOVERA; not yet merged with Roboflow/IEEE
- [ ] Drip-chamber footage recorded (M2)
- [ ] Pretrained distress model selected (M3)
- [ ] M1 fluid-level CNN trained
- [ ] M2 drip-rate module built
- [ ] M3 patient monitoring module built
- [x] M4 fusion logic built (rule-based prototype; consumes live module outputs, including `None` while models are untrained)
- [ ] M5 ETA model built (interface + time-series logger only; no trained weights)
- [x] M6 dashboard built (FastAPI; live frame + fused risk + dataset recorder)
- [x] Capture loop, pipeline, shared state, dataset collection paths
- Last updated by: Cursor Grok 4.6, 2026-09-27
- Known issues / deviations from plan: Training (M1–M3, M5) deferred while the team collects data. Fusion currently runs on incomplete vision outputs (many fields `None`) — re-validate M4 against real M1–M3 outputs before treating it as done. M6 was built in parallel with collection tooling rather than strictly last; it only consumes pipeline state. M1 public starter is saline **bottles** (not IV bags). Roboflow/IEEE not downloaded (account walls).

## 11. Repo structure
```
/data/            raw + processed video/frames, per module
/models/
  fluid_level/    training script, saved weights, inference wrapper
  drip_rate/      YOLO config, weights, tracker wrapper
  patient/        distress model, weights, inference wrapper
  fusion/         risk-scoring logic
  eta/            ETA model, training + inference
/app/
  capture.py      camera capture loop
  pipeline.py     orchestrates M1–M5 per cycle
  dashboard/      FastAPI app (M6)
/tests/           per-module unit tests
PROJECT_MASTER.md
```

## 12. Build workflow (order matters)
1. Build M1 and M2 independently, validate each on held-out video before touching fusion.
2. Build M3 independently, same discipline.
3. Build M4 fusion logic against real M1–M3 outputs, not mocked ones.
4. Build M5 once M1+M2 produce a stable time series to train against.
5. Build M6 last — it only consumes outputs.
6. Update Section 10 after each module before moving to the next.

## 13. Runtime workflow
`capture.py` → `pipeline.py` (runs M1–M5 per cycle) → shared state → `dashboard/` polls/subscribes → renders + alerts. Keep every module independently callable so any one can be swapped or retrained without touching the others.

## 14. Team & context
Batch D3, team of 4, guide Mrs. N Divya, GNITS CSE (Data Science), Phase-I GNR-22 (2026–2027).

## 15. How to run (until models exist)
```
pip install -r requirements.txt
python -m app.dashboard.server
```
Open http://127.0.0.1:8000 — Start pipeline (webcam `0` or a video file path).

Labeled collection (same folders as the dashboard recorder):
```
python -m app.collect --module drip_rate --label 20dpm --seconds 60
python -m app.collect --module fluid_level --label 50 --frames 40 --interval 1
```
Drip clips go to `data/raw/drip_rate/<label>/`. Bag stills go to `data/raw/fluid_level/<label>/`. Time series for later ETA training logs to `data/timeseries/vision_log.csv`.

