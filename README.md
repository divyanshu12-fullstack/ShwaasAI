# 🫁 ShwaasAI-ML: Acoustic Respiratory Disease Screening & Multimodal Health Intelligence Engine

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)

**ShwaasAI-ML** is a research respiratory sound screening prototype. The current backend accepts authenticated WAV recordings for cough sessions, extracts 512-value acoustic features, and runs packaged passive/forced cough, pathology, and optional symptom-fusion classifier heads. Its scores are not clinically validated TB probabilities. The bundled heads do not use Google HeAR embeddings; enabling HeAR requires compatible retrained heads.

---

## 📌 Executive Summary & Key Highlights

* **Acoustic features (512-D)**: Slices WAV audio into 2.0-second 16 kHz windows and extracts the Mel/MFCC feature layout used by the packaged heads.
* **Dual-Head TB Acoustic Screener**: Separate XGBoost classifiers trained on passive and forced cough dynamics to resolve acoustic variability across different cough types.
* **Respiratory Sound Pathology Head**: 4-class classification detecting *Normal Respiratory Sound, Crackles Detected, Wheezes Detected, and Combined Crackles & Wheezes*.
* **Optional symptom fusion**: Combines the acoustic score with submitted symptom answers. Stored session metadata is not yet connected to this input.
* **Authenticated FastAPI routes**: Supabase Auth, owner-scoped sessions, statistics, metadata, embeddings, WAV analysis, health, and OpenAPI documentation.
* **Research status**: Model provenance, external validation, score calibration, and real mobile recorder integration remain release requirements.

---

## 🔬 System Architecture & Pipeline Dataflow

```
                     ┌──────────────────────────────────────────────┐
                     │           Patient Audio Input                │
                     │              (WAV audio)                     │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │          Audio Preprocessor                  │
                     │  • Resample to 16 kHz Mono                   │
                     │  • Normalize amplitude and gate silence     │
                     │  • Window Slicing (2.0s with 50% overlap)    │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │     Mel/MFCC Acoustic Feature Vectors       │
                     │    Extract 512-D Health Representations      │
                     └──────────────┬───────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│     TB Dual-Head Screener     │               │   Pathology Sound Screener    │
│ • Passive Cough Head (XGBoost)│               │ • 4-Class Softmax Classifier  │
│ • Forced Cough Head (XGBoost) │               │   (Normal, Crackles, Wheezes) │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │      Multimodal Clinical Symptom Fusion      │
                     │  • Cough Duration (>14 days prior)           │
                     │  • Fever, Night Sweats, Weight Loss          │
                     │  • Hemoptysis, Smoking, Chest Pain           │
                     └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │     Research Score Aggregation              │
                     │  • Blended Score (Peak + Weighted Energy)    │
                     │  • Risk Tier: Low / Moderate / High Risk     │
                     │  • Follow-up text with a medical disclaimer  │
                     └──────────────────────────────────────────────┘
```

---

## 📊 Model evaluation status

The repository contains a cross-validation script that can train and evaluate on generated synthetic cough recordings. These results cannot establish sensitivity, specificity, AUROC, or WHO target-product-profile compliance on real patients. The checked-in artifacts lack documented real-patient training provenance and independent external validation. Clinical performance and score thresholds must be measured on representative held-out labeled recordings before any deployment for patient decisions.

---

## 📁 Repository Directory Structure

```
ShwaasAI-ML/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routers/
│   │   │       └── analyze.py          # Authenticated waveform screening routes
│   │   ├── ml/
│   │   │   ├── aggregator.py           # Research score aggregation
│   │   │   ├── data_engine.py          # Acoustic physics synthesis & benchmark loader
│   │   │   ├── feature_extractor.py    # 512-D health acoustic feature representation
│   │   │   ├── preprocessor.py         # 16kHz audio normalization & 2s windowing
│   │   │   └── weights/                # Standardized .joblib trained weights
│   │   │       ├── shwaas_tb_dual_head.joblib
│   │   │       ├── shwaas_pathology_head.joblib
│   │   │       └── shwaas_multimodal.joblib
│   │   ├── models/
│   │   │   ├── request.py              # Pydantic schemas for audio & clinical metadata
│   │   │   └── response.py             # Pydantic schemas for patient screening output
│   │   └── services/
│   │       ├── classifier.py           # Downstream dual-head & multimodal inference
│   │       └── hear_service.py         # Google HeAR embedding loader & extractor
│   ├── scripts/
│   │   ├── download_models.py          # Weights downloader & local verification
│   │   ├── download_datasets.py        # Dataset helper (Coswara, ICBHI, COUGHVID)
│   │   ├── test_live_audio.py          # End-to-end CLI screening demonstration
│   │   └── train_and_evaluate.py       # 5-fold CV benchmark & production weight trainer
│   ├── tests/
│   │   ├── api/
│   │   │   └── test_analyze.py         # FastAPI endpoint integration tests
│   │   └── services/
│   │       └── test_classifier.py      # Core unit tests for ML pipeline
│   ├── main.py                         # FastAPI backend server entrypoint
│   └── pyproject.toml                  # Backend project dependencies
├── data/
│   └── samples/                        # Authentic sample WAV clips for validation
│       ├── cough_healthy_forced.wav
│       ├── cough_healthy_passive.wav
│       ├── cough_tb_suspicious.wav
│       └── respiratory_wheeze.wav
├── .gitignore
├── LICENSE
├── pytest.ini
└── README.md
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Tejas-Ranjeet/ShwaasAI-ML.git
cd ShwaasAI-ML
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv sync --project backend --extra dev
```

---

## 🧪 Testing & Quality Assurance

Run the comprehensive unit and integration test suite:

```bash
uv run --project backend pytest backend/tests
```

This covers the auth, session, metadata, analysis API, and classifier tests.

---

## 🩺 Running Live Audio Screening (CLI Demo)

Test the engine on the repository's example WAV files directly from the command line. These files and their names do not establish clinical ground truth:

### Test Healthy Cough Sample:
```bash
python backend/scripts/test_live_audio.py --audio data/samples/cough_healthy_forced.wav --cough_days 0
```

### Test Suspicious Pathological Sample with Clinical Symptoms:
```bash
python backend/scripts/test_live_audio.py --audio data/samples/cough_tb_suspicious.wav --fever --night_sweats --cough_days 21
```

---

## 🌐 Running the FastAPI Backend Server

Start the REST API server locally:

```bash
uv run --project backend uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

* **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative Redoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Service health**: `GET /api/v1/health`; **current API capabilities**: `GET /api/v1/info`

### Example API Request (`POST /api/v1/analyze/audio`)

Create a pending cough session through `POST /api/v1/sessions` first. Send the Supabase access token in `Authorization: Bearer <access_token>` with the analysis request. The model receives server-generated 16 kHz waveform windows from WAV audio; the mobile app does not send spectrograms or embeddings.

```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "session_id": "<session_uuid>",
  "symptoms": {
    "age": 42,
    "gender": "male",
    "cough_duration_days": 21,
    "has_fever": true,
    "has_night_sweats": true,
    "has_unexplained_weight_loss": true,
    "has_hemoptysis": false,
    "is_smoker": true,
    "has_chest_pain": true
  }
}
```

### Example API Screening Response

```json
{
  "session_id": "<session_uuid>",
  "patient_id": null,
  "status": "success",
  "acoustic_tb_risk_score": 0.9614,
  "acoustic_risk_category": "High Risk",
  "primary_pathology": "Crackles Detected",
  "pathology_confidence": 1.0,
  "multimodal_risk_score": 0.9845,
  "multimodal_risk_category": "High Risk",
  "total_audio_duration_sec": 3.5,
  "total_windows_analyzed": 3,
  "windows_with_cough": 2,
  "most_suspicious_window": {
    "window_index": 0,
    "start_time_sec": 0.0,
    "end_time_sec": 2.0,
    "tb_risk_score": 0.9799,
    "pathology": "Crackles Detected",
    "pathology_confidence": 1.0,
    "is_cough_detected": true,
    "rms_energy": 0.0742
  },
  "triage_recommendation": "Priority Referral Recommended: High-probability acoustic or symptomatic signatures detected. Recommend microbiological confirmatory testing (sputum smear microscopy or GeneXpert MTB/RIF) and medical evaluation.",
  "disclaimer": "ShwaasAI is a research-oriented screening and risk-prioritization tool, not a definitive medical diagnostic device. A high risk score indicates need for clinical confirmation via microbiological tests. Always consult a qualified healthcare professional."
}
```

The example scores are illustrative. `POST /api/v1/analyze` also accepts multipart WAV audio with `session_id`, `file`, and optional JSON `symptoms` form fields. Both routes require ownership of a pending cough session and save the result and pooled embedding. The current classifier does not support a separate breathing TB risk head.
The backend deliberately uses the declared `librosa` acoustic feature path with its packaged heads. Google HeAR cannot be swapped in based only on equal embedding length; check `/api/v1/analyze/status` at runtime. The legacy `is_cough_detected` field indicates an RMS sound activity threshold, not a trained cough detector.

---

## ⚖️ Clinical & Ethical Disclaimer

> **Important Medical Notice**:  
> ShwaasAI-ML is developed as an AI-assisted screening, triage, and risk-prioritization software for academic research and clinical decision support. It is **not** a standalone diagnostic device. In accordance with WHO guidelines, presumptive-positive screening results must be confirmed through clinical microbiological assays (e.g., sputum smear microscopy, culture, or GeneXpert MTB/RIF).

---

## 📄 License
This project is open-source and released under the [Creative Commons CC0 1.0 Universal License](LICENSE).
