"""
Integration tests for the /api/v1/analyze endpoints using FastAPI TestClient.
"""

import io
import base64
import numpy as np
import scipy.io.wavfile as wavfile
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def create_dummy_wav_bytes(duration_sec: float = 3.0, sr: int = 16000) -> bytes:
    """Helper to generate in-memory WAV audio bytes."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr))
    # Simulated cough impulse
    sig = 0.5 * np.sin(2 * np.pi * 350 * t) * np.exp(-t * 2.0)
    sig_int16 = (sig * 32767).astype(np.int16)

    bio = io.BytesIO()
    wavfile.write(bio, sr, sig_int16)
    return bio.getvalue()


def test_api_status():
    response = client.get("/api/v1/analyze/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["target_sample_rate"] == 16000
    assert "models" in data


def test_analyze_base64_audio():
    wav_bytes = create_dummy_wav_bytes(duration_sec=3.0)
    b64_str = base64.b64encode(wav_bytes).decode("utf-8")

    payload = {
        "audio_base64": b64_str,
        "patient_id": "TEST_PATIENT_101",
        "cough_type": "both",
        "symptoms": {
            "age": 38,
            "gender": "female",
            "cough_duration_days": 18,
            "has_fever": True,
            "has_night_sweats": True,
            "has_unexplained_weight_loss": True
        }
    }

    response = client.post("/api/v1/analyze/audio", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["patient_id"] == "TEST_PATIENT_101"
    assert 0.0 <= data["acoustic_tb_risk_score"] <= 1.0
    assert data["acoustic_risk_category"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "disclaimer" in data
    assert len(data["window_breakdown"]) >= 2
    assert data["multimodal_risk_score"] is not None


def test_analyze_upload_audio():
    wav_bytes = create_dummy_wav_bytes(duration_sec=2.5)

    files = {"file": ("cough.wav", wav_bytes, "audio/wav")}
    data = {
        "patient_id": "TEST_UPLOAD_202",
        "cough_duration_days": 5,
        "has_fever": False
    }

    response = client.post("/api/v1/analyze/upload", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["patient_id"] == "TEST_UPLOAD_202"
    assert res_data["status"] == "success"
    assert res_data["total_windows_analyzed"] >= 1
