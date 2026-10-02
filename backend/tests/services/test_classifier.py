"""
Unit tests for ShwaasAI core ML modules.
"""

import numpy as np
import pytest

from backend.app.ml.preprocessor import AudioPreprocessor
from backend.app.services.hear_service import HeARService
from backend.app.services.classifier import RespiratoryClassifierService
from backend.app.ml.aggregator import PatientAggregator
from backend.app.models.request import ClinicalSymptoms


def test_audio_preprocessor():
    prep = AudioPreprocessor()
    # 5 seconds of audio at 22050 Hz
    sr = 22050
    t = np.linspace(0, 5.0, int(5.0 * sr))
    sig = np.sin(2 * np.pi * 300 * t).astype(np.float32)

    full_sig, windows = prep.process_raw_audio(sig, original_sr=sr)
    assert len(full_sig) == int(5.0 * 16000)
    assert len(windows) >= 4
    for w in windows:
        assert len(w["samples"]) == 32000
        assert "rms_energy" in w
        assert "is_active" in w


def test_hear_service_embedding():
    hear = HeARService()
    # 2s audio slice at 16kHz
    dummy_2s = np.random.normal(0, 0.1, 32000).astype(np.float32)
    emb = hear.extract_embedding(dummy_2s)
    
    assert emb.shape == (512,)
    assert not np.isnan(emb).any()
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-2)


def test_classifier_service_predictions():
    classifier = RespiratoryClassifierService()
    dummy_emb = np.random.normal(0, 0.05, 512).astype(np.float32)
    norm = np.linalg.norm(dummy_emb)
    dummy_emb = dummy_emb / norm

    # TB risk
    tb_risk = classifier.predict_tb_risk(dummy_emb)
    assert 0.0 <= tb_risk <= 1.0

    # Pathology sound
    pathology, conf = classifier.predict_pathology(dummy_emb)
    assert isinstance(pathology, str)
    assert 0.0 <= conf <= 1.0

    # Multimodal
    symptoms = ClinicalSymptoms(
        age=45,
        cough_duration_days=21,
        has_fever=True,
        has_night_sweats=True,
        has_hemoptysis=True
    )
    multi_risk, multi_cat = classifier.predict_multimodal_fusion(tb_risk, symptoms)
    assert 0.0 <= multi_risk <= 1.0
    assert multi_cat in ["Low Risk", "Moderate Risk", "High Risk"]
    # Severe symptoms should increase risk
    assert multi_risk >= tb_risk


def test_patient_aggregator():
    aggregator = PatientAggregator()
    from backend.app.models.response import AudioWindowScore

    w1 = AudioWindowScore(
        window_index=0,
        start_time_sec=0.0,
        end_time_sec=2.0,
        tb_risk_score=0.85,
        pathology="Crackles Detected",
        pathology_confidence=0.88,
        is_cough_detected=True,
        rms_energy=0.15
    )
    w2 = AudioWindowScore(
        window_index=1,
        start_time_sec=1.0,
        end_time_sec=3.0,
        tb_risk_score=0.40,
        pathology="Normal Respiratory Sound",
        pathology_confidence=0.75,
        is_cough_detected=True,
        rms_energy=0.08
    )

    resp = aggregator.build_screening_response(
        patient_id="PAT_001",
        total_duration_sec=3.0,
        window_scores=[w1, w2],
        multimodal_risk=0.88,
        multimodal_category="High Risk"
    )

    assert resp.patient_id == "PAT_001"
    assert resp.acoustic_tb_risk_score > 0.50
    assert resp.total_windows_analyzed == 2
    assert resp.windows_with_cough == 2
    assert resp.most_suspicious_window.window_index == 0
    assert "Referral" in resp.triage_recommendation or "Recommended" in resp.triage_recommendation
