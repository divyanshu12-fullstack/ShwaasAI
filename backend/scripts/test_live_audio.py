"""
Live demonstration script for ShwaasAI respiratory screening.

Runs an end-to-end test on sample WAV audio files with or without clinical symptoms:
1. Ingests and normalizes audio
2. Slices into 2.0s windows
3. Extracts 512-D health acoustic representations
4. Runs TB Dual-Head (Passive + Forced coughs)
5. Detects respiratory sound pathologies (Normal, Crackles, Wheezes)
6. Performs multimodal clinical symptom fusion
7. Prints a publication-quality clinical triage report
"""

import os
import sys
import argparse
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.ml.preprocessor import AudioPreprocessor
from backend.app.services.hear_service import HeARService
from backend.app.services.classifier import RespiratoryClassifierService
from backend.app.ml.aggregator import PatientAggregator
from backend.app.models.request import ClinicalSymptoms


def run_live_screening(
    audio_path: str,
    patient_id: str = "PAT_DEMO_001",
    cough_type: str = "both",
    symptoms: Optional[ClinicalSymptoms] = None
):
    print("\n" + "="*80)
    print(f" SHWAAS-AI: LIVE SCREENING EXECUTION FOR {audio_path}")
    print("="*80)

    # 1. Initialize services
    preprocessor = AudioPreprocessor()
    hear_service = HeARService()
    classifier = RespiratoryClassifierService()
    aggregator = PatientAggregator()

    # 2. Audio Preprocessing
    print("[1/5] Ingesting and preprocessing audio...")
    full_audio, raw_windows = preprocessor.process_raw_audio(audio_path)
    total_duration = len(full_audio) / preprocessor.target_sr
    print(f"      Duration: {total_duration:.2f}s | Sample Rate: {preprocessor.target_sr} Hz | Windows: {len(raw_windows)}")

    # 3. Health Acoustic Embeddings
    print("[2/5] Extracting 512-dimensional health acoustic representations...")
    embedded_windows = hear_service.extract_embeddings_for_windows(raw_windows)

    # 4. Classify each window
    print("[3/5] Executing TB Dual-Head & Pathology Sound Classifiers...")
    from backend.app.models.response import AudioWindowScore
    window_scores = []
    for w in embedded_windows:
        emb = w["embedding"]
        tb_risk = classifier.predict_tb_risk(emb, cough_type=cough_type)
        pathology_name, path_conf = classifier.predict_pathology(emb)

        window_scores.append(
            AudioWindowScore(
                window_index=w["window_index"],
                start_time_sec=w["start_time_sec"],
                end_time_sec=w["end_time_sec"],
                tb_risk_score=tb_risk,
                pathology=pathology_name,
                pathology_confidence=path_conf,
                is_cough_detected=w["is_active"],
                rms_energy=round(w["rms_energy"], 4),
            )
        )

    # 5. Multimodal Clinical Fusion
    multimodal_risk = None
    multimodal_category = None
    if symptoms is not None:
        print("[4/5] Performing Multimodal Clinical Symptom Fusion...")
        acoustic_score, _, _, _, _ = aggregator.aggregate_windows(window_scores)
        multimodal_risk, multimodal_category = classifier.predict_multimodal_fusion(
            acoustic_score, symptoms
        )

    # 6. Patient-Level Aggregation
    print("[5/5] Aggregating results into clinical triage assessment...")
    report = aggregator.build_screening_response(
        patient_id=patient_id,
        total_duration_sec=total_duration,
        window_scores=window_scores,
        multimodal_risk=multimodal_risk,
        multimodal_category=multimodal_category
    )

    # Display Report
    print("\n" + "-"*80)
    print(" PATIENT SCREENING REPORT")
    print("-"*80)
    print(f" Patient ID:                   {report.patient_id}")
    print(f" Acoustic TB Risk Score:       {report.acoustic_tb_risk_score:.4f} ({report.acoustic_risk_category})")
    print(f" Dominant Pathology Sound:     {report.primary_pathology} (Confidence: {report.pathology_confidence*100:.1f}%)")
    if report.multimodal_risk_score is not None:
        print(f" Multimodal Risk Score:        {report.multimodal_risk_score:.4f} ({report.multimodal_risk_category})")
    print(f" Analyzed Windows:             {report.total_windows_analyzed} ({report.windows_with_cough} with active cough)")

    if report.most_suspicious_window:
        ms = report.most_suspicious_window
        print(f" Primary Suspicious Window:    Window #{ms.window_index} [{ms.start_time_sec:.1f}s - {ms.end_time_sec:.1f}s] (Risk: {ms.tb_risk_score:.4f})")

    print("\n CLINICAL TRIAGE RECOMMENDATION:")
    print(f" >> {report.triage_recommendation}")
    print("\n MANDATORY MEDICAL DISCLAIMER:")
    print(f" >> {report.disclaimer}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate live screening on sample audio.")
    parser.add_argument("--audio", type=str, default="data/samples/cough_tb_suspicious.wav", help="Path to WAV audio file.")
    parser.add_argument("--fever", action="store_true", help="Flag if patient has fever.")
    parser.add_argument("--night_sweats", action="store_true", help="Flag if patient has night sweats.")
    parser.add_argument("--cough_days", type=int, default=18, help="Days of persistent cough.")
    args = parser.parse_args()

    symptoms = ClinicalSymptoms(
        age=42,
        gender="male",
        cough_duration_days=args.cough_days,
        has_fever=bool(args.fever),
        has_night_sweats=bool(args.night_sweats),
        has_unexplained_weight_loss=args.cough_days > 14,
        has_hemoptysis=False
    )

    run_live_screening(args.audio, symptoms=symptoms)
