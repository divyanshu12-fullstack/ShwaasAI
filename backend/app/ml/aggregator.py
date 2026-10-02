"""
Patient-level aggregation engine for ShwaasAI.

Aggregates window-level acoustic scores across an entire recording into a single,
calibrated patient-level screening outcome with clinical explainability.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from backend.app.models.response import AudioWindowScore, PatientScreeningResponse


class PatientAggregator:
    """
    Combines multi-window acoustic inferences into a patient-level assessment.
    """

    def __init__(
        self,
        low_threshold: float = 0.35,
        high_threshold: float = 0.65,
        energy_weight_factor: float = 0.5,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.energy_weight_factor = energy_weight_factor

    def aggregate_windows(
        self, window_scores: List[AudioWindowScore]
    ) -> Tuple[float, str, str, float, Optional[AudioWindowScore]]:
        """
        Computes:
        - patient_tb_risk_score
        - risk_category
        - primary_pathology
        - pathology_confidence
        - most_suspicious_window
        """
        if not window_scores:
            return 0.0, "Low Risk", "Normal Respiratory Sound", 1.0, None

        # Filter active sound windows if any exist, otherwise use all
        active_windows = [w for w in window_scores if w.is_cough_detected]
        target_windows = active_windows if active_windows else window_scores

        scores = [w.tb_risk_score for w in target_windows]
        energies = [max(w.rms_energy, 1e-4) for w in target_windows]

        # 1. Weighted Mean Pooling
        total_energy = sum(energies)
        weighted_mean = sum(s * e for s, e in zip(scores, energies)) / total_energy

        # 2. Top-1 Max Pooling
        max_score = max(scores)

        # 3. Blended Patient Score (50% max peak + 50% weighted energy mean)
        patient_score = round(0.5 * max_score + 0.5 * weighted_mean, 4)
        patient_score = float(np.clip(patient_score, 0.0, 1.0))

        # Risk tier
        if patient_score >= self.high_threshold:
            category = "High Risk"
        elif patient_score >= self.low_threshold:
            category = "Moderate Risk"
        else:
            category = "Low Risk"

        # Find most suspicious window
        most_suspicious = max(target_windows, key=lambda w: w.tb_risk_score)

        # Primary Pathology Voting
        pathologies = [w.pathology for w in target_windows]
        pathology_counts: Dict[str, float] = {}
        for w in target_windows:
            p = w.pathology
            pathology_counts[p] = pathology_counts.get(p, 0.0) + w.pathology_confidence

        primary_pathology = max(pathology_counts.keys(), key=lambda k: pathology_counts[k])
        total_conf = sum(pathology_counts.values())
        pathology_confidence = round(pathology_counts[primary_pathology] / total_conf, 3)

        return patient_score, category, primary_pathology, pathology_confidence, most_suspicious

    def get_triage_recommendation(self, tb_risk: float, multimodal_risk: Optional[float] = None) -> str:
        """
        Generates clinical action guidance based on WHO triage recommendations.
        """
        effective_risk = multimodal_risk if multimodal_risk is not None else tb_risk

        if effective_risk >= self.high_threshold:
            return (
                "Priority Referral Recommended: The screening detected high-probability acoustic "
                "or symptomatic signatures. Recommend microbiological confirmatory testing "
                "(sputum smear microscopy or GeneXpert MTB/RIF) and medical evaluation."
            )
        elif effective_risk >= self.low_threshold:
            return (
                "Clinical Follow-up Advised: Moderate acoustic anomaly detected. If symptoms "
                "(cough > 2 weeks, fever, night sweats) persist or worsen, consult a healthcare provider."
            )
        else:
            return (
                "The research model returned a lower score. This does not rule out TB or other "
                "illness. Consult a healthcare professional if symptoms develop or persist."
            )

    def build_screening_response(
        self,
        patient_id: Optional[str],
        total_duration_sec: float,
        window_scores: List[AudioWindowScore],
        multimodal_risk: Optional[float] = None,
        multimodal_category: Optional[str] = None,
    ) -> PatientScreeningResponse:
        """
        Constructs the final PatientScreeningResponse object.
        """
        (
            acoustic_score,
            acoustic_cat,
            pathology,
            pathology_conf,
            most_suspicious,
        ) = self.aggregate_windows(window_scores)

        active_count = sum(1 for w in window_scores if w.is_cough_detected)
        triage = self.get_triage_recommendation(acoustic_score, multimodal_risk)

        return PatientScreeningResponse(
            patient_id=patient_id,
            status="success",
            acoustic_tb_risk_score=acoustic_score,
            acoustic_risk_category=acoustic_cat,
            primary_pathology=pathology,
            pathology_confidence=pathology_conf,
            multimodal_risk_score=multimodal_risk,
            multimodal_risk_category=multimodal_category,
            total_audio_duration_sec=round(total_duration_sec, 2),
            total_windows_analyzed=len(window_scores),
            windows_with_cough=active_count,
            most_suspicious_window=most_suspicious,
            window_breakdown=window_scores,
            triage_recommendation=triage,
        )
