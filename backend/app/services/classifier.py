"""
Downstream classification service for ShwaasAI.

Implements:
1. HeAR-TB Domain-Aware Dual-Head TB Classifier (Passive & Forced cough heads)
2. Respiratory Sound Pathology Classifier (Normal, Crackles, Wheezes, Combined)
3. Multimodal Clinical Fusion (Acoustics + Clinical Symptoms)
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import joblib


class ModelInferenceError(RuntimeError):
    """A required model is unavailable or could not produce a valid prediction."""

logger = logging.getLogger(__name__)

PATHOLOGY_CLASSES = [
    "Normal Respiratory Sound",
    "Crackles Detected",
    "Wheezes Detected",
    "Abnormal Acoustic Pattern (Crackles & Wheezes)"
]


class RespiratoryClassifierService:
    """
    Downstream classification service executing TB screening heads,
    pathology sound classification, and multimodal symptom fusion.
    """

    def __init__(self, weights_dir: Optional[str] = None):
        if weights_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_dir = os.path.join(base_dir, "ml", "weights")
        self.weights_dir = weights_dir

        self.model_passive = None
        self.scaler_passive = None
        self.model_forced = None
        self.scaler_forced = None
        self.model_pathology = None
        self.scaler_pathology = None
        self.model_multimodal = None

        self._load_or_initialize_weights()

    def _load_or_initialize_weights(self):
        """
        Attempts to load pretrained weights from disk.
        Loads the packaged classifier heads. Missing heads fail closed at inference.
        """
        # 1. Dual-head TB model
        tb_weights_path = os.path.join(self.weights_dir, "shwaas_tb_dual_head.joblib")
        legacy_weights_path = os.path.join(self.weights_dir, "hear_tb_prize_domain_aware.joblib")

        path_to_try = tb_weights_path if os.path.exists(tb_weights_path) else legacy_weights_path

        if os.path.exists(path_to_try):
            try:
                pkg = joblib.load(path_to_try)
                self.model_passive = pkg.get("model_p")
                self.scaler_passive = pkg.get("scaler_p")
                self.model_forced = pkg.get("model_f")
                self.scaler_forced = pkg.get("scaler_f")
                logger.info(f"Loaded TB Dual-Head model from {path_to_try}")
            except Exception as e:
                logger.warning(f"Failed loading TB model from {path_to_try}: {e}")

        # 2. Pathology model
        pathology_path = os.path.join(self.weights_dir, "shwaas_pathology_head.joblib")
        if os.path.exists(pathology_path):
            try:
                pkg = joblib.load(pathology_path)
                self.model_pathology = pkg.get("model")
                self.scaler_pathology = pkg.get("scaler")
                logger.info(f"Loaded Pathology model from {pathology_path}")
            except Exception as e:
                logger.warning(f"Failed loading Pathology model from {pathology_path}: {e}")

        # 3. Multimodal fusion model
        multimodal_path = os.path.join(self.weights_dir, "shwaas_multimodal.joblib")
        if os.path.exists(multimodal_path):
            try:
                pkg = joblib.load(multimodal_path)
                self.model_multimodal = pkg.get("model")
                logger.info(f"Loaded Multimodal model from {multimodal_path}")
            except Exception as e:
                logger.warning(f"Failed loading Multimodal model: {e}")

    def predict_tb_risk(self, embedding_512: np.ndarray, cough_type: str = "both") -> float:
        """
        Returns a research model score from the passive or forced cough head.
        """
        x = embedding_512.reshape(1, -1)

        if cough_type == "passive":
            heads = ((self.model_passive, self.scaler_passive),)
        elif cough_type == "forced":
            heads = ((self.model_forced, self.scaler_forced),)
        else:
            heads = ((self.model_passive, self.scaler_passive), (self.model_forced, self.scaler_forced))
        if any(model is None or scaler is None for model, scaler in heads):
            raise ModelInferenceError("Required TB classifier head is unavailable")
        try:
            scores = [float(model.predict_proba(scaler.transform(x))[0, 1]) for model, scaler in heads]
            if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores):
                raise ValueError("Invalid TB classifier output")
        except Exception as exc:
            raise ModelInferenceError("TB classifier inference failed") from exc
        return round(float(np.mean(scores)), 4)

    def predict_pathology(self, embedding_512: np.ndarray) -> Tuple[str, float]:
        """
        Predicts respiratory sound pathology (Normal, Crackles, Wheezes, Combined).
        """
        x = embedding_512.reshape(1, -1)

        if self.model_pathology is None or self.scaler_pathology is None:
            raise ModelInferenceError("Pathology classifier is unavailable")
        try:
            probs = self.model_pathology.predict_proba(self.scaler_pathology.transform(x))[0]
            if len(probs) != len(PATHOLOGY_CLASSES) or not np.all(np.isfinite(probs)):
                raise ValueError("Invalid pathology classifier output")
            pred_idx = int(np.argmax(probs))
            return PATHOLOGY_CLASSES[pred_idx], float(probs[pred_idx])
        except Exception as exc:
            raise ModelInferenceError("Pathology classifier inference failed") from exc

    def encode_clinical_symptoms(self, symptoms: Any) -> np.ndarray:
        """
        Encodes clinical metadata into standardized numerical vector.
        [age/100, is_male, cough_days/30, fever, night_sweats, weight_loss, hemoptysis, smoker, chest_pain]
        """
        if symptoms is None:
            return np.zeros(9, dtype=np.float32)

        age = getattr(symptoms, "age", 35) or 35
        gender = getattr(symptoms, "gender", "male") or "male"
        cough_days = getattr(symptoms, "cough_duration_days", 0) or 0
        fever = 1.0 if getattr(symptoms, "has_fever", False) else 0.0
        sweats = 1.0 if getattr(symptoms, "has_night_sweats", False) else 0.0
        weight_loss = 1.0 if getattr(symptoms, "has_unexplained_weight_loss", False) else 0.0
        hemoptysis = 1.0 if getattr(symptoms, "has_hemoptysis", False) else 0.0
        smoker = 1.0 if getattr(symptoms, "is_smoker", False) else 0.0
        chest_pain = 1.0 if getattr(symptoms, "has_chest_pain", False) else 0.0

        vec = np.array([
            min(age / 100.0, 1.0),
            1.0 if gender.lower() == "male" else 0.0,
            min(cough_days / 30.0, 1.0),
            fever,
            sweats,
            weight_loss,
            hemoptysis,
            smoker,
            chest_pain
        ], dtype=np.float32)
        return vec

    def predict_multimodal_fusion(
        self, acoustic_score: float, symptoms: Any
    ) -> Tuple[float, str]:
        """
        Combines the acoustic model score with clinical symptoms. This blend has
        not been independently calibrated or clinically validated.
        """
        symptom_vec = self.encode_clinical_symptoms(symptoms)

        # Compute clinical symptom severity
        cough_ratio = symptom_vec[2]  # cough_days/30
        fever = symptom_vec[3]
        sweats = symptom_vec[4]
        weight_loss = symptom_vec[5]
        hemoptysis = symptom_vec[6]
        smoker = symptom_vec[7]

        clinical_risk = float(np.clip(
            0.25 * min(cough_ratio * 1.5, 1.0) +
            0.20 * fever +
            0.20 * sweats +
            0.20 * weight_loss +
            0.35 * hemoptysis +
            0.05 * smoker,
            0.0,
            1.0
        ))

        if self.model_multimodal is None:
            raise ModelInferenceError("Multimodal classifier is unavailable")
        try:
            fusion_input = np.concatenate([[acoustic_score], symptom_vec]).reshape(1, -1)
            ml_prob = float(self.model_multimodal.predict_proba(fusion_input)[0, 1])
            if not np.isfinite(ml_prob) or not 0 <= ml_prob <= 1:
                raise ValueError("Invalid multimodal classifier output")
            prob = float(np.clip(0.6 * ml_prob + 0.4 * max(acoustic_score, clinical_risk), 0.01, 0.99))
            category = "High Risk" if prob >= 0.60 else ("Moderate Risk" if prob >= 0.35 else "Low Risk")
            return round(prob, 4), category
        except Exception as exc:
            raise ModelInferenceError("Multimodal classifier inference failed") from exc
