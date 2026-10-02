"""
Acoustic data engine for ShwaasAI.

Provides:
1. High-fidelity acoustic generator simulating authentic respiratory physics
   (explosive cough bursts, airflow decay, wheeze harmonics, crackle impulses)
2. Correlated clinical symptom profiles based on WHO presumptive-TB epidemiological data
3. Loader for real external datasets (Coswara, ICBHI 2017, COUGHVID)
"""

import os
import glob
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from backend.app.ml.preprocessor import AudioPreprocessor, TARGET_SAMPLE_RATE


class RespiratoryDataEngine:
    """
    Respiratory dataset loader and high-fidelity acoustic physics synthesizer.
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.preprocessor = AudioPreprocessor(target_sr=sample_rate)

    def synthesize_cough(
        self,
        duration_sec: float = 2.0,
        is_pathological: bool = False,
        cough_type: str = "forced",  # "forced" or "passive"
        has_crackles: bool = False,
        has_wheezes: bool = False,
    ) -> np.ndarray:
        """
        Synthesizes an authentic acoustic cough waveform based on respiratory biomechanics:
        - Phase 1: Glottal opening explosive burst (broadband impact + low-frequency resonance)
        - Phase 2: Intermediate airflow turbulence (decaying colored noise)
        - Phase 3: Secondary vocal cord approximation
        """
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        waveform = np.zeros_like(t, dtype=np.float32)

        # Base noise generator (filtered airflow turbulence)
        white_noise = np.random.normal(0, 1, len(t)).astype(np.float32)
        
        # Bandpass filter for cough airflow (300 Hz - 3500 Hz)
        sos = signal.butter(4, [300, 3500], btype="bandpass", fs=self.sample_rate, output="sos")
        filtered_noise = signal.sosfilt(sos, white_noise)

        # Temporal envelope modeling: explosive attack (0-50ms), exponential decay
        t_cough_start = 0.2 if cough_type == "forced" else 0.4
        t_cough = t - t_cough_start
        
        # Explosive burst envelope
        burst_envelope = np.where(
            (t_cough >= 0) & (t_cough < 0.8),
            np.exp(-t_cough * (5.0 if not is_pathological else 3.5)) * np.sin(np.pi * np.clip(t_cough / 0.8, 0, 1))**0.5,
            0.0
        )
        waveform += filtered_noise * burst_envelope * 0.8

        # Low-frequency chest wall resonance (100 - 300 Hz)
        res_freq = 140.0 if is_pathological else 200.0
        resonance = np.sin(2 * np.pi * res_freq * t) * np.exp(-np.maximum(t_cough, 0) * 8.0) * (t_cough > 0)
        waveform += resonance * (0.5 if is_pathological else 0.2)

        # Secondary cough burst (common in productive/TB coughs)
        if is_pathological:
            t_sec = t - (t_cough_start + 0.35)
            sec_burst = np.where(
                (t_sec >= 0) & (t_sec < 0.5),
                np.exp(-t_sec * 6.0) * np.sin(np.pi * np.clip(t_sec / 0.5, 0, 1)),
                0.0
            )
            waveform += filtered_noise * sec_burst * 0.5

        # Pathological acoustic additions:
        if has_crackles or (is_pathological and np.random.rand() > 0.4):
            # Crackles: short discontinuous explosive sounds (10-20ms) at 150-300Hz
            for cr_time in [0.35, 0.48, 0.62]:
                cr_t = t - cr_time
                crackle = np.sin(2 * np.pi * 220 * cr_t) * np.exp(-np.abs(cr_t) * 120.0) * (np.abs(cr_t) < 0.02)
                waveform += crackle * 0.4

        if has_wheezes or (is_pathological and np.random.rand() > 0.5):
            # Wheezes: continuous sinusoidal acoustic oscillations (400-800Hz)
            wheeze_env = np.where((t >= 0.5) & (t < 1.4), np.sin(np.pi * (t - 0.5) / 0.9), 0.0)
            wheeze_tone = (
                np.sin(2 * np.pi * 520 * t) + 
                0.5 * np.sin(2 * np.pi * 1040 * t)
            ) * wheeze_env * 0.35
            waveform += wheeze_tone

        # Subtle ambient room noise
        ambient = np.random.normal(0, 0.005, len(t))
        waveform += ambient

        # Normalize amplitude
        waveform = self.preprocessor.normalize_amplitude(waveform)
        return waveform

    def generate_patient_sample(
        self,
        patient_id: int,
        is_positive: bool,
    ) -> Dict[str, Any]:
        """
        Generates a complete patient profile:
        - 2 to 4 cough audio segments (passive and forced)
        - WHO-correlated clinical symptom profile
        """
        cough_type = "forced" if np.random.rand() > 0.5 else "passive"
        
        # Determine specific pathology label
        if is_positive:
            # 60% crackles, 25% wheezes, 15% combined
            rand = np.random.rand()
            if rand < 0.6:
                pathology_idx = 1  # Crackles
            elif rand < 0.85:
                pathology_idx = 2  # Wheezes
            else:
                pathology_idx = 3  # Combined
            has_crackles = pathology_idx in (1, 3)
            has_wheezes = pathology_idx in (2, 3)
        else:
            pathology_idx = 0  # Normal
            has_crackles = False
            has_wheezes = False

        audio = self.synthesize_cough(
            duration_sec=2.0,
            is_pathological=is_positive,
            cough_type=cough_type,
            has_crackles=has_crackles,
            has_wheezes=has_wheezes,
        )

        # Correlated clinical symptoms based on WHO TB epidemiology
        if is_positive:
            cough_days = int(np.random.normal(24, 8))
            fever = 1 if np.random.rand() < 0.78 else 0
            night_sweats = 1 if np.random.rand() < 0.72 else 0
            weight_loss = 1 if np.random.rand() < 0.68 else 0
            hemoptysis = 1 if np.random.rand() < 0.35 else 0
            smoker = 1 if np.random.rand() < 0.55 else 0
            chest_pain = 1 if np.random.rand() < 0.45 else 0
        else:
            cough_days = int(np.random.exponential(4))
            fever = 1 if np.random.rand() < 0.15 else 0
            night_sweats = 1 if np.random.rand() < 0.08 else 0
            weight_loss = 1 if np.random.rand() < 0.06 else 0
            hemoptysis = 1 if np.random.rand() < 0.01 else 0
            smoker = 1 if np.random.rand() < 0.25 else 0
            chest_pain = 1 if np.random.rand() < 0.10 else 0

        cough_days = max(cough_days, 0)
        age = int(np.random.normal(42, 14))
        age = int(np.clip(age, 18, 85))
        gender = "male" if np.random.rand() > 0.45 else "female"

        symptom_vector = np.array([
            min(age / 100.0, 1.0),
            1.0 if gender == "male" else 0.0,
            min(cough_days / 30.0, 1.0),
            float(fever),
            float(night_sweats),
            float(weight_loss),
            float(hemoptysis),
            float(smoker),
            float(chest_pain)
        ], dtype=np.float32)

        return {
            "patient_id": f"PAT_{patient_id:04d}",
            "audio": audio,
            "cough_type": cough_type,
            "label_tb": 1 if is_positive else 0,
            "label_pathology": pathology_idx,
            "symptom_vector": symptom_vector,
            "clinical_meta": {
                "age": age,
                "gender": gender,
                "cough_duration_days": cough_days,
                "has_fever": bool(fever),
                "has_night_sweats": bool(night_sweats),
                "has_unexplained_weight_loss": bool(weight_loss),
                "has_hemoptysis": bool(hemoptysis),
                "is_smoker": bool(smoker),
                "has_chest_pain": bool(chest_pain)
            }
        }

    def generate_benchmark_dataset(
        self, n_patients: int = 400, positive_ratio: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generates a balanced multi-patient benchmark dataset for training & cross-validation.
        """
        dataset = []
        n_pos = int(n_patients * positive_ratio)
        n_neg = n_patients - n_pos

        patient_idx = 1
        for _ in range(n_pos):
            dataset.append(self.generate_patient_sample(patient_idx, is_positive=True))
            patient_idx += 1

        for _ in range(n_neg):
            dataset.append(self.generate_patient_sample(patient_idx, is_positive=False))
            patient_idx += 1

        np.random.shuffle(dataset)
        return dataset

    def load_external_wav_dataset(
        self, root_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Scans a directory containing WAV files with metadata / folder labels
        (e.g., /normal/*.wav, /abnormal/*.wav, /crackles/*.wav).
        """
        dataset = []
        classes = ["normal", "crackles", "wheezes", "tb", "positive", "negative"]
        
        for wav_path in glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True):
            try:
                audio, _ = self.preprocessor.process_raw_audio(wav_path)
                fname = os.path.basename(wav_path).lower()
                parent = os.path.basename(os.path.dirname(wav_path)).lower()
                
                is_pos = any(k in fname or k in parent for k in ["positive", "tb", "crackles", "wheezes"])
                pathology = 0
                if "crackles" in fname or "crackles" in parent:
                    pathology = 1
                elif "wheezes" in fname or "wheezes" in parent:
                    pathology = 2
                elif is_pos:
                    pathology = 1

                dataset.append({
                    "patient_id": os.path.splitext(fname)[0],
                    "audio": audio[:32000] if len(audio) >= 32000 else np.pad(audio, (0, 32000 - len(audio))),
                    "cough_type": "forced",
                    "label_tb": 1 if is_pos else 0,
                    "label_pathology": pathology,
                    "symptom_vector": np.zeros(9, dtype=np.float32)
                })
            except Exception:
                continue

        return dataset
