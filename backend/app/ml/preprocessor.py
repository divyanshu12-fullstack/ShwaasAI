"""
Audio preprocessing pipeline for ShwaasAI respiratory screening.

Handles:
- Ingestion of WAV/PCM audio from bytes, base64 strings, or file paths
- Conversion of multi-channel audio to mono
- High-precision resampling to 16,000 Hz
- Amplitude and RMS normalization
- Silence gating & cough energy detection
- Slicing into 2.0-second sliding windows (standard format for Google HeAR)
"""

import base64
import io
import math
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

TARGET_SAMPLE_RATE = 16000
WINDOW_DURATION_SEC = 2.0
WINDOW_SAMPLES = int(TARGET_SAMPLE_RATE * WINDOW_DURATION_SEC)  # 32,000 samples
DEFAULT_HOP_SEC = 1.0  # 50% overlap for continuous screening
MIN_ENERGY_THRESHOLD = 0.005  # RMS threshold below which sound is considered silence


class AudioPreprocessor:
    """
    Standardized audio preprocessor for respiratory health acoustic models.
    """

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        window_duration: float = WINDOW_DURATION_SEC,
        hop_duration: float = DEFAULT_HOP_SEC,
        silence_threshold: float = MIN_ENERGY_THRESHOLD,
    ):
        self.target_sr = target_sr
        self.window_duration = window_duration
        self.window_samples = int(target_sr * window_duration)
        self.hop_samples = int(target_sr * hop_duration)
        self.silence_threshold = silence_threshold

    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Loads audio from raw bytes (WAV format or PCM).
        Falls back to soundfile if installed, otherwise pure SciPy/Wave.
        """
        try:
            # First attempt: standard WAV parser via scipy
            bio = io.BytesIO(audio_bytes)
            sr, data = wavfile.read(bio)
            return data, sr
        except Exception:
            pass

        # Try soundfile if available
        try:
            import soundfile as sf
            bio = io.BytesIO(audio_bytes)
            data, sr = sf.read(bio)
            return data, sr
        except Exception as e:
            raise ValueError(f"Unable to decode audio bytes. Ensure valid WAV format. Error: {e}")

    def load_audio_from_base64(self, b64_string: str) -> Tuple[np.ndarray, int]:
        """
        Decodes base64 string (including data URI prefix if present) and loads audio.
        """
        if "," in b64_string:
            # Handle data:audio/wav;base64,DATA
            b64_string = b64_string.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_string)
        return self.load_audio_from_bytes(raw_bytes)

    def load_audio_from_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads audio from a file path.
        """
        try:
            sr, data = wavfile.read(file_path)
            return data, sr
        except Exception:
            try:
                import soundfile as sf
                data, sr = sf.read(file_path)
                return data, sr
            except Exception as e:
                raise ValueError(f"Failed to load audio from {file_path}: {e}")

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Converts stereo or multi-channel audio to mono.
        """
        if audio.ndim == 1:
            return audio.astype(np.float32)
        elif audio.ndim == 2:
            # Average channels along axis 1 (or 0 if transposed)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 2:
                # Shape (channels, samples)
                return np.mean(audio, axis=0).astype(np.float32)
            else:
                # Shape (samples, channels)
                return np.mean(audio, axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Converts integer PCM to float [-1.0, 1.0] and applies peak/RMS normalization.
        """
        audio = audio.astype(np.float32)

        # Scale based on integer dtype if applicable
        if np.issubdtype(audio.dtype, np.integer):
            max_val = np.iinfo(audio.dtype).max
            audio = audio / float(max_val)
        else:
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

        # Center remove DC offset
        audio = audio - np.mean(audio)

        # RMS volume scaling to prevent quiet recordings from underperforming
        rms = np.sqrt(np.mean(audio**2) + 1e-9)
        target_rms = 0.1
        if rms > 1e-4:
            scale = min(target_rms / rms, 5.0)  # limit max amplification to 5x
            audio = audio * scale

        # Final safety clip
        audio = np.clip(audio, -0.99, 0.99)
        return audio

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        High-fidelity resampling to self.target_sr (16,000 Hz) using SciPy polyphase.
        """
        if orig_sr == self.target_sr:
            return audio

        gcd = math.gcd(orig_sr, self.target_sr)
        up = self.target_sr // gcd
        down = orig_sr // gcd

        resampled = signal.resample_poly(audio, up, down)
        return resampled.astype(np.float32)

    def compute_rms_energy(self, audio: np.ndarray) -> float:
        """
        Calculates Root-Mean-Square (RMS) energy.
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio**2) + 1e-10))

    def is_sound_active(self, audio: np.ndarray) -> bool:
        """
        Determines if the audio segment has sufficient energy (not pure silence).
        """
        return self.compute_rms_energy(audio) >= self.silence_threshold

    def segment_into_windows(
        self, audio: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Slices audio into 2.0-second sliding windows with hop.
        If audio is shorter than 2.0s, pads symmetrically to 2.0s.
        
        Returns:
            List of dicts:
            [
                {
                    "window_index": int,
                    "start_time_sec": float,
                    "end_time_sec": float,
                    "samples": np.ndarray (shape 32000,),
                    "rms_energy": float,
                    "is_active": bool
                }, ...
            ]
        """
        total_samples = len(audio)
        windows = []

        if total_samples <= self.window_samples:
            # Pad to 32,000 samples
            pad_needed = self.window_samples - total_samples
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            padded = np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0)
            
            rms = self.compute_rms_energy(padded)
            windows.append({
                "window_index": 0,
                "start_time_sec": 0.0,
                "end_time_sec": round(total_samples / self.target_sr, 3),
                "samples": padded,
                "rms_energy": rms,
                "is_active": rms >= self.silence_threshold,
            })
            return windows

        # Sliding window slicing
        window_idx = 0
        current_start = 0

        while current_start + self.window_samples <= total_samples:
            slice_samples = audio[current_start : current_start + self.window_samples]
            start_sec = round(current_start / self.target_sr, 3)
            end_sec = round((current_start + self.window_samples) / self.target_sr, 3)
            rms = self.compute_rms_energy(slice_samples)

            windows.append({
                "window_index": window_idx,
                "start_time_sec": start_sec,
                "end_time_sec": end_sec,
                "samples": slice_samples,
                "rms_energy": rms,
                "is_active": rms >= self.silence_threshold,
            })

            window_idx += 1
            current_start += self.hop_samples

        # Check if remaining tail > 0.8 seconds to capture last cough
        remaining = total_samples - current_start
        if remaining > int(0.8 * self.target_sr) and current_start < total_samples:
            tail = audio[current_start:]
            pad_needed = self.window_samples - len(tail)
            padded_tail = np.pad(tail, (0, pad_needed), mode="constant", constant_values=0)
            rms = self.compute_rms_energy(padded_tail)
            windows.append({
                "window_index": window_idx,
                "start_time_sec": round(current_start / self.target_sr, 3),
                "end_time_sec": round(total_samples / self.target_sr, 3),
                "samples": padded_tail,
                "rms_energy": rms,
                "is_active": rms >= self.silence_threshold,
            })

        return windows

    def process_raw_audio(
        self, audio_input: Union[bytes, str, np.ndarray], original_sr: int = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Unified end-to-end preprocessing entry point:
        1. Decodes/loads audio
        2. Converts to mono
        3. Resamples to 16 kHz
        4. Normalizes amplitude & volume
        5. Segments into 2-second windows
        
        Returns:
            (full_resampled_audio, list_of_2s_windows)
        """
        if isinstance(audio_input, bytes):
            audio, sr = self.load_audio_from_bytes(audio_input)
        elif isinstance(audio_input, str):
            if audio_input.startswith("data:") or len(audio_input) > 500 and "/" in audio_input:
                audio, sr = self.load_audio_from_base64(audio_input)
            else:
                audio, sr = self.load_audio_from_file(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input
            sr = original_sr if original_sr else self.target_sr
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        # Step 2: To mono
        mono_audio = self.to_mono(audio)

        # Step 3: Resample to 16 kHz
        resampled_audio = self.resample(mono_audio, sr)

        # Step 4: Normalize
        normalized_audio = self.normalize_amplitude(resampled_audio)

        # Step 5: Segment
        windows = self.segment_into_windows(normalized_audio)

        return normalized_audio, windows
