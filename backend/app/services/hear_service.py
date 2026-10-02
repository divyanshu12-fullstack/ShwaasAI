"""
HeAR (Health Acoustic Representations) embedding service for ShwaasAI.

Manages:
- Google HeAR foundation model inference (when model weights are loaded/available)
- 512-dimensional acoustic feature extractor used by the bundled heads
- Batch window embedding generation
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from backend.app.ml.feature_extractor import AcousticFeatureExtractor

from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

logger = logging.getLogger(__name__)


class HeARService:
    """
    Health Acoustic Representation Service.
    Maps 2.0-second 16-kHz audio segments into 512-dimensional health embeddings.
    """

    def __init__(self, model_dir: Optional[str] = None, enable_foundation_model: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback_extractor = AcousticFeatureExtractor()
        self.hear_model = None
        self.is_foundation_model_loaded = False
        
        # Model path check
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "ml", "weights")
        self.model_dir = model_dir

        # Shipped classifier heads were trained on AcousticFeatureExtractor
        # vectors. Equal dimensionality does not make HeAR vectors compatible.
        # A future HeAR-trained head must opt in to that extractor explicitly.
        if enable_foundation_model:
            self._initialize_model()
        else:
            logger.info("Using the acoustic extractor matched to the shipped classifier heads.")

    def _initialize_model(self):
        """
        Attempts to load Google HeAR PyTorch weights if present locally or via Hugging Face.
        """
        hear_pt_path = os.path.join(self.model_dir, "hear_pytorch.pt")
        if os.path.exists(hear_pt_path):
            try:
                logger.info(f"Loading local Google HeAR model from {hear_pt_path}...")
                self.hear_model = torch.jit.load(hear_pt_path, map_location=self.device)
                self.hear_model.eval()
                self.is_foundation_model_loaded = True
                logger.info("Successfully loaded Google HeAR foundation model.")
                return
            except Exception as e:
                logger.warning(f"Could not load local hear_pytorch.pt: {e}")

        # Check if already cached locally or if explicit full download requested
        token = os.getenv("HF_TOKEN")
        if os.getenv("DOWNLOAD_HF_HEAR_FULL") == "1" and token:
            try:
                from transformers import AutoModel
                logger.info("Attempting to download google/hear-pytorch from Hugging Face...")
                self.hear_model = AutoModel.from_pretrained("google/hear-pytorch", token=token).to(self.device)
                self.hear_model.eval()
                self.is_foundation_model_loaded = True
                logger.info("Loaded google/hear-pytorch from Hugging Face.")
                return
            except Exception as e:
                logger.info(f"Hugging Face full download note: {e}")
        else:
            # Fast check: load from HuggingFace local cache if already downloaded
            try:
                from transformers import AutoModel
                self.hear_model = AutoModel.from_pretrained("google/hear-pytorch", token=token, local_files_only=True).to(self.device)
                self.hear_model.eval()
                self.is_foundation_model_loaded = True
                logger.info("Loaded cached google/hear-pytorch from local cache.")
                return
            except Exception:
                pass

        logger.info("Using the 512-D acoustic extractor; HeAR is unavailable.")

    def extract_embedding(self, audio_2s: np.ndarray) -> np.ndarray:
        """
        Extracts a single 512-dimensional embedding vector from 2.0s audio (32,000 samples).
        """
        if self.is_foundation_model_loaded and self.hear_model is not None:
            try:
                with torch.no_grad():
                    tensor = torch.from_numpy(audio_2s).float().unsqueeze(0).to(self.device)
                    output = self.hear_model(tensor)
                    if hasattr(output, "last_hidden_state"):
                        emb = output.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    elif isinstance(output, torch.Tensor):
                        emb = output.cpu().numpy()[0]
                    else:
                        raise ValueError("Unexpected HeAR model output")
                    # Normalize
                    norm = np.linalg.norm(emb) + 1e-8
                    return (emb / norm).astype(np.float32)
            except Exception as e:
                raise RuntimeError("HeAR neural forward failed") from e
        else:
            return self.fallback_extractor.extract_features(audio_2s)

    def extract_embeddings_for_windows(
        self, windows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Processes a list of window dicts produced by AudioPreprocessor,
        attaching a 512-dim 'embedding' field to each window.
        """
        for w in windows:
            samples = w["samples"]
            embedding = self.extract_embedding(samples)
            w["embedding"] = embedding
            w["embedding_norm"] = float(np.linalg.norm(embedding))

        return windows
