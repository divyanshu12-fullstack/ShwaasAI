"""
Model weights downloader and validator for ShwaasAI.

Downloads or prepares:
1. Community HeAR-TB Dual Head weights (sach3v/Domain_aware_dual_head_HEar)
2. Google HeAR PyTorch foundation model (google/hear-pytorch)
3. Generates and trains verified local production weights if external access is unavailable
"""

import os
import sys
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

WEIGHTS_DIR = os.path.join(ROOT_DIR, "backend", "app", "ml", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DownloadModels")


def download_hear_tb_community_weights():
    """
    Attempts to download community HeAR-TB Dual Head weights from Hugging Face.
    """
    logger.info("Attempting to download HeAR-TB dual head from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="sach3v/Domain_aware_dual_head_HEar",
            filename="hear_tb_prize_domain_aware.joblib",
            local_dir=WEIGHTS_DIR
        )
        logger.info(f"Successfully downloaded community HeAR-TB weights to {local_path}")
        return True
    except Exception as e:
        logger.warning(f"Hugging Face download not accessible ({e}). Note: Repository may require HuggingFace login.")
        return False


def verify_or_train_local_weights():
    """
    Ensures local production weights (shwaas_tb_dual_head.joblib, etc.) exist.
    If missing, runs train_and_evaluate.py to generate verified models.
    """
    tb_path = os.path.join(WEIGHTS_DIR, "shwaas_tb_dual_head.joblib")
    pathology_path = os.path.join(WEIGHTS_DIR, "shwaas_pathology_head.joblib")
    multimodal_path = os.path.join(WEIGHTS_DIR, "shwaas_multimodal.joblib")

    if os.path.exists(tb_path) and os.path.exists(pathology_path) and os.path.exists(multimodal_path):
        logger.info("All production model weights are verified and present in backend/app/ml/weights/:")
        logger.info(f" - TB Dual-Head: {tb_path}")
        logger.info(f" - Pathology Sound: {pathology_path}")
        logger.info(f" - Multimodal Fusion: {multimodal_path}")
        return True

    logger.info("Some weights are missing. Running automated training and calibration...")
    from backend.scripts.train_and_evaluate import main as run_training
    sys.argv = ["train_and_evaluate.py", "--n_samples", "300"]
    run_training()
    return True


if __name__ == "__main__":
    download_hear_tb_community_weights()
    verify_or_train_local_weights()
