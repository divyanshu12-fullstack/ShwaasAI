"""
Dataset acquisition helper for ShwaasAI.

Provides instructions and download links for the standard respiratory datasets:
1. ICBHI 2017 Respiratory Sound Database (Public open access)
2. Coswara Dataset (IISc Bangalore - Open on GitHub)
3. COUGHVID (Zenodo - Open crowdsourced)
4. CODA TB DREAM Challenge (Synapse.org)
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DownloadDatasets")

DATASETS_INFO = {
    "ICBHI_2017": {
        "name": "ICBHI 2017 Respiratory Sound Database",
        "description": "920 respiratory recordings from 126 subjects with crackles, wheezes, and diagnosis labels.",
        "url": "https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge",
        "kaggle_mirror": "https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database",
        "setup_instructions": "Download and extract the WAV and TXT files into 'data/icbhi/'."
    },
    "COSWARA": {
        "name": "Coswara Dataset (IISc Bangalore)",
        "description": "Indian respiratory sound database (cough, breathing, phonation, symptoms).",
        "url": "https://github.com/iiscleap/Coswara-Data",
        "setup_instructions": "Run: git clone https://github.com/iiscleap/Coswara-Data.git data/coswara"
    },
    "COUGHVID": {
        "name": "COUGHVID Crowdsourced Cough Dataset",
        "description": "27,000+ cough audio files with physician clinical validations.",
        "url": "https://zenodo.org/records/4498364",
        "setup_instructions": "Download from Zenodo and extract into 'data/coughvid/'."
    },
    "CODA_TB": {
        "name": "CODA TB DREAM Challenge",
        "description": "733,756 cough sounds with microbiologically confirmed TB reference standard.",
        "url": "https://www.synapse.org/Synapse:syn31470195",
        "setup_instructions": "Register free account on Synapse.org, accept data usage agreement, and download."
    }
}


def print_dataset_guide():
    print("\n" + "="*80)
    print(" SHWAAS-AI: OPEN RESPIRATORY HEALTH DATASETS GUIDE")
    print("="*80)
    for key, info in DATASETS_INFO.items():
        print(f"\n[{info['name']}]")
        print(f" Description: {info['description']}")
        print(f" URL:         {info['url']}")
        if "kaggle_mirror" in info:
            print(f" Kaggle:      {info['kaggle_mirror']}")
        print(f" Quick Setup: {info['setup_instructions']}")
    print("\n" + "="*80)
    print(" NOTE:")
    print(" You can run the built-in acoustic benchmark training without waiting for any download:")
    print("   python backend/scripts/train_and_evaluate.py --n_samples 500")
    print(" Whenever you download an external dataset, simply point the training script to it:")
    print("   python backend/scripts/train_and_evaluate.py --data_dir data/icbhi/")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_dataset_guide()
