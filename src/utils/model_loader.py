"""
AeroRisk - Model Loader
========================
Downloads ML models from Hugging Face Hub on startup.
This allows hosting models externally while keeping GitHub repo small.

Author: Umang Kumar
Date: 2024-01-15
"""

import os
import joblib
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

# Hugging Face Hub Repository
HF_REPO = "UmangKumar17/aerorisk-models"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Local cache directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


# ============================================
# Model Files to Download
# ============================================

MODEL_FILES = {
    "risk_predictor": "xgboost_risk_predictor_v1.pkl",
    "severity_classifier": "lightgbm_severity_classifier_v1.pkl", 
    "anomaly_detector": "isolation_forest_anomaly_v1.pkl",
    "anomaly_scaler": "anomaly_scaler.pkl",
}

FEATURE_FILES = {
    "risk_features": "xgboost_risk_features.txt",
    "anomaly_features": "anomaly_features.txt",
    "severity_mapping": "severity_class_mapping.txt",
}


# ============================================
# Download Functions
# ============================================

def download_file(url: str, local_path: Path) -> bool:
    """Download a file from URL to local path."""
    try:
        logger.info(f"Downloading {url}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded: {local_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def ensure_models_downloaded() -> dict:
    """Ensure all models are downloaded from Hugging Face Hub."""
    results = {"success": [], "failed": []}
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model files
    for name, filename in MODEL_FILES.items():
        local_path = MODELS_DIR / filename
        
        # Check if already exists
        if local_path.exists():
            logger.info(f"Model {name} already exists locally")
            results["success"].append(name)
            continue
        
        # Download from Hugging Face
        url = f"{HF_BASE_URL}/{filename}"
        if download_file(url, local_path):
            results["success"].append(name)
        else:
            results["failed"].append(name)
    
    # Download feature files
    for name, filename in FEATURE_FILES.items():
        local_path = MODELS_DIR / filename
        
        if local_path.exists():
            continue
            
        url = f"{HF_BASE_URL}/{filename}"
        download_file(url, local_path)
    
    return results


def load_model(model_name: str):
    """Load a model, downloading if necessary."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")
    
    filename = MODEL_FILES[model_name]
    local_path = MODELS_DIR / filename
    
    # Download if not exists
    if not local_path.exists():
        url = f"{HF_BASE_URL}/{filename}"
        if not download_file(url, local_path):
            return None
    
    return joblib.load(local_path)


def load_features(feature_name: str) -> list:
    """Load feature list from file."""
    if feature_name not in FEATURE_FILES:
        return []
    
    filename = FEATURE_FILES[feature_name]
    local_path = MODELS_DIR / filename
    
    # Download if not exists
    if not local_path.exists():
        url = f"{HF_BASE_URL}/{filename}"
        download_file(url, local_path)
    
    if local_path.exists():
        with open(local_path) as f:
            return f.read().strip().split('\n')
    
    return []


# ============================================
# Main - Test Download
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤗 AeroRisk - Model Loader (Hugging Face Hub)")
    print("=" * 60)
    
    print(f"\nHugging Face Repo: {HF_REPO}")
    print(f"Local Cache: {MODELS_DIR}")
    
    print("\n📥 Downloading models...")
    results = ensure_models_downloaded()
    
    print(f"\n✅ Success: {results['success']}")
    if results['failed']:
        print(f"❌ Failed: {results['failed']}")
    
    print("\n" + "=" * 60)
    print("""
    To upload your models to Hugging Face:
    
    1. Create account at https://huggingface.co
    2. Create new Model repository: aerorisk-models
    3. Install huggingface_hub: pip install huggingface_hub
    4. Upload models:
    
       from huggingface_hub import HfApi
       api = HfApi()
       api.upload_folder(
           folder_path="models",
           repo_id="umangkumarchaudhary/aerorisk-models",
           repo_type="model"
       )
    """)
    print("=" * 60)
