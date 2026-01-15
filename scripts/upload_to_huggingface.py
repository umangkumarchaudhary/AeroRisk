"""
Upload AeroRisk Models to Hugging Face Hub
============================================
This script uploads your trained ML models to Hugging Face Hub
so they can be downloaded by the production API.

Usage:
    1. pip install huggingface_hub
    2. huggingface-cli login
    3. python scripts/upload_to_huggingface.py

Author: Umang Kumar
Date: 2024-01-15
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi, create_repo, upload_file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

# Your Hugging Face username
HF_USERNAME = "UmangKumar17"  # Your actual HF username
HF_REPO_NAME = "aerorisk-models"
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Files to upload
FILES_TO_UPLOAD = [
    # Models
    "xgboost_risk_predictor_v1.pkl",
    "lightgbm_severity_classifier_v1.pkl",
    "isolation_forest_anomaly_v1.pkl",
    "anomaly_scaler.pkl",
    # Feature files
    "xgboost_risk_features.txt",
    "anomaly_features.txt",
    "severity_class_mapping.txt",
]


# ============================================
# Main Upload Function
# ============================================

def upload_models():
    """Upload all models to Hugging Face Hub."""
    print("=" * 60)
    print("🤗 AeroRisk - Upload Models to Hugging Face Hub")
    print("=" * 60)
    
    api = HfApi()
    
    # Check authentication
    try:
        user = api.whoami()
        print(f"\n✅ Logged in as: {user['name']}")
    except Exception as e:
        print(f"\n❌ Not logged in! Run: huggingface-cli login")
        print(f"   Error: {e}")
        return
    
    # Create repo if not exists
    print(f"\n📦 Creating/accessing repo: {HF_REPO_ID}")
    try:
        create_repo(
            repo_id=HF_REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"   ✅ Repository ready")
    except Exception as e:
        print(f"   ❌ Error creating repo: {e}")
        return
    
    # Upload files
    print(f"\n📤 Uploading files from {MODELS_DIR}...")
    
    uploaded = []
    failed = []
    
    for filename in FILES_TO_UPLOAD:
        filepath = MODELS_DIR / filename
        
        if not filepath.exists():
            print(f"   ⚠️  Skipped (not found): {filename}")
            continue
        
        try:
            print(f"   📤 Uploading: {filename}...", end=" ")
            
            upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=HF_REPO_ID,
                repo_type="model"
            )
            
            print("✅")
            uploaded.append(filename)
            
        except Exception as e:
            print(f"❌ {e}")
            failed.append(filename)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 UPLOAD SUMMARY")
    print("=" * 60)
    print(f"\n✅ Uploaded: {len(uploaded)} files")
    for f in uploaded:
        print(f"   - {f}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} files")
        for f in failed:
            print(f"   - {f}")
    
    print(f"\n🔗 Your models are now at:")
    print(f"   https://huggingface.co/{HF_REPO_ID}")
    
    print("\n" + "=" * 60)
    print("🎉 DONE! Your models are now hosted on Hugging Face!")
    print("=" * 60)


if __name__ == "__main__":
    upload_models()
