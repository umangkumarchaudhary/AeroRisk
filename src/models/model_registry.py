"""
AeroRisk - Model Registry
==========================
A production-grade model registry for tracking ML models, versions, and performance.

Features:
- Model version tracking
- Performance metrics storage
- Model metadata management
- Deployment status tracking
- Model comparison utilities

Author: Umang Kumar
Date: 2024-01-14
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


# ============================================
# Data Classes for Model Registry
# ============================================

@dataclass
class ModelMetrics:
    """Performance metrics for a model."""
    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # General metrics
    training_samples: int = 0
    test_samples: int = 0
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    
    # Custom metrics
    custom: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """A specific version of a registered model."""
    version: str
    created_at: str
    model_path: str
    model_hash: str
    metrics: ModelMetrics
    features: List[str]
    hyperparameters: Dict[str, Any]
    description: str
    status: str  # "staging", "production", "archived"
    tags: List[str] = field(default_factory=list)


@dataclass 
class RegisteredModel:
    """A registered model with all its versions."""
    name: str
    model_type: str  # "regressor", "classifier", "anomaly_detector"
    description: str
    created_at: str
    updated_at: str
    versions: List[ModelVersion] = field(default_factory=list)
    production_version: Optional[str] = None
    staging_version: Optional[str] = None


class ModelRegistry:
    """
    Production-grade model registry for AeroRisk.
    
    Features:
    - Register new models
    - Track model versions
    - Store performance metrics
    - Manage deployment status
    - Compare model versions
    """
    
    def __init__(self, registry_path: str = None):
        """Initialize model registry."""
        if registry_path is None:
            registry_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'models', 'registry'
            )
        
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_path / "registry.json"
        self.models: Dict[str, RegisteredModel] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                for name, model_data in data.items():
                    # Reconstruct nested dataclasses
                    versions = []
                    for v in model_data.get('versions', []):
                        metrics = ModelMetrics(**v['metrics'])
                        v['metrics'] = metrics
                        versions.append(ModelVersion(**v))
                    model_data['versions'] = versions
                    self.models[name] = RegisteredModel(**model_data)
    
    def _save_registry(self):
        """Save registry to disk."""
        data = {}
        for name, model in self.models.items():
            model_dict = asdict(model)
            data[name] = model_dict
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _compute_model_hash(self, model_path: str) -> str:
        """Compute SHA256 hash of model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
    
    def register_model(
        self,
        name: str,
        model_type: str,
        description: str = ""
    ) -> RegisteredModel:
        """Register a new model in the registry."""
        if name in self.models:
            print(f"   Model '{name}' already registered")
            return self.models[name]
        
        now = datetime.now().isoformat()
        model = RegisteredModel(
            name=name,
            model_type=model_type,
            description=description,
            created_at=now,
            updated_at=now,
            versions=[],
            production_version=None,
            staging_version=None
        )
        
        self.models[name] = model
        self._save_registry()
        print(f"   [OK] Registered model: {name}")
        return model
    
    def log_version(
        self,
        model_name: str,
        model_path: str,
        metrics: ModelMetrics,
        features: List[str],
        hyperparameters: Dict[str, Any] = None,
        description: str = "",
        tags: List[str] = None
    ) -> ModelVersion:
        """Log a new version of a model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model = self.models[model_name]
        
        # Generate version number
        version_num = len(model.versions) + 1
        version = f"v{version_num}.0"
        
        # Copy model to registry
        version_dir = self.registry_path / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = version_dir / "model.pkl"
        shutil.copy2(model_path, dest_path)
        
        # Compute hash
        model_hash = self._compute_model_hash(str(dest_path))
        
        # Create version record
        version_record = ModelVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            model_path=str(dest_path),
            model_hash=model_hash,
            metrics=metrics,
            features=features,
            hyperparameters=hyperparameters or {},
            description=description,
            status="staging",
            tags=tags or []
        )
        
        model.versions.append(version_record)
        model.updated_at = datetime.now().isoformat()
        model.staging_version = version
        
        # Save feature list
        with open(version_dir / "features.json", 'w') as f:
            json.dump(features, f, indent=2)
        
        # Save hyperparameters
        with open(version_dir / "hyperparameters.json", 'w') as f:
            json.dump(hyperparameters or {}, f, indent=2)
        
        # Save metrics
        with open(version_dir / "metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        self._save_registry()
        print(f"   [OK] Logged version {version} for {model_name}")
        return version_record
    
    def promote_to_production(self, model_name: str, version: str = None):
        """Promote a model version to production."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model = self.models[model_name]
        
        if version is None:
            version = model.staging_version
        
        if version is None:
            raise ValueError("No version to promote")
        
        # Archive current production
        if model.production_version:
            for v in model.versions:
                if v.version == model.production_version:
                    v.status = "archived"
        
        # Promote new version
        for v in model.versions:
            if v.version == version:
                v.status = "production"
                model.production_version = version
                break
        
        model.updated_at = datetime.now().isoformat()
        self._save_registry()
        print(f"   [OK] Promoted {model_name} {version} to production")
    
    def get_production_model(self, model_name: str):
        """Load the production version of a model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model = self.models[model_name]
        
        if not model.production_version:
            raise ValueError(f"No production version for {model_name}")
        
        for v in model.versions:
            if v.version == model.production_version:
                return joblib.load(v.model_path)
        
        raise ValueError(f"Production version not found")
    
    def compare_versions(self, model_name: str) -> pd.DataFrame:
        """Compare all versions of a model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model = self.models[model_name]
        
        rows = []
        for v in model.versions:
            row = {
                'version': v.version,
                'status': v.status,
                'created': v.created_at[:10],
                'samples': v.metrics.training_samples,
            }
            
            # Add relevant metrics based on model type
            if model.model_type == 'regressor':
                row['rmse'] = v.metrics.rmse
                row['r2'] = v.metrics.r2
            elif model.model_type == 'classifier':
                row['accuracy'] = v.metrics.accuracy
                row['f1'] = v.metrics.f1_score
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def list_models(self) -> pd.DataFrame:
        """List all registered models."""
        rows = []
        for name, model in self.models.items():
            rows.append({
                'name': name,
                'type': model.model_type,
                'versions': len(model.versions),
                'production': model.production_version or '-',
                'staging': model.staging_version or '-',
                'updated': model.updated_at[:10]
            })
        return pd.DataFrame(rows)
    
    def get_model_info(self, model_name: str) -> dict:
        """Get detailed info about a model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        return asdict(self.models[model_name])


# ============================================
# Main Execution - Register Existing Models
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("📦 AeroRisk - Model Registry Setup")
    print("=" * 60)
    
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    
    # Initialize registry
    print("\n🔧 Initializing Model Registry...")
    registry = ModelRegistry()
    
    # =========================================
    # Register XGBoost Risk Predictor
    # =========================================
    print("\n📊 Registering XGBoost Risk Predictor...")
    
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_risk_predictor_v1.pkl')
    xgb_features_path = os.path.join(MODELS_DIR, 'xgboost_risk_features.txt')
    
    if os.path.exists(xgb_path):
        registry.register_model(
            name="risk_predictor",
            model_type="regressor",
            description="XGBoost model for predicting flight risk scores (0-100)"
        )
        
        # Load features
        features = []
        if os.path.exists(xgb_features_path):
            with open(xgb_features_path) as f:
                features = f.read().strip().split('\n')
        
        # Log version
        registry.log_version(
            model_name="risk_predictor",
            model_path=xgb_path,
            metrics=ModelMetrics(
                rmse=11.04,
                r2=0.85,
                training_samples=26498,
                test_samples=6625
            ),
            features=features,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            },
            description="Initial XGBoost risk predictor",
            tags=["xgboost", "risk", "regression"]
        )
        
        # Promote to production
        registry.promote_to_production("risk_predictor")
    
    # =========================================
    # Register LightGBM Severity Classifier
    # =========================================
    print("\n📊 Registering LightGBM Severity Classifier...")
    
    lgb_path = os.path.join(MODELS_DIR, 'lightgbm_severity_classifier_v1.pkl')
    
    if os.path.exists(lgb_path):
        registry.register_model(
            name="severity_classifier",
            model_type="classifier",
            description="LightGBM model for classifying incident severity (4 classes)"
        )
        
        registry.log_version(
            model_name="severity_classifier",
            model_path=lgb_path,
            metrics=ModelMetrics(
                accuracy=0.85,
                f1_score=0.78,
                training_samples=26498,
                test_samples=6625
            ),
            features=["Various numeric features"],
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 8,
                "class_weight": "balanced"
            },
            description="Initial LightGBM severity classifier",
            tags=["lightgbm", "classification", "severity"]
        )
        
        registry.promote_to_production("severity_classifier")
    
    # =========================================
    # Register Isolation Forest Anomaly Detector
    # =========================================
    print("\n📊 Registering Anomaly Detector...")
    
    iso_path = os.path.join(MODELS_DIR, 'isolation_forest_anomaly_v1.pkl')
    
    if os.path.exists(iso_path):
        registry.register_model(
            name="anomaly_detector",
            model_type="anomaly_detector",
            description="Isolation Forest for detecting anomalous operations"
        )
        
        registry.log_version(
            model_name="anomaly_detector",
            model_path=iso_path,
            metrics=ModelMetrics(
                custom={"contamination": 0.05, "anomalies_found": 2500},
                training_samples=50000
            ),
            features=["Operational risk features"],
            hyperparameters={
                "n_estimators": 100,
                "contamination": 0.05
            },
            description="Initial Isolation Forest anomaly detector",
            tags=["isolation_forest", "anomaly", "unsupervised"]
        )
        
        registry.promote_to_production("anomaly_detector")
    
    # =========================================
    # Display Registry Summary
    # =========================================
    print("\n" + "=" * 60)
    print("📦 MODEL REGISTRY SUMMARY")
    print("=" * 60)
    
    models_df = registry.list_models()
    print("\nRegistered Models:")
    print(models_df.to_string(index=False))
    
    # Show version comparison for each model
    for model_name in registry.models:
        print(f"\n{model_name} versions:")
        print(registry.compare_versions(model_name).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("[OK] Model Registry Setup Complete!")
    print("=" * 60)
    print(f"""
   Registry Location: {registry.registry_path}
   
   Files:
   - registry.json          (main registry)
   - <model>/<version>/     (version folders)
     - model.pkl            (model artifact)
     - features.json        (feature list)
     - hyperparameters.json (training params)
     - metrics.json         (performance)
   
   Usage:
   
   from model_registry import ModelRegistry
   
   registry = ModelRegistry()
   model = registry.get_production_model("risk_predictor")
   predictions = model.predict(new_data)
""")
    print("=" * 60)
