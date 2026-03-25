"""
Model loading module for FastAPI
Handles efficient loading of all models and preprocessing objects
"""

import joblib
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent / "models"


class ModelLoader:
    """Load and manage all ML models"""

    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.scaler = None
        self.features = None
        self.label_encoder_target = None
        self.is_loaded = False

    def load_models(self):
        """Load all models at startup"""
        try:
            logger.info("Loading models...")

            # Check if model directory exists
            if not MODEL_DIR.exists():
                raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

            # Load regression model
            reg_path = MODEL_DIR / "regression_model.pkl"
            if not reg_path.exists():
                raise FileNotFoundError(f"Regression model not found: {reg_path}")
            self.regression_model = joblib.load(reg_path)
            logger.info(f"[OK] Regression model loaded from {reg_path}")

            # Load classification model
            clf_path = MODEL_DIR / "classification_model.pkl"
            if not clf_path.exists():
                raise FileNotFoundError(f"Classification model not found: {clf_path}")
            self.classification_model = joblib.load(clf_path)
            logger.info(f"[OK] Classification model loaded from {clf_path}")

            # Load scaler
            scaler_path = MODEL_DIR / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            logger.info(f"[OK] Scaler loaded from {scaler_path}")

            # Load features
            features_path = MODEL_DIR / "features.pkl"
            if not features_path.exists():
                raise FileNotFoundError(f"Features not found: {features_path}")
            self.features = joblib.load(features_path)
            logger.info(f"[OK] Features loaded from {features_path} ({len(self.features)} features)")

            # Load label encoder for target
            le_path = MODEL_DIR / "label_encoder_target.pkl"
            if not le_path.exists():
                raise FileNotFoundError(f"Label encoder not found: {le_path}")
            self.label_encoder_target = joblib.load(le_path)
            logger.info(f"[OK] Label encoder loaded from {le_path}")

            self.is_loaded = True
            logger.info("[SUCCESS] All models loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to load models: {str(e)}")
            self.is_loaded = False
            raise

    def predict(self, X_scaled):
        """Make predictions"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        reg_pred = self.regression_model.predict(X_scaled)[0]
        clf_pred = self.classification_model.predict(X_scaled)[0]
        clf_prob = self.classification_model.predict_proba(X_scaled)[0]

        # Decode classification prediction
        risk_level = self.label_encoder_target.inverse_transform([clf_pred])[0]

        # Get confidence (max probability)
        confidence = float(max(clf_prob))

        return {
            'burnout_score': float(reg_pred),
            'risk_level': risk_level,
            'confidence': confidence
        }

    def get_shap_explainer(self):
        """Get SHAP explainer for model"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        try:
            import shap
            explainer = shap.TreeExplainer(self.regression_model)
            return explainer
        except ImportError:
            logger.warning("SHAP not installed. Explanations not available.")
            return None


# Global model loader instance
model_loader = ModelLoader()
