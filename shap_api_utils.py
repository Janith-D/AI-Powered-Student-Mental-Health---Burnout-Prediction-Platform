"""
SHAP explanation utilities for FastAPI
Provides explanations for model predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def get_shap_explanation(explainer, X_scaled, X_original, features, sample_idx=0):
    """
    Get SHAP explanation for a single prediction

    Args:
        explainer: SHAP TreeExplainer
        X_scaled: Scaled feature values
        X_original: Original (unscaled) feature values
        features: Feature names
        sample_idx: Index to explain

    Returns:
        Dictionary with explanation
    """
    try:
        if explainer is None:
            return None

        shap_values = explainer.shap_values(X_scaled)
        base_value = float(explainer.expected_value)
        prediction = float(explainer.model.predict(X_scaled)[sample_idx])

        # Get top contributing features
        sample_shap = shap_values[sample_idx]
        top_idx = np.argsort(np.abs(sample_shap))[::-1][:10]

        contributors = []
        protection_factors = []
        risk_factors = []

        for idx in top_idx:
            feature_name = features[idx]
            feature_value = float(X_original.iloc[sample_idx, idx]) if hasattr(X_original, 'iloc') else float(X_original[sample_idx, idx])
            shap_val = float(sample_shap[idx])
            abs_shap = abs(shap_val)

            # Determine direction
            direction = "increases" if shap_val > 0 else "decreases"

            # Determine impact strength
            if abs_shap > 5:
                impact = "strong"
            elif abs_shap > 1:
                impact = "moderate"
            else:
                impact = "weak"

            contributor = {
                'feature_name': feature_name,
                'feature_value': feature_value,
                'shap_value': shap_val,
                'direction': direction,
                'impact': impact
            }
            contributors.append(contributor)

            # Categorize
            if direction == "decreases" and impact in ["strong", "moderate"]:
                protection_factors.append(feature_name)
            elif direction == "increases" and impact in ["strong", "moderate"]:
                risk_factors.append(feature_name)

        difference = prediction - base_value

        # Generate summary
        summary = _generate_summary(protection_factors, risk_factors, difference)

        return {
            'burnout_score': prediction,
            'base_value': base_value,
            'prediction_difference': difference,
            'top_contributors': contributors[:5],
            'protection_factors': list(set(protection_factors))[:3],
            'risk_factors': list(set(risk_factors))[:3],
            'summary': summary
        }

    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        return None


def _generate_summary(protection_factors: List[str], risk_factors: List[str], difference: float) -> str:
    """Generate human-readable summary"""
    summary_parts = []

    if difference < -10:
        summary_parts.append("This student has SIGNIFICANTLY LOWER burnout than average.")
    elif difference < 0:
        summary_parts.append("This student has lower burnout than average.")
    elif difference < 10:
        summary_parts.append("This student has typical burnout levels.")
    else:
        summary_parts.append("This student has higher burnout than average.")

    if protection_factors:
        factors_str = ", ".join(protection_factors)
        summary_parts.append(f"Protective factors: {factors_str}.")

    if risk_factors:
        factors_str = ", ".join(risk_factors)
        summary_parts.append(f"Risk factors: {factors_str}.")

    return " ".join(summary_parts)


def prepare_data_for_prediction(student_data, features, scaler, label_encoder=None):
    """
    Prepare student data for model prediction

    Args:
        student_data: StudentDataInput object
        features: List of feature names
        scaler: Fitted StandardScaler
        label_encoder: LabelEncoder for major (optional)

    Returns:
        Tuple of (X_scaled, X_original)
    """
    try:
        # Convert to dictionary
        data_dict = student_data.dict()

        # Encode gender
        gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
        data_dict['gender'] = gender_map.get(data_dict['gender'], 1)

        # One-hot encode major
        major_value = data_dict.pop('major', 'STEM')
        for major_option in ['Arts', 'Business', 'Humanities', 'STEM']:
            data_dict[f'major_{major_option}'] = 1 if major_option == major_value else 0

        # Create feature vector
        X_values = []
        for feature in features:
            X_values.append(data_dict.get(feature, 0))

        X_original = np.array([X_values])
        X_scaled = scaler.transform(X_original)

        # Also return as DataFrame for SHAP
        X_df = pd.DataFrame(X_original, columns=features)

        return X_scaled, X_df

    except Exception as e:
        logger.error(f"Error preparing data for prediction: {str(e)}")
        raise


def validate_data(student_data) -> Tuple[bool, str]:
    """Validate student input data"""
    try:
        # Basic validation
        if student_data.sleep_quality < 1 or student_data.sleep_quality > 10:
            return False, "sleep_quality must be between 1 and 10"

        if student_data.stress_level < 0 or student_data.stress_level > 100:
            return False, "stress_level must be between 0 and 100"

        if student_data.exercise_hours < 0:
            return False, "exercise_hours cannot be negative"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
