"""
Advanced SHAP Analysis Functions - Reusable Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')


def explain_single_prediction(model, scaler, X_data, feature_names, sample_idx=0, base_value=None):
    """
    Explain a single prediction with SHAP values

    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        X_data: Input features (DataFrame or numpy array)
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        base_value: Expected value (calculated if None)

    Returns:
        Dictionary with prediction explanation
    """
    explainer = shap.TreeExplainer(model)
    X_scaled = scaler.transform(X_data) if hasattr(scaler, 'transform') else X_data

    shap_values = explainer.shap_values(X_scaled)
    prediction = model.predict(X_scaled)[sample_idx]
    base = base_value or explainer.expected_value

    top_idx = np.argsort(np.abs(shap_values[sample_idx]))[::-1][:10]

    return {
        'prediction': prediction,
        'base_value': base,
        'difference': prediction - base,
        'top_features': [(feature_names[i], shap_values[sample_idx][i]) for i in top_idx],
        'all_shap_values': shap_values[sample_idx]
    }


def get_feature_directions(model, X_train, feature_names):
    """
    Determine if features increase or decrease the target

    Returns:
        Dictionary with feature directions and magnitudes
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    mean_shap = shap_values.mean(axis=0)

    directions = {}
    for i, feature in enumerate(feature_names):
        directions[feature] = {
            'direction': 'Increases' if mean_shap[i] > 0 else 'Decreases',
            'magnitude': abs(mean_shap[i]),
            'shap_value': mean_shap[i]
        }

    return directions


def create_dependence_plot(model, X_data, feature_names, feature_to_plot, sample_size=None):
    """
    Create SHAP dependence plot for a specific feature

    Shows how the feature value affects the model output
    """
    explainer = shap.TreeExplainer(model)

    if sample_size and len(X_data) > sample_size:
        X_sample = X_data.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_data

    shap_values = explainer.shap_values(X_sample.values)

    feature_idx = feature_names.index(feature_to_plot)

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot - {feature_to_plot}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{feature_to_plot}.png', dpi=300, bbox_inches='tight')
    plt.close()


def waterfall_plot(model, X_data, feature_names, sample_idx=0):
    """
    Create SHAP waterfall plot - shows how each feature contributes to final prediction
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data.values)

    plt.figure(figsize=(12, 8))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[sample_idx],
        X_data.iloc[sample_idx],
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Waterfall Plot - Sample #{sample_idx}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("[OK] SHAP utilities module loaded")
    print("Use: from shap_utils import explain_single_prediction, get_feature_directions")
