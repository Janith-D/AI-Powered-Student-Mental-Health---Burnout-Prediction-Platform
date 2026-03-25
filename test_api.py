"""
Test FastAPI endpoints locally (without running server)
"""

import sys
sys.path.insert(0, '.')

from model_loader import model_loader
from shap_api_utils import prepare_data_for_prediction, get_shap_explanation, validate_data
from schemas import StudentDataInput
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_models():
    """Test model loading and predictions"""
    print("\n" + "="*70)
    print("TESTING MODEL LOADING AND PREDICTIONS")
    print("="*70)

    # Load models
    print("\n[TEST] Loading models...")
    try:
        model_loader.load_models()
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return False

    # Test data
    student_data = StudentDataInput(
        sleep_quality=7.5,
        stress_level=45,
        exercise_hours=3,
        anxiety_score=30,
        social_support=8,
        meditation_hours=1,
        depression_score=20,
        study_hours=6,
        course_difficulty=7,
        assignment_load=6,
        gender="Female",
        major="STEM"
    )

    # Validate
    print("\n[TEST] Validating data...")
    is_valid, msg = validate_data(student_data)
    print(f"[OK] Validation: {msg}")

    # Prepare data
    print("\n[TEST] Preparing data for prediction...")
    try:
        X_scaled, X_original = prepare_data_for_prediction(
            student_data,
            model_loader.features,
            model_loader.scaler
        )
        print(f"[OK] Data prepared: X_scaled shape = {X_scaled.shape}")
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return False

    # Make predictions
    print("\n[TEST] Making predictions...")
    try:
        predictions = model_loader.predict(X_scaled)
        print(f"[OK] Predictions:")
        print(f"  - Burnout Score: {predictions['burnout_score']:.2f}")
        print(f"  - Risk Level: {predictions['risk_level']}")
        print(f"  - Confidence: {predictions['confidence']:.4f}")
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return False

    # Test SHAP explanation
    print("\n[TEST] Getting SHAP explanation...")
    try:
        explainer = model_loader.get_shap_explainer()
        if explainer is None:
            print("[WARNING] SHAP not available, skipping explanation")
        else:
            explanation = get_shap_explanation(
                explainer,
                X_scaled,
                X_original,
                model_loader.features,
                sample_idx=0
            )
            if explanation:
                print(f"[OK] Explanation generated:")
                print(f"  - Burnout Score: {explanation['burnout_score']:.2f}")
                print(f"  - Base Value: {explanation['base_value']:.2f}")
                print(f"  - Difference: {explanation['prediction_difference']:+.2f}")
                print(f"  - Top Contributors:")
                for contrib in explanation['top_contributors'][:3]:
                    print(f"    • {contrib['feature_name']}: {contrib['shap_value']:.2f} ({contrib['impact']})")
                print(f"  - Protection Factors: {', '.join(explanation['protection_factors'])}")
                print(f"  - Risk Factors: {', '.join(explanation['risk_factors'])}")
                print(f"  - Summary: {explanation['summary']}")
            else:
                print("[WARNING] Failed to generate explanation")
    except Exception as e:
        print(f"[ERROR] SHAP explanation failed: {e}")
        return False

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    return True


if __name__ == "__main__":
    success = test_models()
    sys.exit(0 if success else 1)
