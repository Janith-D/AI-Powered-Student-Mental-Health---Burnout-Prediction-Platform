"""
SHAP Explainable AI Analysis for XGBoost Regression Model
Student Mental Health & Burnout Score Prediction
Production-Ready Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "student_mental_health_burnout.csv"
SAMPLE_SIZE = None  # Set to 1000 for faster testing on large datasets
RANDOM_STATE = 42

IMPORTANT_FEATURES = [
    'sleep_quality', 'stress_level', 'exercise_hours', 'anxiety_score',
    'social_support', 'meditation_hours', 'depression_score',
    'study_hours', 'course_difficulty', 'assignment_load'
]

COLUMNS_TO_DROP = ['student_id', 'age', 'study_year', 'gpa', 'sleep_hours', 'bmi', 'caffeine_intake']


def prepare_data():
    """Load and preprocess data for SHAP analysis"""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)

    # Load data
    df = pd.read_csv(DATASET_PATH)
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"[OK] Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Clean data
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], inplace=True)
    print(f"[OK] Cleaned: {df.shape}")

    # Encode categoricals
    if 'gender' in df.columns:
        le_gender = LabelEncoder()
        df['gender'] = le_gender.fit_transform(df['gender'].astype(str))

    if 'major' in df.columns:
        df = pd.get_dummies(df, columns=['major'], prefix='major', drop_first=False)

    # Select features
    available_features = [f for f in IMPORTANT_FEATURES if f in df.columns]
    major_cols = [col for col in df.columns if col.startswith('major_')]
    available_features.extend(major_cols)
    if 'gender' in df.columns:
        available_features.append('gender')

    X = df[available_features]
    y = df['burnout_score'].values

    print(f"[OK] Features: {len(available_features)}")
    return X, y, available_features


def train_model(X, y):
    """Train XGBoost model with preprocessing"""
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[OK] Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)
    print(f"[OK] Model trained")

    return model, scaler, X_train_scaled, X_test_scaled, X_train.columns.tolist(), y_test


def create_shap_explainer(model, X_train, feature_names):
    """Create SHAP explainer object"""
    print("\n" + "="*70)
    print("STEP 3: CREATING SHAP EXPLAINER")
    print("="*70)

    explainer = shap.TreeExplainer(model)

    # Use background data for SHAP (sample for efficiency)
    if len(X_train) > 1000:
        background_data = shap.sample(X_train, 100, random_state=RANDOM_STATE)
        print(f"[OK] Background data: 100 samples (from {len(X_train):,})")
    else:
        background_data = X_train
        print(f"[OK] Background data: {len(background_data):,} samples")

    explainer = shap.TreeExplainer(model, background_data)
    print(f"[OK] SHAP explainer created")

    return explainer, background_data


def compute_shap_values(explainer, X_test, feature_names):
    """Compute SHAP values for test data"""
    print("\n" + "="*70)
    print("STEP 4: COMPUTING SHAP VALUES")
    print("="*70)

    # Convert to DataFrame if needed
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # Sample for efficiency if dataset is large
    if len(X_test) > 500:
        X_test_sample = X_test.sample(n=500, random_state=RANDOM_STATE)
        print(f"[OK] Computing SHAP for 500 samples (from {len(X_test):,})")
    else:
        X_test_sample = X_test
        print(f"[OK] Computing SHAP for {len(X_test_sample):,} samples")

    shap_values = explainer.shap_values(X_test_sample.values)
    print(f"[OK] SHAP values computed: shape {shap_values.shape}")

    return shap_values, X_test_sample


def plot_shap_summary(explainer, X_test_sample, feature_names):
    """Generate SHAP summary plot"""
    print("\n" + "="*70)
    print("STEP 5: GENERATING SHAP SUMMARY PLOT")
    print("="*70)

    shap_values = explainer.shap_values(X_test_sample.values)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Burnout Score Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] shap_summary_plot.png")
    plt.close()


def plot_shap_bar(explainer, X_test_sample, feature_names):
    """Generate SHAP bar plot (mean importance)"""
    print("\n" + "="*70)
    print("STEP 6: GENERATING SHAP BAR PLOT")
    print("="*70)

    shap_values = explainer.shap_values(X_test_sample.values)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Mean Absolute SHAP Values)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] shap_bar_plot.png")
    plt.close()


def plot_force_plot(explainer, X_test_sample, feature_names, sample_idx=0):
    """Generate SHAP force plot for single prediction"""
    print("\n" + "="*70)
    print("STEP 7: GENERATING SHAP FORCE PLOT")
    print("="*70)

    shap_values = explainer.shap_values(X_test_sample.values)
    base_value = explainer.expected_value

    print(f"[INFO] Explaining prediction for sample #{sample_idx}")
    print(f"  - Base value (mean prediction): {base_value:.2f}")
    print(f"  - Actual prediction: {explainer.model.predict(X_test_sample.values)[sample_idx]:.2f}")

    # Create force plot
    shap.force_plot(
        base_value,
        shap_values[sample_idx],
        X_test_sample.iloc[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - Sample #{sample_idx}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] shap_force_plot.png")
    plt.close()


def analyze_feature_effects(explainer, X_test_sample, feature_names):
    """Analyze positive and negative feature effects"""
    print("\n" + "="*70)
    print("STEP 8: FEATURE EFFECT ANALYSIS")
    print("="*70)

    shap_values = explainer.shap_values(X_test_sample.values)

    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Get feature importance ranking
    importance_idx = np.argsort(mean_shap)[::-1]

    print(f"\n[ANALYSIS] Top 10 Features by SHAP Importance:")
    print(f"{'Rank':<5} {'Feature':<25} {'Mean |SHAP|':<15} {'Direction':<15}")
    print("-" * 60)

    for rank, idx in enumerate(importance_idx[:10], 1):
        feature = feature_names[idx]
        importance = mean_shap[idx]

        # Determine direction (positive or negative effect on output)
        mean_effect = shap_values[:, idx].mean()
        direction = "Increases burnout" if mean_effect > 0 else "Decreases burnout"

        print(f"{rank:<5} {feature:<25} {importance:<15.4f} {direction:<15}")

    # Detailed feature interpretation
    print(f"\n[INTERPRETATION]")
    print(f"\nFEATURES THAT INCREASE BURNOUT (Positive SHAP values):")
    for idx in importance_idx[:5]:
        feature = feature_names[idx]
        mean_effect = shap_values[:, idx].mean()
        if mean_effect > 0:
            print(f"  * {feature}")
            print(f"    - Average SHAP contribution: +{mean_effect:.2f}")
            print(f"    - Higher values -> Higher burnout")

    print(f"\nFEATURES THAT DECREASE BURNOUT (Negative SHAP values):")
    for idx in importance_idx[:5]:
        feature = feature_names[idx]
        mean_effect = shap_values[:, idx].mean()
        if mean_effect < 0:
            print(f"  * {feature}")
            print(f"    - Average SHAP contribution: {mean_effect:.2f}")
            print(f"    - Higher values -> Lower burnout (PROTECTIVE)")


def plot_individual_prediction(explainer, X_test_sample, feature_names, sample_idx=0):
    """Create detailed plot for individual prediction"""
    print("\n" + "="*70)
    print("STEP 9: INDIVIDUAL PREDICTION EXPLANATION")
    print("="*70)

    shap_values = explainer.shap_values(X_test_sample.values)
    prediction = explainer.model.predict(X_test_sample.values)[sample_idx]
    base_value = explainer.expected_value

    # Get top features for this prediction
    sample_shap = shap_values[sample_idx]
    top_features_idx = np.argsort(np.abs(sample_shap))[::-1][:10]

    print(f"\n[SAMPLE #{sample_idx}]")
    print(f"  - Base value (mean): {base_value:.2f}")
    print(f"  - Prediction: {prediction:.2f}")
    print(f"  - Difference: {prediction - base_value:+.2f}")

    print(f"\n[TOP 10 CONTRIBUTING FEATURES]")
    print(f"{'Feature':<25} {'Value':<12} {'SHAP Value':<12} {'Effect':<15}")
    print("-" * 65)

    for rank, idx in enumerate(top_features_idx, 1):
        feature = feature_names[idx]
        value = X_test_sample.iloc[sample_idx, idx]
        shap_val = sample_shap[idx]
        effect = "[+] Increases" if shap_val > 0 else "[-] Decreases"

        print(f"{feature:<25} {value:<12.2f} {shap_val:<12.4f} {effect:<15}")


def main():
    """Main SHAP analysis pipeline"""
    print("\n" + "="*70)
    print("SHAP EXPLAINABLE AI ANALYSIS - XGBoost REGRESSION MODEL")
    print("Student Mental Health & Burnout Score Prediction")
    print("="*70)

    # Step 1: Prepare data
    X, y, feature_names = prepare_data()

    # Step 2: Train model
    model, scaler, X_train_scaled, X_test_scaled, features, y_test = train_model(X, y)

    # Step 3: Create SHAP explainer
    explainer, background_data = create_shap_explainer(model, X_train_scaled, features)

    # Step 4: Compute SHAP values
    shap_values_test, X_test_sample = compute_shap_values(explainer, X_test_scaled, features)

    # Step 5: Generate visualizations
    plot_shap_summary(explainer, X_test_sample, features)
    plot_shap_bar(explainer, X_test_sample, features)
    plot_force_plot(explainer, X_test_sample, features, sample_idx=0)

    # Step 6: Feature analysis
    analyze_feature_effects(explainer, X_test_sample, features)

    # Step 7: Individual prediction explanation
    plot_individual_prediction(explainer, X_test_sample, features, sample_idx=0)

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n[FILES GENERATED]")
    print(f"  • shap_summary_plot.png - Feature importance (SHAP values)")
    print(f"  • shap_bar_plot.png - Mean absolute SHAP values")
    print(f"  • shap_force_plot.png - Individual prediction explanation")
    print(f"\n[INTERPRETATION]")
    print(f"  • Positive SHAP values = features that increase burnout")
    print(f"  • Negative SHAP values = protective factors (decrease burnout)")
    print(f"  • Magnitude = feature importance for that prediction\n")


if __name__ == "__main__":
    main()
