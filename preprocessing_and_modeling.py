"""
Data Preprocessing and Model Building Pipeline
Student Mental Health & Burnout Prediction
Production-ready, optimized for large datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "student_mental_health_burnout.csv"
SAMPLE_SIZE = 100000  # Set to None to use full dataset
RANDOM_STATE = 42

# Feature configuration
IMPORTANT_FEATURES = [
    'sleep_quality', 'stress_level', 'exercise_hours', 'anxiety_score',
    'social_support', 'meditation_hours', 'depression_score',
    'study_hours', 'course_difficulty', 'assignment_load'
]

COLUMNS_TO_DROP = ['student_id', 'age', 'study_year', 'gpa', 'sleep_hours', 'bmi', 'caffeine_intake']


def load_data(path, sample_size=None):
    """Load dataset with optional sampling for large files"""
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)

    try:
        df = pd.read_csv(path)

        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_STATE)
            print(f"[OK] Loaded sample: {len(df):,} rows (total: {len(df):,})")
        else:
            print(f"[OK] Full dataset loaded: {len(df):,} rows")

        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        return None


def clean_data(df):
    """Handle missing values and drop unnecessary columns"""
    print("\n" + "="*70)
    print("STEP 2: DATA CLEANING")
    print("="*70)

    df_clean = df.copy()

    # Handle missing values - median imputation
    print(f"\n[MISSING] Handling missing values:")
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  - {col}: {missing_count} values -> median imputation")

    # Drop unnecessary columns
    print(f"\n[DROP] Removing unnecessary columns:")
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df_clean.columns]
    df_clean.drop(columns=cols_to_drop, inplace=True)
    print(f"  - Dropped: {cols_to_drop}")

    print(f"\n[SHAPE] After cleaning: {df_clean.shape}")
    return df_clean


def encode_features(df):
    """Encode categorical features"""
    print("\n" + "="*70)
    print("STEP 3: ENCODING CATEGORICAL FEATURES")
    print("="*70)

    df_encoded = df.copy()

    # Label encoding for gender
    if 'gender' in df_encoded.columns:
        le_gender = LabelEncoder()
        df_encoded['gender'] = le_gender.fit_transform(df_encoded['gender'].astype(str))
        print(f"\n[LABEL] gender: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

    # One-hot encoding for major
    if 'major' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['major'], prefix='major', drop_first=False)
        print(f"[ONE-HOT] major: 4 categories -> 4 binary columns")

    print(f"\n[SHAPE] After encoding: {df_encoded.shape}")
    return df_encoded


def select_features(df):
    """Select important features for modeling"""
    print("\n" + "="*70)
    print("STEP 4: FEATURE SELECTION")
    print("="*70)

    # Get available features from important list
    available_features = [f for f in IMPORTANT_FEATURES if f in df.columns]

    # Add one-hot encoded major columns
    major_cols = [col for col in df.columns if col.startswith('major_')]
    available_features.extend(major_cols)

    # Add encoded gender if it exists
    if 'gender' in df.columns:
        available_features.append('gender')

    print(f"\n[SELECTED] {len(available_features)} features:")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i}. {feat}")

    return available_features


def prepare_targets(df):
    """Prepare regression and classification targets"""
    print("\n" + "="*70)
    print("STEP 5: PREPARING TARGETS")
    print("="*70)

    y_regression = df['burnout_score'].values
    y_classification = df['risk_level'].values

    print(f"\n[REGRESSION] burnout_score")
    print(f"  - Type: Continuous (0-100)")
    print(f"  - Samples: {len(y_regression):,}")
    print(f"  - Range: [{y_regression.min():.2f}, {y_regression.max():.2f}]")
    print(f"  - Mean: {y_regression.mean():.2f}")

    print(f"\n[CLASSIFICATION] risk_level")
    unique_classes = np.unique(y_classification)
    print(f"  - Type: Multi-class ({len(unique_classes)} classes)")
    print(f"  - Classes: {unique_classes}")
    for cls in unique_classes:
        count = np.sum(y_classification == cls)
        pct = count / len(y_classification) * 100
        print(f"    - {cls}: {count:,} ({pct:.1f}%)")

    return y_regression, y_classification


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print("\n" + "="*70)
    print("STEP 6: TRAIN-TEST SPLIT")
    print("="*70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n[SPLIT] 80-20 Train-Test Split")
    print(f"  - Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Apply feature scaling"""
    print("\n" + "="*70)
    print("STEP 7: FEATURE SCALING")
    print("="*70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[SCALING] StandardScaler applied")
    print(f"  - Train mean: {X_train_scaled.mean():.4f}")
    print(f"  - Train std:  {X_train_scaled.std():.4f}")
    print(f"  - Test mean:  {X_test_scaled.mean():.4f}")
    print(f"  - Test std:   {X_test_scaled.std():.4f}")

    return X_train_scaled, X_test_scaled, scaler


def train_regression_model(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=6):
    """Train XGBoost Regression model"""
    print("\n" + "="*70)
    print("STEP 8: TRAINING XGBoost REGRESSOR")
    print("="*70)

    print(f"\n[PARAMS] Model Configuration:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - max_depth: {max_depth}")

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    print(f"\n[TRAINING] Starting model training...")
    model.fit(X_train, y_train)
    print(f"[OK] Model trained successfully")

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    print("\n" + "="*70)
    print("STEP 9: MODEL EVALUATION")
    print("="*70)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n[TRAIN SET]")
    print(f"  - RMSE: {train_rmse:.4f}")
    print(f"  - R² Score: {train_r2:.4f}")

    print(f"\n[TEST SET]")
    print(f"  - RMSE: {test_rmse:.4f}")
    print(f"  - R² Score: {test_r2:.4f}")

    # Overfitting check
    rmse_diff = test_rmse - train_rmse
    if rmse_diff > 2:
        print(f"\n[WARNING] Potential overfitting detected (RMSE gap: {rmse_diff:.2f})")
    else:
        print(f"\n[OK] Good generalization (RMSE gap: {rmse_diff:.2f})")

    return {
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }


def get_feature_importance(model, feature_names, top_n=10):
    """Extract and display feature importance"""
    print("\n" + "="*70)
    print("STEP 10: FEATURE IMPORTANCE")
    print("="*70)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n[TOP {top_n}] Important Features:")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}")

    return importances, indices


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("DATA PREPROCESSING & MODEL BUILDING PIPELINE")
    print("="*70)

    # Step 1: Load data
    df = load_data(DATASET_PATH, sample_size=SAMPLE_SIZE)
    if df is None:
        return

    # Step 2: Clean data
    df = clean_data(df)

    # Step 3: Encode categorical features
    df = encode_features(df)

    # Step 4: Select features
    selected_features = select_features(df)
    X = df[selected_features].values

    # Step 5: Prepare targets
    y_regression, y_classification = prepare_targets(df)

    # Step 6: Train-Test split (for regression)
    X_train, X_test, y_train, y_test = split_data(X, y_regression)

    # Step 7: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 8: Train regression model
    model = train_regression_model(X_train_scaled, y_train)

    # Step 9: Evaluate model
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 10: Feature importance
    importances, top_indices = get_feature_importance(model, selected_features)

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\n[SUMMARY]")
    print(f"  - Dataset: {len(df):,} samples")
    print(f"  - Features: {len(selected_features)}")
    print(f"  - Test R² Score: {metrics['test_r2']:.4f}")
    print(f"  - Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"\n[NEXT STEPS]")
    print(f"  - Classification model training")
    print(f"  - Hyperparameter tuning")
    print(f"  - Cross-validation")
    print(f"  - Model deployment\n")

    return {
        'df': df,
        'model': model,
        'scaler': scaler,
        'features': selected_features,
        'metrics': metrics,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_classification': y_classification
    }


if __name__ == "__main__":
    results = main()
