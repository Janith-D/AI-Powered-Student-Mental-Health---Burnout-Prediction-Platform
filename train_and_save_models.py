"""
Train and save models for deployment
Saves: regression model, classification model, scaler, features
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "student_mental_health_burnout.csv"
RANDOM_STATE = 42

IMPORTANT_FEATURES = [
    'sleep_quality', 'stress_level', 'exercise_hours', 'anxiety_score',
    'social_support', 'meditation_hours', 'depression_score',
    'study_hours', 'course_difficulty', 'assignment_load'
]

COLUMNS_TO_DROP = ['student_id', 'age', 'study_year', 'gpa', 'sleep_hours', 'bmi', 'caffeine_intake']


def prepare_data():
    """Load and preprocess data"""
    print("[LOAD] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Clean
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], inplace=True)

    # Encode
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
    y_regression = df['burnout_score'].values
    y_classification_raw = df['risk_level'].values

    # Encode classification target
    le_target = LabelEncoder()
    y_classification = le_target.fit_transform(y_classification_raw)

    print(f"[OK] Features: {len(available_features)}")
    return X, y_regression, y_classification, available_features, le_target


def train_models(X, y_regression, y_classification):
    """Train both regression and classification models"""
    print("\n[TRAIN] Splitting data...")
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_regression, y_classification, test_size=0.2, random_state=RANDOM_STATE, stratify=y_classification
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[OK] Train: {len(X_train):,}, Test: {len(X_test):,}")
    print("[TRAIN] Training regression model...")

    reg_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    reg_model.fit(X_train_scaled, y_reg_train)

    reg_score = reg_model.score(X_test_scaled, y_reg_test)
    print(f"[OK] Regression R² Score: {reg_score:.4f}")

    print("[TRAIN] Training classification model...")

    clf_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf_model.fit(X_train_scaled, y_clf_train)

    clf_score = clf_model.score(X_test_scaled, y_clf_test)
    print(f"[OK] Classification Accuracy: {clf_score:.4f}")

    return reg_model, clf_model, scaler


def save_models(reg_model, clf_model, scaler, features, le_target):
    """Save models and preprocessing objects"""
    print("\n[SAVE] Saving models...")

    joblib.dump(reg_model, 'models/regression_model.pkl')
    print("[OK] Regression model saved: models/regression_model.pkl")

    joblib.dump(clf_model, 'models/classification_model.pkl')
    print("[OK] Classification model saved: models/classification_model.pkl")

    joblib.dump(scaler, 'models/scaler.pkl')
    print("[OK] Scaler saved: models/scaler.pkl")

    joblib.dump(features, 'models/features.pkl')
    print("[OK] Features saved: models/features.pkl")

    joblib.dump(le_target, 'models/label_encoder_target.pkl')
    print("[OK] Target encoder saved: models/label_encoder_target.pkl")


def main():
    """Main pipeline"""
    print("="*70)
    print("MODEL TRAINING & SERIALIZATION FOR DEPLOYMENT")
    print("="*70)

    X, y_reg, y_clf, features, le_target = prepare_data()
    reg_model, clf_model, scaler = train_models(X, y_reg, y_clf)
    save_models(reg_model, clf_model, scaler, features, le_target)

    print("\n" + "="*70)
    print("MODELS SAVED SUCCESSFULLY")
    print("="*70)
    print("\n[READY] Models are ready for deployment")
    print("[FILES]")
    print("  - models/regression_model.pkl")
    print("  - models/classification_model.pkl")
    print("  - models/scaler.pkl")
    print("  - models/features.pkl")
    print("  - models/label_encoder_target.pkl\n")


if __name__ == "__main__":
    main()
