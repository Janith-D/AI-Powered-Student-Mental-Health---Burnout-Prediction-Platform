"""
Classification Model Training and Evaluation
Risk Level Prediction (Low/Medium/High)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier
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
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)

    # Load
    df = pd.read_csv(DATASET_PATH)
    print(f"\n[LOAD] Dataset: {df.shape}")

    # Clean
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], inplace=True)
    print(f"[CLEAN] After cleaning: {df.shape}")

    # Encode
    if 'gender' in df.columns:
        le_gender = LabelEncoder()
        df['gender'] = le_gender.fit_transform(df['gender'].astype(str))

    if 'major' in df.columns:
        df = pd.get_dummies(df, columns=['major'], prefix='major', drop_first=False)

    print(f"[ENCODE] After encoding: {df.shape}")

    # Select features
    available_features = [f for f in IMPORTANT_FEATURES if f in df.columns]
    major_cols = [col for col in df.columns if col.startswith('major_')]
    available_features.extend(major_cols)
    if 'gender' in df.columns:
        available_features.append('gender')

    X = df[available_features].values
    y_raw = df['risk_level'].values

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)

    print(f"[FEATURES] Selected: {len(available_features)}")
    print(f"[TARGET] Classes: {np.unique(y_raw)}")
    print(f"[ENCODE] Target encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    return X, y, available_features, le_target


def train_classification_model(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=6):
    """Train XGBoost classifier"""
    print("\n" + "="*70)
    print("TRAINING XGBoost CLASSIFIER")
    print("="*70)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        scale_pos_weight=1  # or compute for imbalanced classes
    )

    print(f"\n[PARAMS]")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - max_depth: {max_depth}")

    print(f"\n[TRAINING] Starting...")
    model.fit(X_train, y_train)
    print(f"[OK] Model trained")

    return model


def evaluate_classification(model, X_train, X_test, y_train, y_test):
    """Evaluate classifier performance"""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train metrics
    print(f"\n[TRAIN SET]")
    print(f"  - Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  - Precision: {precision_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
    print(f"  - Recall: {recall_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
    print(f"  - F1-Score: {f1_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")

    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    print(f"\n[TEST SET]")
    print(f"  - Accuracy: {test_accuracy:.4f}")
    print(f"  - Precision: {test_precision:.4f}")
    print(f"  - Recall: {test_recall:.4f}")
    print(f"  - F1-Score: {test_f1:.4f}")

    # Confusion matrix
    print(f"\n[CONFUSION MATRIX]")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    # Classification report
    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1
    }


def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation"""
    print("\n" + "="*70)
    print("CROSS-VALIDATION (5-Fold)")
    print("="*70)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

    print(f"\n[CV SCORES]")
    for fold, score in enumerate(scores, 1):
        print(f"  - Fold {fold}: {score:.4f}")

    print(f"\n[CV MEAN]: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return scores


def get_feature_importance_clf(model, feature_names, top_n=10):
    """Extract feature importance for classifier"""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n[TOP {top_n}] Features:")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}")

    return importances


def main():
    """Main pipeline"""
    print("\nCLASSIFICATION MODEL TRAINING PIPELINE")
    print("Risk Level Prediction (Low/Medium/High)")

    # Prepare data
    X, y, features, le_target = prepare_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n[SPLIT]")
    print(f"  - Train: {len(X_train):,} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
    print(f"  - Test: {len(X_test):,} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[SCALING] StandardScaler applied")

    # Train model
    model = train_classification_model(X_train_scaled, y_train)

    # Evaluate
    metrics = evaluate_classification(model, X_train_scaled, X_test_scaled, y_train, y_test)

    # Cross-validation
    cv_scores = cross_validate_model(model, X_train_scaled, y_train, cv=5)

    # Feature importance
    importances = get_feature_importance_clf(model, features, top_n=10)

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\n[SUMMARY]")
    print(f"  - Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"  - CV Mean F1: {cv_scores.mean():.4f}")
    print(f"\n")


if __name__ == "__main__":
    main()
