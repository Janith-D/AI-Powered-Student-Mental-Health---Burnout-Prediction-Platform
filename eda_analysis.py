"""
Exploratory Data Analysis (EDA) for Student Mental Health and Burnout Dataset
Optimized for large datasets (1M rows, 20 features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "student_mental_health_burnout.csv"
SAMPLE_SIZE = 50000
RANDOM_STATE = 42

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(path, use_sample=True, sample_size=SAMPLE_SIZE):
    """Load dataset with optional sampling"""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    try:
        if use_sample:
            total_rows = sum(1 for _ in open(path)) - 1
            skip = np.random.choice(np.arange(1, total_rows + 1),
                                   total_rows - sample_size,
                                   replace=False)
            df = pd.read_csv(path, skiprows=skip)
            print(f"[OK] Dataset loaded with sampling: {len(df):,} rows (total: {total_rows:,})")
        else:
            df = pd.read_csv(path)
            print(f"[OK] Full dataset loaded: {len(df):,} rows")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File '{path}' not found")
        return None


def basic_exploration(df):
    """Display basic dataset information"""
    print("\n" + "=" * 70)
    print("STEP 2: BASIC DATASET INFORMATION")
    print("=" * 70)

    print("\n[DATA] First 5 rows:")
    print(df.head())

    print(f"\n[SHAPE] Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n[INFO] Data Types & Missing Values:")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Missing': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    print(info_df.to_string(index=False))


def check_missing_values(df):
    """Analyze missing values"""
    print("\n" + "=" * 70)
    print("STEP 3: MISSING VALUES ANALYSIS")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing Count', ascending=False)

    if missing_df['Missing Count'].sum() == 0:
        print("[OK] No missing values found!")
    else:
        print("\n[WARNING] Missing Values Summary:")
        print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))

    return missing_df


def statistical_summary(df):
    """Display statistical summary"""
    print("\n" + "=" * 70)
    print("STEP 4: STATISTICAL SUMMARY")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n[STATS] Statistical Summary ({len(numeric_cols)} numerical features):")
    print(df[numeric_cols].describe().round(3))


def identify_features(df):
    """Identify and categorize features"""
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE IDENTIFICATION")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"\n[NUMERIC] Features ({len(numeric_cols)}): {numeric_cols}")
    print(f"[CATEGORICAL] Features ({len(categorical_cols)}): {categorical_cols}")

    return numeric_cols, categorical_cols


def encode_categorical(df, categorical_cols):
    """Encode categorical features"""
    print("\n" + "=" * 70)
    print("STEP 6: CATEGORICAL ENCODING")
    print("=" * 70)

    df_encoded = df.copy()
    encoding_mapping = {}

    for col in categorical_cols:
        unique_values = df_encoded[col].nunique()

        if unique_values <= 2:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoding_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"[ENCODE] {col}: Binary Encoded -> {encoding_mapping[col]}")
        else:
            if unique_values <= 10:
                df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
                print(f"[ENCODE] {col}: One-Hot Encoded ({unique_values} categories)")
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                print(f"[ENCODE] {col}: Label Encoded ({unique_values} categories)")

    print(f"\n[SHAPE] Encoded dataset: {df_encoded.shape}")

    return df_encoded, encoding_mapping


def correlation_analysis(df_encoded, numeric_cols):
    """Generate correlation analysis"""
    print("\n" + "=" * 70)
    print("STEP 7: CORRELATION ANALYSIS")
    print("=" * 70)

    numeric_data = df_encoded.select_dtypes(include=[np.number])

    if 'burnout_score' in numeric_data.columns:
        burnout_corr = numeric_data.corr()['burnout_score'].sort_values(ascending=False)
        print("\n[CORR] Correlation with burnout_score:")
        print(burnout_corr)

    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap - Student Mental Health Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n[SAVED] Correlation heatmap saved: correlation_heatmap.png")

    return correlation_matrix


def feature_importance(df_encoded, numeric_cols):
    """Identify important features"""
    print("\n" + "=" * 70)
    print("STEP 8: FEATURE IMPORTANCE")
    print("=" * 70)

    numeric_data = df_encoded.select_dtypes(include=[np.number])
    feature_variance = numeric_data.var().sort_values(ascending=False)

    print("\n[VARIANCE] Top features by variance:")
    print(feature_variance.head(15))

    if 'burnout_score' in numeric_data.columns:
        burnout_corr = numeric_data.corr()['burnout_score'].abs().sort_values(ascending=False)

        print("\n[TOP 10] Features Correlated with burnout_score:")
        top_features = burnout_corr[1:11]
        print(top_features)

        plt.figure(figsize=(10, 6))
        top_features.plot(kind='barh', color='steelblue')
        plt.xlabel('Absolute Correlation', fontweight='bold')
        plt.title('Top 10 Features Correlated with Burnout Score', fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_correlation_burnout.png', dpi=300, bbox_inches='tight')
        print("[SAVED] Feature correlation plot: feature_correlation_burnout.png")

        return top_features.index[0:10].tolist()

    return None


def generate_recommendations(df_encoded, important_features):
    """Generate modeling recommendations"""
    print("\n" + "=" * 70)
    print("STEP 9: MODELING RECOMMENDATIONS")
    print("=" * 70)

    print("""
[REGRESSION] BURNOUT SCORE PREDICTION:
  - Algorithm: XGBoost, LightGBM, Random Forest
  - Metric: RMSE, MAE, R2 Score
  - CV: 5-Fold

[CLASSIFICATION] RISK LEVEL:
  - Algorithm: XGBoost Classifier, LightGBM, Ensemble
  - Metric: F1-Score, AUC-ROC, Precision-Recall
  - Handling: class_weight or SMOTE for imbalance

[TOP FEATURES] FOR MODELING:
""")

    if important_features:
        for i, feature in enumerate(important_features, 1):
            print(f"  {i}. {feature}")


def main():
    """Main EDA pipeline"""
    print("\n[START] Exploratory Data Analysis - Student Mental Health Dataset\n")

    df = load_data(DATASET_PATH, use_sample=True, sample_size=SAMPLE_SIZE)
    if df is None:
        return

    basic_exploration(df)
    check_missing_values(df)
    statistical_summary(df)
    numeric_cols, categorical_cols = identify_features(df)
    df_encoded, encoding_mapping = encode_categorical(df, categorical_cols)
    correlation_matrix = correlation_analysis(df_encoded, numeric_cols)
    important_features = feature_importance(df_encoded, numeric_cols)
    generate_recommendations(df_encoded, important_features)

    print("\n" + "=" * 70)
    print("[DONE] EDA COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n[FILES] Generated:")
    print("  - correlation_heatmap.png")
    print("  - feature_correlation_burnout.png\n")


if __name__ == "__main__":
    main()
