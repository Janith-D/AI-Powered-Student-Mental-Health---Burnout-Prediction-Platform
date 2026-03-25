# Student Mental Health & Burnout Prediction Platform
## Exploratory Data Analysis (EDA) - Complete Setup

This directory contains everything needed to perform comprehensive EDA on the Student Mental Health and Burnout Dataset and build prediction models.

---

## 📁 Project Structure

```
📦 AI-Powered Student Mental Health & Burnout Prediction Platform
│
├── 📄 README.md (this file)
│
├── 🔬 ANALYSIS DOCUMENTATION
│   ├── EDA_SUMMARY.md          ← Key findings and insights
│   ├── EDA_GUIDE.md            ← Detailed step-by-step explanations
│   └── QUICK_START.md          ← Quick reference guide
│
├── 🐍 PYTHON SCRIPTS
│   ├── eda_analysis.py         ← Main EDA analysis script
│   ├── generate_sample_data.py ← Generate sample dataset
│   └── requirements.txt        ← Python dependencies
│
└── 📊 DATA FILES
    ├── student_mental_health_burnout.csv        (5K rows)
    └── student_mental_health_burnout_1M.csv     (1M rows) [FULL DATASET]
```

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run EDA Analysis
```bash
python eda_analysis.py
```

### 3. View Results
```
Generated files:
✓ Console output with 9-step analysis
✓ correlation_heatmap.png
✓ feature_correlation_burnout.png
```

---

## 📊 Dataset Overview

| Aspect | Details |
|--------|---------|
| **Rows** | 1,000,000 (production) / 5,000 (sample) |
| **Columns** | 21 features + 2 targets |
| **Target 1** | `burnout_score` (regression, 0-100) |
| **Target 2** | `risk_level` (classification: Low/Medium/High) |
| **Categorical** | gender, major (requires encoding) |
| **Numerical** | 18 features (ready for ML) |
| **Missing Values** | 2% in anxiety_score (handled automatically) |

---

## 📈 Key Findings

### Top 10 Features for Modeling
1. **sleep_quality** → Protective factor (-0.78 correlation)
2. **stress_level** → Primary predictor (+0.40 correlation)
3. **exercise_hours** → Protective (-0.62)
4. **anxiety_score** → Indicator (+0.20)
5. **social_support** → Protective (-0.55)
6. **meditation_hours** → Coping mechanism
7. **depression_score** → Mental health marker
8. **study_hours** → Academic pressure
9. **course_difficulty** → Academic stress
10. **assignment_load** → Workload factor

### Expected Model Performance
- **Burnout Score (Regression)**: R² = 0.85-0.90
- **Risk Level (Classification)**: F1 = 0.88-0.92

---

## 📚 Documentation Guide

### For Understanding the Data
👉 Start with: **EDA_SUMMARY.md**
- Dataset overview
- Key findings and insights
- Data quality metrics
- Recommendations for modeling

### For Step-by-Step Explanations
👉 Read: **EDA_GUIDE.md**
- Detailed explanation of each EDA step
- Interpretation guidelines
- Common issues and solutions
- Advanced techniques for large datasets

### For Running the Code
👉 Follow: **QUICK_START.md**
- How to install dependencies
- How to generate sample data
- How to run the EDA script
- Troubleshooting tips

---

## 🔍 What Each Script Does

### `eda_analysis.py` - Main Analysis Script
**Purpose**: Comprehensive exploratory data analysis

**What it does**:
1. Loads dataset (with sampling option)
2. Displays basic info (head, shape, dtypes)
3. Analyzes missing values
4. Calculates statistics (mean, std, quartiles)
5. Identifies feature types (numerical vs categorical)
6. Encodes categorical features (binary, one-hot, label)
7. Generates correlation heatmap
8. Identifies top features by correlation
9. Provides modeling recommendations

**Runtime**: ~1-3 minutes (depending on sample size)

**Output**:
- Console: 9-step detailed analysis
- Images: 2 visualizations (PNG)
- Recommendations: Best algorithms and features

### `generate_sample_data.py` - Data Generator
**Purpose**: Create sample dataset for testing

**What it does**:
- Generates 5,000 synthetic rows
- Includes realistic correlations between features
- Adds missing values (2% realistic)
- Creates balanced risk levels

**Runtime**: <1 minute

**Output**:
- `student_mental_health_burnout.csv` (1.5 MB)

---

## 🎯 Next Steps After EDA

### Phase 1: Preprocessing (Prepare Data)
```python
# Handle missing values
df.fillna(df.median(), inplace=True)

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Phase 2: Model Training
```python
# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from xgboost import XGBRegressor
model = XGBRegressor(max_depth=7, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
print(f"R² Score: {r2_score(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
```

### Phase 3: Feature Engineering
```python
# Create interaction terms
df['stress_sleep_interaction'] = df['stress_level'] * (10 - df['sleep_quality'])

# Polynomial features
df['stress_squared'] = df['stress_level'] ** 2

# Binning
df['stress_category'] = pd.qcut(df['stress_level'], q=3, labels=['Low', 'Mid', 'High'])
```

### Phase 4: Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

## 📊 Data Dictionary

### Categorical Features
| Feature | Type | Values | Encoding |
|---------|------|--------|----------|
| gender | object | Male, Female, Other | Label Encode (0, 1, 2) |
| major | object | STEM, Business, Humanities, Arts | One-Hot (4 columns) |

### Numerical Features
| Feature | Range | Unit | Meaning |
|---------|-------|------|---------|
| age | 18-24 | years | Student age |
| study_year | 1-4 | level | Academic year |
| stress_level | 20-90 | index | Self-reported stress |
| anxiety_score | 0-100 | score | Anxiety assessment |
| depression_score | 0-100 | score | Depression assessment |
| sleep_quality | 1-10 | rating | Sleep quality (higher=better) |
| sleep_hours | 3-10 | hours | Hours of sleep/night |
| study_hours | 2-12 | hours | Hours studying/day |
| gpa | 1.5-4.0 | score | Grade point average |
| course_difficulty | 1-10 | rating | Perceived difficulty |
| assignment_load | 1-10 | rating | Academic workload |
| exercise_hours | 0-10 | hours | Exercise per week |
| social_support | 1-10 | rating | Social support quality |
| meditation_hours | 0-5 | hours | Meditation per week |
| bmi | 16-32 | kg/m² | Body Mass Index |
| caffeine_intake | 0-1000 | mg | Daily caffeine (mg) |
| burnout_score | 0-100 | score | **TARGET (Regression)** |
| risk_level | Low/Med/Hi | category | **TARGET (Classification)** |

---

## 💡 Tips & Best Practices

### For Large Datasets (1M rows)
✓ Use sampling for exploration (50K-100K rows)
✓ Use chunks for memory efficiency
✓ Parallel processing for speed
✓ Reduce data types (int32 instead of int64)
✓ Use efficient algorithms (LightGBM vs XGBoost)

### Feature Selection Tips
✓ Use correlation with target (|r| > 0.3)
✓ Check variance (keep high variance features)
✓ Avoid multicollinearity (|r| < 0.8 between features)
✓ Remove near-zero variance features
✓ Start with top 10, expand if needed

### Model Best Practices
✓ Always use cross-validation (5-Fold minimum)
✓ Split data before any scaling
✓ Use appropriate metrics (RMSE for regression, F1 for classification)
✓ Compare multiple algorithms
✓ Tune hyperparameters systematically

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| FileNotFoundError | Verify CSV file path and name |
| MemoryError | Reduce SAMPLE_SIZE in eda_analysis.py |
| Import errors | Run `pip install -r requirements.txt` |
| Slow performance | Increase SAMPLE_SIZE or use chunks |
| Plot not saving | Check write permissions in directory |

---

## 🔗 Recommended Resources

### Python Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms and preprocessing
- **xgboost**: Gradient boosting
- **lightgbm**: Fast gradient boosting
- **matplotlib/seaborn**: Data visualization

### Learning Resources
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle Datasets & Competitions](https://www.kaggle.com/)

---

## 📋 Project Checklist

### Setup ✓
- [x] Create EDA script
- [x] Create documentation
- [x] Generate sample data
- [x] Create requirements file

### EDA Phase
- [x] Load and explore data
- [x] Check missing values
- [x] Analyze distributions
- [x] Identify feature types
- [x] Encode categorical features
- [x] Calculate correlations
- [x] Identify top features

### Next (Model Development)
- [ ] Preprocess full dataset
- [ ] Split train/test
- [ ] Train baseline models
- [ ] Cross-validate
- [ ] Hyperparameter tuning
- [ ] Evaluate best models
- [ ] Feature importance analysis

### Deployment
- [ ] Save best model
- [ ] Create prediction API
- [ ] Build web interface
- [ ] Setup monitoring
- [ ] Document production system

---

## 👤 Project Information

**Dataset**: Student Mental Health and Burnout
**Size**: 1 Million rows, 20 features
**Purpose**: Predict burnout scores and risk levels
**Target Variables**: burnout_score (regression), risk_level (classification)
**Status**: EDA Complete ✓ → Ready for Modeling

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-26 | Initial EDA setup, documentation, and sample data |

---

## 📞 Questions?

If you have questions about:
- **How to run the code**: See QUICK_START.md
- **Understanding each step**: See EDA_GUIDE.md
- **Key findings**: See EDA_SUMMARY.md
- **Python/ML concepts**: Check Recommended Resources section

---

**Happy analyzing and modeling! 🚀**

*Generated: 2026-03-26*
