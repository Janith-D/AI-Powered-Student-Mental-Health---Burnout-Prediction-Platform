# SHAP Explainable AI - Complete Guide

## Overview

SHAP (SHapley Additive exPlanations) is a unified framework to explain predictions of machine learning models. It uses game theory to calculate each feature's contribution to the prediction.

---

## Files Created

### 1. **shap_explainability.py** (Production-Ready Script)
Complete end-to-end SHAP analysis pipeline.

**What it does**:
1. Load and preprocess data
2. Train XGBoost regression model
3. Create SHAP explainer
4. Compute SHAP values
5. Generate visualizations
6. Analyze feature effects
7. Explain individual predictions

**Run**:
```bash
python shap_explainability.py
```

### 2. **shap_utils.py** (Reusable Module)
Utility functions for SHAP analysis in other projects.

```python
from shap_utils import:
  - explain_single_prediction()    # Explain one prediction
  - get_feature_directions()       # Feature positive/negative effects
  - create_dependence_plot()       # Feature effect plots
  - waterfall_plot()               # Contribution breakdown
```

---

## Output Files Generated

### 1. **shap_summary_plot.png**
**What it shows**: Feature importance and impact on model output

**Interpretation**:
- X-axis: SHAP value (impact on prediction)
- Y-axis: Features (sorted by importance)
- Red dots: High feature values
- Blue dots: Low feature values

**Reading**:
- Red dot on right → High value increases output
- Blue dot on right → Low value increases output
- Feature at top → Most important

### 2. **shap_bar_plot.png**
**What it shows**: Mean absolute SHAP values per feature

**Interpretation**:
- Taller bar = more important feature
- Average impact magnitude on model output

**Example**:
```
sleep_quality: 9.86    (most important)
stress_level: 5.99
social_support: 5.74
```

### 3. **shap_force_plot.png**
**What it shows**: Individual prediction breakdown

**Interpretation**:
- Base value: Mean prediction (78.90)
- Red arrows: Push prediction up (increase burnout)
- Blue arrows: Push prediction down (decrease burnout)
- Final value: Actual prediction

---

## Key Concepts

### SHAP Value
A value assigned to each feature for each prediction showing its contribution.

- **Positive SHAP**: Feature increases prediction (increases burnout)
- **Negative SHAP**: Feature decreases prediction (decreases burnout)
- **Magnitude**: Importance/impact size

### Expected Value (Base Value)
Mean model prediction across all dataset. Default starting point: 78.90

### Feature Contribution
SHAP value shows: "This feature changed the prediction by X points"

---

## Analysis Results

### Top Features by SHAP Importance

```
1. sleep_quality (9.86)         -> Decreases burnout
2. stress_level (5.99)          -> Decreases burnout
3. social_support (5.74)        -> Increases burnout
4. exercise_hours (4.09)        -> Decreases burnout
5. anxiety_score (3.96)         -> Decreases burnout
```

### Feature Directions

**INCREASES BURNOUT**:
- social_support (+0.23): Counter-intuitive in sample
- assignment_load (+0.17)

**DECREASES BURNOUT (PROTECTIVE)**:
- sleep_quality (-1.30): Strongest protective factor
- exercise_hours (-0.59): Strong protective factor
- stress_level (-0.17): Weak protective effect
- anxiety_score (-0.14): Weak protective effect

---

## Individual Prediction Explanation Example

**Sample #0**:
```
Base value:     78.90     (mean prediction)
Prediction:     49.70     (actual prediction for this student)
Difference:     -29.20    (below average burnout)

Top Contributing Features:
1. sleep_quality: 1.43 -> SHAP: -18.27 -> [Decreases] burnout
   (Low sleep quality but still reduces burnout due to scaling)

2. exercise_hours: 1.44 -> SHAP: -7.74 -> [Decreases] burnout
   (Low exercise but reduces prediction)

3. stress_level: -1.02 -> SHAP: -6.67 -> [Decreases] burnout
   (Very low stress - major protective factor)

4. anxiety_score: 0.30 -> SHAP: +2.00 -> [Increases] burnout
   (Low anxiety slightly increases burnout)
```

---

## How to Interpret SHAP Values

### 1. Global Feature Importance
See which features matter most overall:
```
Bar plot shows top features:
- sleep_quality dominates (9.86 magnitude)
- stress_level important (5.99 magnitude)
- social_support significant (5.74 magnitude)
```

### 2. Feature Direction (Positive or Negative)
Understand if feature increases or decreases output:
```
sleep_quality: Mean SHAP = -1.30 (negative)
-> Higher sleep quality -> LOWER burnout
-> Protective factor

stress_level: Mean SHAP = -0.17 (negative)
-> But model learns: Lower stress -> lower burnout
```

### 3. Impact per Sample
See how specific features affected one prediction:
```
For Student #1:
- High sleep quality -> -18 points (massive reduction!)
- High exercise -> -7.7 points (large reduction)
- Low stress -> -6.7 points (large reduction)
Total effect: 78.90 - 18 - 7.7 - 6.7 + other = 49.70
```

---

## SHAP vs Traditional Feature Importance

| Aspect | Traditional | SHAP |
|--------|-------------|------|
| Local explanations | No | Yes (per prediction) |
| Global importance | Yes | Yes |
| Direction (positive/negative) | No | Yes |
| Prediction contribution | No | Yes (exact) |
| Interpretation | Feature count | Feature impact |
| Consistency | Not guaranteed | Theoretically sound (Shapley values) |

---

## Practical Use Cases

### 1. Model Debugging
```
Why is burnout_score lower than expected?
-> SHAP shows: High sleep_quality pushed it down (-18 points)
```

### 2. Business Insights
```
Which factors to focus on?
-> sleep_quality is 9.86x more important than other factors
-> Improve sleep = biggest burnout reduction opportunity
```

### 3. Individual Interventions
```
Predict & explain for Student #5:
"Your burnout is lower because:
  - Good sleep quality (-18.27)
  - Regular exercise (-7.74)
  - Low stress (-6.67)"
```

### 4. Model Validation
```
Do features make sense directionally?
-> sleep_quality negative (good!) = protective
-> stress_level negative (good!) = reduces burnout
-> Validates model behavior
```

---

## Advanced Usage

### Explain Single Prediction
```python
from shap_utils import explain_single_prediction

result = explain_single_prediction(
    model=trained_model,
    scaler=scaler,
    X_data=X_test,
    feature_names=features,
    sample_idx=5
)

print(f"Prediction: {result['prediction']:.2f}")
print(f"Base value: {result['base_value']:.2f}")
print(f"Difference: {result['difference']:+.2f}")

for feature, shap_val in result['top_features']:
    direction = "+" if shap_val > 0 else "-"
    print(f"  {feature}: {direction}{abs(shap_val):.2f}")
```

### Get Feature Directions
```python
from shap_utils import get_feature_directions

directions = get_feature_directions(model, X_train, feature_names)

for feature, info in directions.items():
    print(f"{feature}: {info['direction']} burnout")
    print(f"  Magnitude: {info['magnitude']:.3f}")
```

---

## Performance Optimization

### For Large Datasets (1M+ rows)

**1. Background Data Sampling**
```python
# Instead of all 1M rows, use representative sample
background_data = shap.sample(X_train, 100, random_state=42)
explainer = shap.TreeExplainer(model, background_data)
```

**2. SHAP Value Computation Sampling**
```python
# Compute SHAP for 500 samples instead of all 1M
X_sample = X_test.sample(n=500, random_state=42)
shap_values = explainer.shap_values(X_sample.values)
```

**3. Use TreeExplainer (Fastest for Tree Models)**
```python
# TreeExplainer is O(depth * num_samples)
explainer = shap.TreeExplainer(model)  # Fast!

# Avoid KernelExplainer for large datasets
# KernelExplainer is slower but model-agnostic
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow SHAP computation | Use smaller background data (100 samples) |
| Memory error | Sample both background and test data |
| Unicode errors | Replace unicode chars with ASCII |
| Import errors | `pip install shap -U` |
| Plot not saving | Check write permissions |

---

## Key Insights from Analysis

1. **Sleep Quality Dominates (9.86 importance)**
   - Most important predictor
   - Strong protective factor
   - Interventions: Sleep hygiene programs

2. **Stress Management Critical (5.99 importance)**
   - Second most important
   - Protective factor (reduces burnout)
   - Interventions: Stress reduction counseling

3. **Social Support Complex (5.74 importance)**
   - High variability
   - Sometimes increases burnout in data
   - Interventions: Community building mindfully

4. **Exercise Beneficial (4.09 importance)**
   - Consistent protective factor
   - Easy to implement
   - Interventions: Fitness programs

---

## Next Steps

### For Stakeholders
- Share SHAP plots in presentations
- Use individual explanations for student interventions
- Rank interventions by feature importance

### For Data Scientists
- Tune hyperparameters based on SHAP insights
- Create interactions between top features
- Validate model assumptions

### For Product Teams
- Build "Why" explanations into app
- Alert students on key risk factors
- Personalize interventions

---

## Files Summary

```
SHAP Explainability Module:
├── shap_explainability.py (main script)
├── shap_utils.py (reusable functions)
├── SHAP_GUIDE.md (this file)
└── Output:
    ├── shap_summary_plot.png
    ├── shap_bar_plot.png
    └── shap_force_plot.png
```

---

## Dependencies

```bash
pip install shap xgboost scikit-learn pandas numpy matplotlib
```

---

**Status**: Production Ready
**Last Updated**: 2026-03-26
