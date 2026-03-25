# Quick Start Guide - EDA Analysis

## 🚀 Getting Started in 3 Steps

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Generate Sample Dataset (Optional)**
If you don't have your dataset yet, generate a sample:
```bash
python generate_sample_data.py
```
This creates `student_mental_health_burnout.csv` with 5,000 synthetic rows for testing.

### **Step 3: Run EDA Analysis**
```bash
python eda_analysis.py
```

---

## 📊 Output

The script will generate:

1. **Console Output**: Detailed statistics in 9 steps
   ```
   ✓ Dataset loaded with sampling: 50,000 rows
   ✓ First 5 rows displayed
   ✓ ... (all 9 steps with insights)
   ```

2. **Visualizations**:
   - `correlation_heatmap.png` - Feature correlation matrix
   - `feature_correlation_burnout.png` - Top features bar chart

3. **Feature Recommendations**:
   - Top 10 features for modeling
   - Best algorithms to use
   - Data preparation tips

---

## 🔧 Using Your Own Dataset

### **Option 1: Replace the Default Dataset**
```python
# In eda_analysis.py, line ~15
DATASET_PATH = "path/to/your/student_mental_health_burnout.csv"
```

### **Option 2: Specify Dataset Path**
```bash
# Modify the script and run
python eda_analysis.py
```

### **Requirements for Your Dataset:**
- Format: CSV file
- Columns: Should include `burnout_score`, `risk_level`, and other features
- At least 1 numerical target variable
- Can have categorical variables (auto-encoded)

---

## ⚙️ Configuration Options

Edit these variables in `eda_analysis.py` to customize the analysis:

```python
# Line 16-18
DATASET_PATH = "student_mental_health_burnout.csv"
SAMPLE_SIZE = 50000  # Increase for more detailed analysis
RANDOM_STATE = 42    # For reproducibility
```

### **When to Increase SAMPLE_SIZE:**
- More data = slower but more accurate insights
- For 1M row dataset: 50K-100K is reasonable
- For final modeling: Use full dataset

---

## 📈 Understanding Key Outputs

### **1. Correlation Heatmap**
- Red = positive correlation (features move together)
- Blue = negative correlation (inverse relationship)
- White = no correlation

**What to look for:**
- Dark red/blue near feature columns = important predictors
- High correlation between features = potential multicollinearity

### **2. Feature Correlation Chart**
- Shows top 10 features by correlation with burnout_score
- Use these features for your prediction models
- Example:
  ```
  study_hours: 0.85      ← Strongest predictor
  sleep_quality: -0.78    ← Strong protective factor
  stress_level: 0.72      ← Second strongest predictor
  ```

### **3. Feature Recommendations**
Lists best practices for:
- Burnout score prediction (regression)
- Risk level classification
- Feature engineering ideas
- Data preprocessing steps

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` | Check DATASET_PATH spelling and location |
| `MemoryError` | Reduce SAMPLE_SIZE or use full dataset with chunking |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Plots not displaying | Ensure matplotlib backend is set (usually automatic) |
| Encoding errors | Data file might use different encoding; check CSV file |

---

## 📋 EDA Checklist

After running the script, verify you have:

- [ ] Dataset shape and structure understood
- [ ] Missing values identified and strategy planned
- [ ] Data types verified (numerical vs categorical)
- [ ] Features encoded for ML algorithms
- [ ] Correlation heatmap generated
- [ ] Top 10 important features identified
- [ ] Visualizations saved
- [ ] Modeling recommendations reviewed

---

## 🎯 Next Steps After EDA

1. **Data Cleaning**
   ```python
   # Handle missing values
   df.fillna(df.median(), inplace=True)

   # Remove outliers (IQR method)
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   df = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]
   ```

2. **Feature Scaling**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   df_scaled = scaler.fit_transform(df_encoded)
   ```

3. **Train-Test Split**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

4. **Model Training** (example)
   ```python
   from xgboost import XGBRegressor
   model = XGBRegressor()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   ```

---

## 📚 Files Generated

```
📁 Project Directory
├── eda_analysis.py                           # Main EDA script
├── generate_sample_data.py                   # Generate synthetic data
├── requirements.txt                          # Python dependencies
├── EDA_GUIDE.md                              # Detailed explanation
├── QUICK_START.md                            # This file
├── student_mental_health_burnout.csv         # Your dataset (CSV)
├── correlation_heatmap.png                   # Generated visualization
└── feature_correlation_burnout.png           # Generated visualization
```

---

## 💡 Tips for Large Datasets (1M+ rows)

### **Efficient Sampling**
```python
# Random stratified sampling by risk_level
df_sample = df.groupby('risk_level', group_keys=False).apply(
    lambda x: x.sample(frac=0.05)  # 5% sample
)
```

### **Memory-Efficient Processing**
```python
# Process in chunks
chunk_size = 100000
for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### **Parallel Processing**
```python
# Use all CPU cores
df_processed = df.groupby('category').parallel_apply(function)
```

---

## ❓ FAQ

**Q: Should I use sampling or full dataset?**
A: Use sampling (50K-100K) for initial EDA to save time. Use full dataset for final model training.

**Q: How long does EDA take?**
A: ~1-5 minutes depending on sample size and your computer.

**Q: What if my dataset has different column names?**
A: The script auto-detects numerical/categorical features. It will work with any column names.

**Q: Can I run this on production data?**
A: Yes! Just update DATASET_PATH and ensure sensitive data is handled properly.

---

## 📞 Support

For issues with the script:
1. Check the EDA_GUIDE.md for detailed explanations
2. Verify your dataset format (CSV, proper encoding)
3. Ensure all dependencies are installed: `pip list`
4. Check console output for specific error messages

---

**Happy analyzing! 🎉**
