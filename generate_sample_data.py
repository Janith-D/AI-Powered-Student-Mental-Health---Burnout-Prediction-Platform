"""Generate sample Student Mental Health and Burnout dataset"""

import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 5000

print("[GENERATING] Sample dataset: 5,000 rows, 20 features")

data = {
    'student_id': np.arange(1, n_samples + 1),
    'age': np.random.randint(18, 25, n_samples),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
    'study_year': np.random.choice([1, 2, 3, 4], n_samples),
    'major': np.random.choice(['STEM', 'Business', 'Humanities', 'Arts'], n_samples),
    'stress_level': np.random.uniform(20, 90, n_samples),
    'anxiety_score': np.random.uniform(0, 100, n_samples),
    'depression_score': np.random.uniform(0, 100, n_samples),
    'sleep_quality': np.random.uniform(1, 10, n_samples),
    'sleep_hours': np.random.uniform(3, 10, n_samples),
    'study_hours': np.random.uniform(2, 12, n_samples),
    'gpa': np.random.uniform(1.5, 4.0, n_samples),
    'course_difficulty': np.random.uniform(1, 10, n_samples),
    'assignment_load': np.random.uniform(1, 10, n_samples),
    'exercise_hours': np.random.uniform(0, 10, n_samples),
    'social_support': np.random.uniform(1, 10, n_samples),
    'meditation_hours': np.random.uniform(0, 5, n_samples),
    'bmi': np.random.uniform(16, 32, n_samples),
    'caffeine_intake': np.random.uniform(0, 1000, n_samples),
}

df = pd.DataFrame(data)

# Add realistic correlations
df['burnout_score'] = (
    df['stress_level'] * 0.4 +
    df['anxiety_score'] * 0.2 +
    df['study_hours'] * 0.15 +
    (10 - df['sleep_quality']) * 5 +
    (10 - df['exercise_hours']) * 2 +
    (10 - df['social_support']) * 3 +
    np.random.normal(0, 5, n_samples)
).clip(0, 100)

df['risk_level'] = pd.cut(df['burnout_score'],
                           bins=[0, 35, 65, 100],
                           labels=['Low', 'Medium', 'High'])

# Add some missing values (realistic)
missing_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
df.loc[missing_indices, 'anxiety_score'] = np.nan

df.to_csv('student_mental_health_burnout.csv', index=False)
print("[OK] Dataset saved: student_mental_health_burnout.csv")
print(f"[INFO] Shape: {df.shape}")
print(f"[INFO] Missing: {df.isnull().sum().sum()} values total")
