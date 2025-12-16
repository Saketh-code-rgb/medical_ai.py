import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. GENERATE DATA
def generate_health_data(samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 80, samples),
        'BMI': np.random.uniform(18, 45, samples),
        'Glucose': np.random.randint(70, 250, samples),
        'Blood_Pressure': np.random.randint(90, 180, samples),
        'Cholesterol': np.random.randint(150, 350, samples)
    }
    df = pd.DataFrame(data)
    risk_score = (df['Glucose'] * 0.4) + (df['BMI'] * 0.3) + (df['Age'] * 0.2)
    df['Risk_Label'] = (risk_score > 95).astype(int)
    return df

# 2. TRAIN MODEL
df = generate_health_data()
X = df.drop('Risk_Label', axis=1)
y = df['Risk_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 3. NEW: VISUALIZE FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("--- Feature Importance ---")
print(feature_importance_df)

# To see the accuracy
y_pred = model.predict(X_test)
print(f"\n✅ Diagnostic Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
