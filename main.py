import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. GENERATE SYNTHETIC MEDICAL DATA
def generate_health_data(samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, samples),
        'bmi': np.random.uniform(18, 45, samples),
        'glucose': np.random.randint(70, 250, samples),
        'bp_systolic': np.random.randint(90, 180, samples),
        'cholesterol': np.random.randint(150, 350, samples)
    }
    df = pd.DataFrame(data)
    # Define risk: combination of high glucose, age, and BMI
    risk_score = (df['glucose'] * 0.4) + (df['bmi'] * 0.3) + (df['age'] * 0.2)
    df['risk_label'] = (risk_score > 95).astype(int)
    return df

# 2. MODEL TRAINING
df = generate_health_data()
X = df.drop('risk_label', axis=1)
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training Medical AI Model (Gradient Boosting)...")
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 3. RESULTS
y_pred = model.predict(X_test)
print(f"\n✅ Diagnostic Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\n--- Clinical Classification Report ---")
print(classification_report(y_test, y_pred))
