# model_kaggle.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess_kaggle import preprocess_data

# 🔹 Create folders if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

# 🔹 Load and preprocess Kaggle data
df = pd.read_csv("data/personality_kaggle.csv")
X, y = preprocess_data(df)

# 🔹 Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Logistic Regression
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_report = classification_report(y_test, y_pred_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# 🔹 Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_report = classification_report(y_test, y_pred_rf)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# 🔹 Print & Save Reports
print("🔹 Logistic Regression Report:")
print(lr_report)
print("Accuracy:", lr_accuracy)

print("\n🔹 Random Forest Report:")
print(rf_report)
print("Accuracy:", rf_accuracy)

with open("output/logistic_report.txt", "w") as f:
    f.write("🔹 Logistic Regression Report\n")
    f.write(lr_report)
    f.write(f"\nAccuracy: {lr_accuracy:.5f}\n")

with open("output/random_forest_report.txt", "w") as f:
    f.write("🔹 Random Forest Report\n")
    f.write(rf_report)
    f.write(f"\nAccuracy: {rf_accuracy:.5f}\n")

# 🔹 Save trained models
joblib.dump(lr, 'models/logistic_model.pkl')
joblib.dump(rf, 'models/random_forest_model.pkl')

print("\n✅ Models and reports saved.")
