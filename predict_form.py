# predict_form.py

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from preprocess_form import preprocess_form_data

# 🔹 Load the models
rf = joblib.load('models/random_forest_model.pkl')
lr = joblib.load('models/logistic_model.pkl')

# 🔹 Load and preprocess form data
X_form, y_form = preprocess_form_data()

# 🔹 Ensure output directory exists
os.makedirs("output", exist_ok=True)

# 🔹 Predict and evaluate with Random Forest
y_pred_rf = rf.predict(X_form)
report_rf = classification_report(y_form, y_pred_rf)
accuracy_rf = accuracy_score(y_form, y_pred_rf)

print("\n🔍 Comparing Random Forest Predictions with Self-Identified Labels:")
print(report_rf)
print(f"\n✅ Proposed Accuracy (Random Forest): {accuracy_rf * 100:.2f} %")

# 🔹 Save RF report
with open("output/form_predictions_rf.txt", "w") as f:
    f.write("🔹 Random Forest Report (Form Data)\n")
    f.write(report_rf)
    f.write(f"\nAccuracy: {accuracy_rf:.5f}\n")

# 🔹 Predict and evaluate with Logistic Regression
y_pred_lr = lr.predict(X_form)
report_lr = classification_report(y_form, y_pred_lr)
accuracy_lr = accuracy_score(y_form, y_pred_lr)

print("\n🔍 Comparing Logistic Regression Predictions with Self-Identified Labels:")
print(report_lr)
print(f"\n✅ Proposed Accuracy (Logistic Regression): {accuracy_lr * 100:.2f} %")

# 🔹 Save LR report
with open("output/form_predictions_lr.txt", "w") as f:
    f.write("🔹 Logistic Regression Report (Form Data)\n")
    f.write(report_lr)
    f.write(f"\nAccuracy: {accuracy_lr:.5f}\n")
