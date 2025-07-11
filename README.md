# Personality Type Detection using Data Mining Techniques

## 🧠 Project Overview
This project classifies individuals into personality types — **Introvert**, **Ambivert**, or **Extrovert** — using machine learning. We trained models using a clean Kaggle dataset of 20,000 samples and validated them on real-world responses collected via a Google Form. This was done as part of the **Data Warehousing and Data Mining** course under the **Data Analytics Specialization**.

---

## 🗃️ Folder Structure

personality_dwdm_project/
│
├── data/ # Contains raw CSV datasets
│ ├── personality_kaggle.csv
│ └── form_responses.csv
│
├── models/ # Saved ML models
│ ├── logistic_model.pkl
│ └── random_forest_model.pkl
│
├── output/ # Model evaluation reports and heatmaps
├── outputs/ # Google Form bar charts per question
│ └── form_response_charts/
│
├── preprocess_kaggle.py # Preprocessing function for Kaggle dataset
├── preprocess_form.py # Preprocessing function for Form dataset
├── model_kaggle.py # Training models on Kaggle dataset
├── predict_form.py # Predicting personality types from form
├── visualize_kaggle.py # Heatmap and charts for Kaggle
├── visualize_form_responses.py # Bar graphs for form responses
├── utils.py # Helper functions
├── requirements.txt # All required libraries
└── README.md # This file

yaml
Copy
Edit

---

## 🔧 Setup Instructions

Follow these steps to run the project on any machine:

### ✅ 1. Create Virtual Environment (optional but recommended)

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
✅ 2. Install Required Packages

pip install -r requirements.txt
✅ 3. Train the Model using Kaggle Dataset

python model_kaggle.py
This trains Logistic Regression and Random Forest on the Kaggle dataset.

Saves models in models/

Generates reports in output/

✅ 4. Predict Using Google Form Responses

python predict_form.py
Predicts personality types using form_responses.csv

Uses saved models to test on real-world data

Outputs prediction reports and accuracy

✅ 5. Visualize Responses

python visualize_form_responses.py
Generates bar graphs for each form question in outputs/form_response_charts/

📊 Key Metrics
Kaggle Dataset: 20,000 samples, 29 attributes, balanced across all 3 personality types

Form Dataset: 64 real-world entries, 30 personality questions

📉 Results
Model	Kaggle Dataset	Form Responses
Logistic Regression	~99.6%	~71.87%
Random Forest	~99.4%	~70.31%

❓ Why Accuracy Dropped on Google Form Data
Kaggle data is clean and balanced, with no noise.

Google Form responses are fewer and contain real-world inconsistencies and human bias.

This reflects the gap between controlled training and real-world deployment.

📌 What We Learned
Importance of data preprocessing and cleaning

Difference between training accuracy and generalization accuracy

Practical application of data mining techniques

Usage of classification models and model serialization

✅ Technologies Used
Python 3.11+

Pandas, Scikit-learn, Matplotlib

Joblib for saving models

🏁 Final Notes
This project fulfills the Data Warehousing and Data Mining course deliverables by:

Applying warehouse-style structured storage using CSVs

Mining patterns using classification algorithms

Performing real-world validation using unseen data

Visualizing and interpreting results through charts and heatmaps


