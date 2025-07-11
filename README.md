# DWDM Project: Personality Type Detection using Data Mining Techniques

This is a personality classification system built using Python for a **Data Warehousing and Data Mining** project.

---

## 📦 Prerequisites

Make sure Python 3 is installed on your system.

Then open terminal in this folder and run:
```bash
pip install -r requirements.txt

## Dataset
- Source: Kaggle - Synthetic Personality Traits Dataset
- Size: 20,000 samples, 30 columns
- Target: personality_type (Introvert, Ambivert, Extrovert)

## Workflow
1. Load and preprocess data
2. Apply Logistic Regression & Random Forest
3. Evaluate accuracy and visualize results

## Output
- Accuracy reports
- Graphs in `output/` folder
🚀 How to Run the Project
1. Train the Model using Kaggle Dataset
bash
Copy
Edit
python model_kaggle.py
This will:

Preprocess the Kaggle dataset

Train two models (Logistic Regression and Random Forest)

Save them inside models/ folder

2. Predict on Real-World Google Form Data
bash
Copy
Edit
python predict_form.py
This will:

Load and preprocess form responses

Use saved model to predict each response

Save the output report in outputs/prediction_report.txt

3. Visualize the Survey Data (Bar Graphs)
bash
Copy
Edit
python visualize_form_responses.py
This will:

Create bar graphs for each question from the Google Form

Save them in outputs/form_response_charts/

📊 Project Flow
Kaggle Dataset → Preprocessing → Training → Saving Model

Google Form Responses → Preprocessing → Load Model → Predict

Bar Graphs → Heatmaps → Report Analysis

📁 Folder Structure
data/: Contains CSV files for training and testing

models/: Saved models (.pkl)

outputs/: Visualizations and prediction results

.py files: Scripts for training, testing, visualizing

README.md: This file

requirements.txt: All required packages

