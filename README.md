# Personality Type Detection using Data Mining Techniques

## 🧠 Project Overview
This project classifies individuals into personality types — **Introvert**, **Ambivert**, or **Extrovert** — using machine learning. We trained models using a clean Kaggle dataset of 20,000 samples and validated them on real-world responses collected via a Google Form. This was done as part of the **Data Warehousing and Data Mining** course under the **Data Analytics Specialization**.

---

## 🗃️ Folder Structure
.
├── .DS_Store
├── .gitignore
├── README.md
├── data
│   ├── form_responses.csv
│   └── personality_kaggle.csv
├── folder_structure.txt
├── model_kaggle.py
├── models
│   ├── logistic_model.pkl
│   └── random_forest_model.pkl
├── output
│   ├── class_distribution.png
│   ├── form_class_distribution.png
│   ├── form_heatmap.png
│   ├── form_predictions_lr.txt
│   ├── form_predictions_rf.txt
│   ├── heatmap.png
│   ├── logistic_report.txt
│   └── random_forest_report.txt
├── outputs
│   ├── .DS_Store
│   └── form_response_charts
│       ├── How_adventurous_are_you?_(adventurousness).png
│       ├── How_calm_and_emotionally_stable_are_you_under_stress?_(emotional_stability).png
│       ├── How_comfortable_are_you_in_group_settings?_(group_comfort).png
│       ├── How_comfortable_are_you_spending_time_alone?_(alone_time_preference).png
│       ├── How_comfortable_are_you_taking_on_leadership_roles?_(leadership).png
│       ├── How_comfortable_are_you_with_public_speaking?_(public_speaking_comfort).png
│       ├── How_creative_do_you_consider_yourself?_(creativity).png
│       ├── How_curious_are_you_to_explore_new_ideas_or_topics?_(curiosity).png
│       ├── How_frequently_do_you_use_gadgets_or_tech_devices?_(gadget_usage).png
│       ├── How_friendly_and_approachable_are_you?_(friendliness).png
│       ├── How_good_are_your_active_listening_skills?_(listening_skill).png
│       ├── How_interested_are_you_in_sports_or_physical_activity?_(sports_interest).png
│       ├── How_much_do_you_enjoy_parties_or_social_events?_(party_liking).png
│       ├── How_much_do_you_plan_ahead_before_making_decisions?_(planning).png
│       ├── How_much_do_you_prefer_a_routine-based_lifestyle?_(routine_preference).png
│       ├── How_much_do_you_prefer_working_in_teams_over_working_alone?_(work_style_collaborative).png
│       ├── How_much_energy_do_you_gain_from_social_interactions?_(social_energy).png
│       ├── How_much_time_do_you_spend_on_social_media_or_online_platforms?_(online_social_usage).png
│       ├── How_often_do_you_act_on_impulse?_(spontaneity).png
│       ├── How_often_do_you_engage_in_deep_or_introspective_thinking?_(deep_reflection).png
│       ├── How_often_do_you_read_books_or_articles?_(reading_habit).png
│       ├── How_often_do_you_seek_new_and_exciting_experiences?_(excitement_seeking).png
│       ├── How_organized_are_you_in_your_daily_life?_(organization).png
│       ├── How_quickly_do_you_usually_make_decisions?_(decision_speed).png
│       ├── How_strong_is_your_desire_to_travel_and_explore_new_places?_(travel_desire).png
│       ├── How_talkative_are_you_in_general?_(talkativeness).png
│       ├── How_well_can_you_understand_others'_emotions?_(empathy).png
│       ├── How_well_do_you_manage_stress_in_your_life?_(stress_handling).png
│       ├── How_willing_are_you_to_take_risks?_(risk_taking).png
│       ├── How_would_you_describe_your_personality_type?_(personality_type_user).png
│       └── Timestamp.png
├── predict_form.py
├── preprocess_form.py
├── preprocess_kaggle.py
├── requirements.txt
├── utils.py
├── visualize_form.py
├── visualize_form_responses.py
└── visualize_kaggle.py

6 directories, 57 files


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


