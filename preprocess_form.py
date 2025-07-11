# preprocess_form.py

import pandas as pd

def preprocess_form_data(path='data/form_responses.csv'):
    df = pd.read_csv(path)

    # Drop timestamp and email columns if present
    cols_to_drop = [col for col in df.columns if "timestamp" in col.lower() or "email" in col.lower()]
    df = df.drop(columns=cols_to_drop)

    # Dynamically rename label column
    label_col = None
    for col in df.columns:
        if "personality_type_user" in col.lower():
            label_col = col
            break

    if label_col is None:
        raise ValueError("❌ Could not find 'personality_type_user' column.")

    df = df.rename(columns={label_col: 'personality_type_user'})

    # Encode labels
    df['personality_type_user'] = df['personality_type_user'].map({
        'Introvert': 0,
        'Ambivert': 1,
        'Extrovert': 2
    })

    X = df.drop(columns=['personality_type_user'])
    y = df['personality_type_user']
    return X.values, y.values
