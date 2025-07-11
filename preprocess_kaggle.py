# preprocess_kaggle.py

def preprocess_data(df):
    df['personality_type'] = df['personality_type'].map({
        'Introvert': 0,
        'Ambivert': 1,
        'Extrovert': 2
    })

    X = df.drop(columns=['personality_type'])
    y = df['personality_type']
    return X, y
