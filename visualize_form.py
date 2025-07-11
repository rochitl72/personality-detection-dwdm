# visualize_form.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_form_distributions(path='data/form_responses.csv'):
    # Load the form data
    df = pd.read_csv(path)

    # Drop timestamp/email columns
    df = df.drop(columns=[col for col in df.columns if "timestamp" in col.lower() or "email" in col.lower()])

    # Rename label column if needed
    label_col = None
    for col in df.columns:
        if "personality_type_user" in col.lower():
            label_col = col
            break

    if label_col is None:
        raise ValueError("❌ Label column with 'personality_type_user' not found in the form dataset.")

    df = df.rename(columns={label_col: 'personality_type_user'})

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # 🔹 Plot 1: Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='personality_type_user', data=df, palette='Set3')
    plt.title("Self-Identified Personality Distribution (Form)")
    plt.xlabel("Personality Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("output/form_class_distribution.png", dpi=300)
    print("✅ Saved: output/form_class_distribution.png")
    plt.clf()

    # 🔹 Plot 2: Correlation Heatmap
    df_encoded = df.copy()
    df_encoded['personality_type_user'] = df_encoded['personality_type_user'].map({
        'Introvert': 0,
        'Ambivert': 1,
        'Extrovert': 2
    })

    corr = df_encoded.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (Form Data)")
    plt.tight_layout()
    plt.savefig("output/form_heatmap.png", dpi=300)
    print("✅ Saved: output/form_heatmap.png")
    plt.clf()

if __name__ == "__main__":
    plot_form_distributions()
