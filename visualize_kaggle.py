# visualize_kaggle.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_distributions(path='data/personality_kaggle.csv'):
    # Load dataset
    df = pd.read_csv(path)

    # Ensure output folder exists
    os.makedirs("output", exist_ok=True)

    # 🔹 Plot 1: Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='personality_type', data=df, palette='Set2')
    plt.title("Distribution of Personality Types")
    plt.xlabel("Personality Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("output/class_distribution.png", dpi=300)
    print("✅ Saved: output/class_distribution.png")
    plt.clf()

    # 🔹 Plot 2: Correlation Heatmap
    df_encoded = df.copy()
    df_encoded['personality_type'] = df_encoded['personality_type'].map({
        'Introvert': 0,
        'Ambivert': 1,
        'Extrovert': 2
    })
    corr = df_encoded.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("output/heatmap.png", dpi=300)
    print("✅ Saved: output/heatmap.png")
    plt.clf()

if __name__ == "__main__":
    plot_distributions()
