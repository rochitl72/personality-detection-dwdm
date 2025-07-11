# visualize_form_responses.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the form data
form_df = pd.read_csv("data/form_responses.csv")

# Create the output folder if it doesn't exist
output_dir = "outputs/form_response_charts"
os.makedirs(output_dir, exist_ok=True)

# Loop over each column (question)
for column in form_df.columns:
    plt.figure(figsize=(8, 5))

    # Value counts and plotting
    counts = form_df[column].value_counts().sort_index()
    counts.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title(f"Responses for: {column}", fontsize=12)
    plt.xlabel("Answer Options", fontsize=10)
    plt.ylabel("Number of Responses", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save each chart
    filename = os.path.join(output_dir, f"{column.replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close()

print(f"✅ Saved {len(form_df.columns)} charts to: {output_dir}")
