import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed CSV file with predicted and original traits
file_path = "bfi_from_liwc_with_original_traits.csv"
df = pd.read_csv(file_path)

# List of personality traits
traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# Separate Person 1 and Person 2 data
df_person_1 = df[df["Speaker"] == "Person_One"]
df_person_2 = df[df["Speaker"] == "Person_Two"]

# Initialize accuracy storage
accuracy_results = {}

# Compute accuracy correctly for each trait
for trait in traits:
    df_person_1[f"Match_{trait}"] = df_person_1[f"Predicted_{trait}"] == df_person_1[f"Original_{trait}"]
    df_person_2[f"Match_{trait}"] = df_person_2[f"Predicted_{trait}"] == df_person_2[f"Original_{trait}"]

    # Compute accuracy as a percentage
    accuracy_results[trait] = {
        "Person_One_Accuracy": df_person_1[f"Match_{trait}"].mean() * 100,
        "Person_Two_Accuracy": df_person_2[f"Match_{trait}"].mean() * 100
    }

# Convert accuracy results to a DataFrame
accuracy_df = pd.DataFrame(accuracy_results).T

# Save accuracy results to CSV
accuracy_df.to_csv("trait_accuracy_results_fixed.csv")

# Display accuracy results
print("✅ Fixed Accuracy Calculation Completed!\n")
print(accuracy_df)

# =======================
# Visualization - Bar Chart
# =======================

plt.figure(figsize=(10, 6))
accuracy_df.plot(kind="bar", figsize=(12, 6), color=["royalblue", "orange"])
plt.title("Corrected Trait Prediction Accuracy for Person 1 & 2")
plt.ylabel("Accuracy (%)")
plt.xlabel("Personality Traits")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.legend(["Person 1", "Person 2"], loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# =======================
# Visualization - Heatmap
# =======================

plt.figure(figsize=(8, 6))
sns.heatmap(accuracy_df, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5)
plt.title("Corrected Accuracy Heatmap of Trait Predictions")
plt.xlabel("Prediction Accuracy (%)")
plt.ylabel("Personality Traits")
plt.show()
