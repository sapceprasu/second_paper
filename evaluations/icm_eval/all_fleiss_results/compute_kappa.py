# Step 5: Compute Fleiss' Kappa Using the Validated Input Data

from statsmodels.stats.inter_rater import fleiss_kappa
import pandas as pd

# Reload the validated Fleiss' Kappa input dataset
df_fleiss_input = pd.read_csv("fleiss_kappa_input_data.csv")

# Initialize list for Fleiss' Kappa calculations
fleiss_results = []

# Iterate over each trait and properly structure the ratings matrix
for trait in traits:
    ratings = []
    
    for _, row in df_fleiss_input[df_fleiss_input["trait"] == trait].iterrows():
        # Collect ratings for all models (P1 and P2 combined)
        trait_values = [
            row["GPT-4o_P1"], row["GPT-4o_P2"],
            row["GPT-40-mini_P1"], row["GPT-40-mini_P2"],
            row["LLaMA_P1"], row["LLaMA_P2"],
            row["Qwen_P1"], row["Qwen_P2"]
        ]
        
        # Count occurrences of "High" and "Low"
        count_high = trait_values.count("High")
        count_low = trait_values.count("Low")

        # Ensure each row sums to total number of judges (4 models * 2 persons = 8)
        if count_high + count_low == len(trait_values):
            ratings.append([count_high, count_low])

    # Compute Fleiss' Kappa only if there are valid ratings
    if len(ratings) > 0:
        fleiss_kappa_score = fleiss_kappa(ratings, method='fleiss')
    else:
        fleiss_kappa_score = None

    # Store computed results
    fleiss_results.append({"Trait": trait, "Fleiss_Kappa": fleiss_kappa_score})

# Convert to DataFrame and Save Results
df_fleiss_results = pd.DataFrame(fleiss_results)
fleiss_results_csv = "fleiss_kappa_results_final.csv"
df_fleiss_results.to_csv(fleiss_results_csv, index=False)

print(f"✅ Fleiss' Kappa successfully computed and saved as '{fleiss_results_csv}'.")
