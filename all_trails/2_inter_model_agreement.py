import pandas as pd
import itertools
import json

# Load the mapped dataset
df = pd.read_csv("/media/data_4tb/pranav/llms_gen_eval/evaluations/to_be_evaluated/updated_mapped_model_data.csv")

# List of models and personality traits
models = ["GPT-4o", "LLaMA3.2", "LLaMA3.3", "Qwen", "DeepSeek"]
traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# Initialize storage for Inter-Model Agreement (IMA) results
ima_results = []

# Compute IMA for each unique discourse-personality instance
for _, row in df.iterrows():
    unique_id = row["unique_id"]  # Ensure we compare within the same personality setting
    topic = row["topic"]
    ima_scores = {}

    for trait in traits:
        agreements = 0
        total_comparisons = 0

        # Collect trait values for Person One and Person Two across models
        trait_values_p1 = {model: row[f"P1_{trait}"] for model in models if f"P1_{trait}" in row}
        trait_values_p2 = {model: row[f"P2_{trait}"] for model in models if f"P2_{trait}" in row}

        # Compare models' assessments using pairwise agreement for Person One and Person Two
        for values in [trait_values_p1, trait_values_p2]:
            for (m1, m2) in itertools.combinations(values.keys(), 2):
                if values[m1] == values[m2]:
                    agreements += 1
                total_comparisons += 1

        # Compute final agreement score for the trait
        ima_scores[trait] = round(agreements / total_comparisons * 100, 2) if total_comparisons > 0 else None

    # Store results for this unique discourse-personality setting
    ima_results.append({"unique_id": unique_id, "topic": topic, "ima_scores": ima_scores})

    print(f"✅ Processed unique instance: {unique_id}")

# Save results as JSON
output_file = "inter_model_agreement.json"
with open(output_file, "w") as f:
    json.dump(ima_results, f, indent=4)

print(f"✅ IMA results saved to {output_file}")
