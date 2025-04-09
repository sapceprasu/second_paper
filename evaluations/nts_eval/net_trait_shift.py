import json
import pandas as pd

# Load dataset
with open("/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_gpt4o_gptvsgpt.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# Define personality traits
traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# Initialize storage for Net Trait Shift (NTS) calculations
nts_results = []

# Process each topic and calculate NTS
for topic in data:
    assigned_traits = topic.get("traits", {})
    analysis = topic.get("analysis")  # Do not use `.get()` here directly

    # Skip if analysis is missing
    if analysis is None:
        print(f"⚠️ Missing analysis field in topic '{topic['topic']}', skipping...")
        continue

    for trait in traits:
        # Check if assigned trait exists
        if trait not in assigned_traits:
            print(f"⚠️ Missing assigned trait '{trait}' in topic '{topic['topic']}', skipping...")
            continue

        base_value = 1 if assigned_traits[trait] == "High" else -1

        # Extract predictions safely
        person_one_data = analysis.get("Person_One", {})
        person_two_data = analysis.get("Person_Two", {})

        pred_p1 = person_one_data.get("predicted_bfi", {}).get(trait)
        pred_p2 = person_two_data.get("predicted_bfi", {}).get(trait)

        # Handle missing predictions
        if pred_p1 is None or pred_p2 is None:
            print(f"⚠️ Missing prediction for trait '{trait}' in topic '{topic['topic']}', skipping...")
            continue

        pred_p1 = 1 if pred_p1 == "High" else -1
        pred_p2 = 1 if pred_p2 == "High" else -1

        # Compute Net Trait Shift (NTS)
        nts_p1 = pred_p1 - base_value
        nts_p2 = pred_p2 - base_value

        # Store results
        nts_results.append([topic["topic"], trait, base_value, pred_p1, nts_p1, pred_p2, nts_p2])

# Convert to DataFrame
df_nts = pd.DataFrame(nts_results, columns=["Topic", "Trait", "Base_Value", "Predicted_P1", "NTS_P1", "Predicted_P2", "NTS_P2"])

df_nts.to_csv("nts_gpt_eval_gptvgptresults.csv", index=False)
print("✅ Net Trait Shift (NTS) results saved as 'net_trait_shift_results.csv'")
