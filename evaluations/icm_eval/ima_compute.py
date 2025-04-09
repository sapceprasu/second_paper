# # Step 1: Load and Align Data for Each Judge Model (Fixing Sparse Storage Issue)

# import json
# import pandas as pd

# # # Define paths for the four judge models
# # model_files = {
# #     "GPT-4o": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_gpt4o_gptvsgpt.json",
# #     "GPT-40-mini": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_gpt4omini_gptvsgpt.json",
# #     "LLaMA": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_llama_gptvsgpt.json",
# #     "Qwen": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_qwen_act_gptvsgpt.json"
# # }

# # # Define paths for the four judge models
# model_files = {
#     "GPT-4o":"/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_gpt4o_gptvllama.json",
#     "GPT-40-mini":"/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_gpt4omini_gptvllama.json",
#     "LLaMA": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_llama_gptvsllama.json",
#     "Qwen": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_qwen_gptvsllama.json"
# }

# # # Define paths for the four judge models
# # model_files = {
# #     "GPT-4o":"/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_gpt40_gptvsldeepseek.json",
# #     "GPT-40-mini":"/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_gpt40mini_gptvsdeepseek.json",
# #     "LLaMA": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_llama_gptvsdeepseek.json",
# #     "Qwen": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_qwen_act_gptvdeepseek.json"
# # }

# # Initialize tracking dictionaries
# topic_model_data = {}  # Store data by (topic, model)
# missing_ratings = []  # To log missing values

# traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# # Step 2: Load data and structure it correctly per topic-model pair
# judgments = {}
# for model, file_path in model_files.items():
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             judgments[model] = json.load(f)
            
#             for instance in judgments[model]:
#                 topic = instance.get("topic", "Unknown Topic")
#                 assigned_traits = instance.get("traits", {})

#                 # Initialize data storage if topic-model pair does not exist
#                 key = (topic, model)
#                 if key not in topic_model_data:
#                     topic_model_data[key] = {
#                         "topic": topic,
#                         "model": model,
#                         **{f"P1_{trait}": "Missing" for trait in traits},
#                         **{f"P2_{trait}": "Missing" for trait in traits},
#                         **{f"A_{trait}": assigned_traits.get(trait, "Missing") for trait in traits}  # Ground truth
#                     }

#                 # Extract predicted personality traits
#                 analysis = instance.get("analysis", {})
#                 person_one_data = analysis.get("Person_One", {})
#                 person_two_data = analysis.get("Person_Two", {})

#                 person_one_traits = person_one_data.get("predicted_bfi", {})
#                 person_two_traits = person_two_data.get("predicted_bfi", {})

#                 # Store values under the correct topic-model pair
#                 for trait in traits:
#                     topic_model_data[key][f"P1_{trait}"] = person_one_traits.get(trait, "Missing")
#                     topic_model_data[key][f"P2_{trait}"] = person_two_traits.get(trait, "Missing")

#     except Exception as e:
#         print(f"❌ ERROR: Could not load {file_path}: {str(e)}")

# # Step 3: Convert structured data into a DataFrame and Save
# df_validated = pd.DataFrame(topic_model_data.values())
# validated_csv = "check_validated_model_data_fixed.csv"
# df_validated.to_csv(validated_csv, index=False)

# print(f"✅ Fixed validated dataset saved as '{validated_csv}'. Proceeding with Fleiss' Kappa input preparation.")

# # Step 4: Construct a Proper CSV File for Fleiss' Kappa

# # Initialize a list to store data
# fleiss_kappa_input_data = []

# # Iterate over each topic and trait combination
# for topic in df_validated["topic"].unique():
#     for trait in traits:
#         # Extract the assigned ground-truth trait value
#         assigned_trait = df_validated[df_validated["topic"] == topic][f"A_{trait}"].values[0]

#         # Collect ratings from all models
#         trait_ratings_p1 = []
#         trait_ratings_p2 = []

#         for model in model_files.keys():
#             model_data = df_validated[(df_validated["topic"] == topic) & (df_validated["model"] == model)]
            
#             if not model_data.empty:
#                 p1_value = model_data[f"P1_{trait}"].values[0]  # Predicted by the model for Person 1
#                 p2_value = model_data[f"P2_{trait}"].values[0]  # Predicted by the model for Person 2
#             else:
#                 p1_value, p2_value = "Missing", "Missing"  # Mark missing values

#             trait_ratings_p1.append(p1_value)
#             trait_ratings_p2.append(p2_value)

#         # Store the structured data
#         fleiss_kappa_input_data.append({
#             "topic": topic,
#             "trait": trait,
#             "assigned_trait": assigned_trait,
#             "GPT-4o_P1": trait_ratings_p1[0], "GPT-4o_P2": trait_ratings_p2[0],
#             "GPT-40-mini_P1": trait_ratings_p1[1], "GPT-40-mini_P2": trait_ratings_p2[1],
#             "LLaMA_P1": trait_ratings_p1[2], "LLaMA_P2": trait_ratings_p2[2],
#             "Qwen_P1": trait_ratings_p1[3], "Qwen_P2": trait_ratings_p2[3],
#         })

# # Convert to DataFrame and Save CSV File
# df_fleiss_input = pd.DataFrame(fleiss_kappa_input_data)
# fleiss_input_csv = "check_fleiss_kappa_input_data_gptvgpt.csv"
# df_fleiss_input.to_csv(fleiss_input_csv, index=False)

# print(f"✅ Fleiss' Kappa input data saved as '{fleiss_input_csv}'.")


# from statsmodels.stats.inter_rater import fleiss_kappa

# # Reload the validated Fleiss' Kappa input dataset
# df_fleiss_input = pd.read_csv("check_fleiss_kappa_input_data_gptvgpt.csv")

# # Initialize lists for Fleiss' Kappa calculations for P1 and P2 separately
# fleiss_results_p1 = []
# fleiss_results_p2 = []

# # Iterate over each trait and properly structure the ratings matrix for P1 and P2 separately
# for trait in traits:
#     ratings_p1 = []
#     ratings_p2 = []
    
#     for _, row in df_fleiss_input[df_fleiss_input["trait"] == trait].iterrows():
#         # Collect ratings for all models for P1
#         trait_values_p1 = [
#             row["GPT-4o_P1"],
#             row["GPT-40-mini_P1"],
#             row["LLaMA_P1"],
#             row["Qwen_P1"]
#         ]

#         # Collect ratings for all models for P2
#         trait_values_p2 = [
#             row["GPT-4o_P2"],
#             row["GPT-40-mini_P2"],
#             row["LLaMA_P2"],
#             row["Qwen_P2"]
#         ]
        

#         # Count occurrences of "High" and "Low"
#         count_high_p1 = trait_values_p1.count("High")
#         count_low_p1 = trait_values_p1.count("Low")

#         count_high_p2 = trait_values_p2.count("High")
#         count_low_p2 = trait_values_p2.count("Low")

#         # Ensure each row sums to total number of judges (4 models)
#         if count_high_p1 + count_low_p1 == len(trait_values_p1):
#             ratings_p1.append([count_high_p1, count_low_p1])
        
#         if count_high_p2 + count_low_p2 == len(trait_values_p2):
#             ratings_p2.append([count_high_p2, count_low_p2])

#     # Compute Fleiss' Kappa only if there are valid ratings
#     fleiss_kappa_score_p1 = fleiss_kappa(ratings_p1, method='fleiss') if len(ratings_p1) > 0 else None
#     fleiss_kappa_score_p2 = fleiss_kappa(ratings_p2, method='fleiss') if len(ratings_p2) > 0 else None

#     # Store computed results
#     fleiss_results_p1.append({"Trait": trait, "Fleiss_Kappa_P1": fleiss_kappa_score_p1})
#     fleiss_results_p2.append({"Trait": trait, "Fleiss_Kappa_P2": fleiss_kappa_score_p2})

# # Convert to DataFrame and Save Results Separately for P1 and P2
# df_fleiss_results_p1 = pd.DataFrame(fleiss_results_p1)
# df_fleiss_results_p2 = pd.DataFrame(fleiss_results_p2)

# fleiss_results_p1_csv = "check_fleiss_kappa_results_p1_gptvgpt_v2.csv"
# fleiss_results_p2_csv = "check_fleiss_kappa_results_p2_gptvgpt_v2.csv"

# df_fleiss_results_p1.to_csv(fleiss_results_p1_csv, index=False)
# df_fleiss_results_p2.to_csv(fleiss_results_p2_csv, index=False)

# print(f"✅ Fleiss' Kappa successfully computed and saved separately for P1 in '{fleiss_results_p1_csv}' and for P2 in '{fleiss_results_p2_csv}'.")
import json
import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# 📌 Define the paths for the four judge models
model_files = {
    "GPT-4o": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_gpt4o_gptvllama.json",
    "GPT-40-mini": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_gpt4omini_gptvllama.json",
    "LLaMA": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_llama_gptvsllama.json",
    "Qwen": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_qwen_gptvsllama.json"
}

# 📌 Define the personality traits we are analyzing
traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# 📌 Initialize tracking dictionary
topic_trait_ratings = {}

# Step 1: Load data from JSON files and structure it properly
judgments = {}
for model, file_path in model_files.items():
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            judgments[model] = json.load(f)

            for instance in judgments[model]:
                topic = instance.get("topic", "Unknown Topic")
                analysis = instance.get("analysis", {})

                # Extract predicted personality traits for each participant
                person_one_data = analysis.get("Person_One", {}).get("predicted_bfi", {})
                person_two_data = analysis.get("Person_Two", {}).get("predicted_bfi", {})

                for trait in traits:
                    key = (topic, trait)  # Unique pair of (topic, trait)

                    if key not in topic_trait_ratings:
                        topic_trait_ratings[key] = {
                            "topic": topic,
                            "trait": trait,
                            "P1_ratings": {model: "Missing" for model in model_files.keys()},
                            "P2_ratings": {model: "Missing" for model in model_files.keys()}
                        }

                    topic_trait_ratings[key]["P1_ratings"][model] = person_one_data.get(trait, "Missing")
                    topic_trait_ratings[key]["P2_ratings"][model] = person_two_data.get(trait, "Missing")

    except Exception as e:
        print(f"❌ ERROR: Could not load {file_path}: {str(e)}")

# Step 2: Convert structured data into a DataFrame
df_fleiss_input = pd.DataFrame([
    {"topic": topic, "trait": trait, **data["P1_ratings"], **data["P2_ratings"]}
    for (topic, trait), data in topic_trait_ratings.items()
])

# Debugging: Ensure DataFrame is correctly structured
if df_fleiss_input.empty:
    print("🚨 ERROR: Fleiss' Kappa input DataFrame is empty! Check JSON loading.")
    exit()

fleiss_input_csv = "fleiss_kappa_input_data_per_topic_trait.csv"
df_fleiss_input.to_csv(fleiss_input_csv, index=False)

print(f"✅ Fleiss' Kappa input data saved as '{fleiss_input_csv}'.")

# Step 3: Compute Fleiss' Kappa per (topic, trait) pair with Debugging
fleiss_results = []
trait_kappa_p1 = {trait: [] for trait in traits}
trait_kappa_p2 = {trait: [] for trait in traits}

for (topic, trait), data in topic_trait_ratings.items():
    ratings_p1 = []
    ratings_p2 = []

    # Convert dictionary values into lists
    trait_values_p1 = list(data["P1_ratings"].values())
    trait_values_p2 = list(data["P2_ratings"].values())

    # Ignore "Missing" values
    trait_values_p1 = [v for v in trait_values_p1 if v in ["High", "Low"]]
    trait_values_p2 = [v for v in trait_values_p2 if v in ["High", "Low"]]

    # Debugging: Print trait ratings before computation
    print(f"🔎 DEBUG: Topic: {topic} | Trait: {trait}")
    print(f"   P1 Ratings: {trait_values_p1}")
    print(f"   P2 Ratings: {trait_values_p2}")

    # Check if all values are the same (this would cause NaN in Kappa)
    if len(set(trait_values_p1)) == 1:
        fleiss_kappa_p1 = 1.0  # Perfect agreement
    else:
        ratings_p1.append([trait_values_p1.count("High"), trait_values_p1.count("Low")])

    if len(set(trait_values_p2)) == 1:
        fleiss_kappa_p2 = 1.0  # Perfect agreement
    else:
        ratings_p2.append([trait_values_p2.count("High"), trait_values_p2.count("Low")])

    # Compute Fleiss' Kappa only if valid
    try:
        if len(ratings_p1) > 0:
            fleiss_kappa_p1 = fleiss_kappa(ratings_p1)
        if len(ratings_p2) > 0:
            fleiss_kappa_p2 = fleiss_kappa(ratings_p2)
    except Exception as e:
        print(f"❌ ERROR computing Fleiss' Kappa for (topic: {topic}, trait: {trait}): {str(e)}")
        fleiss_kappa_p1, fleiss_kappa_p2 = None, None

    print(f"   ✅ Fleiss' Kappa P1: {fleiss_kappa_p1}")
    print(f"   ✅ Fleiss' Kappa P2: {fleiss_kappa_p2}")

    # Store per-topic results
    fleiss_results.append({
        "Topic": topic, "Trait": trait,
        "Fleiss_Kappa_P1": fleiss_kappa_p1,
        "Fleiss_Kappa_P2": fleiss_kappa_p2
    })

    # Store results for averaging later
    if fleiss_kappa_p1 is not None:
        trait_kappa_p1[trait].append(fleiss_kappa_p1)
    if fleiss_kappa_p2 is not None:
        trait_kappa_p2[trait].append(fleiss_kappa_p2)

# Step 4: Save per-topic Fleiss' Kappa results
df_fleiss_results = pd.DataFrame(fleiss_results)
df_fleiss_results.to_csv("fleiss_kappa_results_per_topic_trait.csv", index=False)

print("✅ Fleiss' Kappa computation per topic completed and saved!")

# Step 5: Compute Average Fleiss' Kappa Per Trait
trait_avg_kappa = []
for trait in traits:
    avg_p1 = np.mean(trait_kappa_p1[trait]) if trait_kappa_p1[trait] else None
    avg_p2 = np.mean(trait_kappa_p2[trait]) if trait_kappa_p2[trait] else None
    trait_avg_kappa.append({"Trait": trait, "Avg_Fleiss_Kappa_P1": avg_p1, "Avg_Fleiss_Kappa_P2": avg_p2})

# Save Trait Averages
df_trait_avg_kappa = pd.DataFrame(trait_avg_kappa)
df_trait_avg_kappa.to_csv("fleiss_kappa_avg_per_trait.csv", index=False)

print("✅ Fleiss' Kappa trait averages computed and saved!")
