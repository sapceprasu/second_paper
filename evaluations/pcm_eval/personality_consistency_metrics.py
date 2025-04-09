import json
import pandas as pd
import pdb


# Load dataset
file_path = "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_llama_gptvsgpt.json"  # Replace with actual path
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize lists for structured data and error tracking
structured_data = []
error_instances = []  # Store problematic instancess

for index, instance in enumerate(data):
    try:
        topic = instance.get("topic", "Unknown Topic")  # Get topic safely
        assigned_traits = instance.get("traits", {})  # Get assigned traits safely

        # Ensure "analysis" key exists and contains both speakers
        analysis = instance.get("analysis", {})
        if not isinstance(analysis, dict) or "Person_One" not in analysis or "Person_Two" not in analysis:
            print(f"⚠️ Skipping instance {index}: Missing or invalid 'analysis' key")
            error_instances.append({"index": index, "instance": instance, "error": "Missing Person_One or Person_Two"})
            continue  # Skip the entire instance

        # Get predictions for both speakers
        person_one_data = analysis.get("Person_One", {})
        person_two_data = analysis.get("Person_Two", {})

        # Ensure "predicted_bfi" exists for both speakers
        if "predicted_bfi" not in person_one_data or "predicted_bfi" not in person_two_data:
            print(f"⚠️ Skipping instance {index}: Missing 'predicted_bfi' for one or both speakers")
            error_instances.append({"index": index, "instance": instance, "error": "Missing 'predicted_bfi'"})
            continue  # Skip the entire instance

        # Extract predicted traits
        person_one_traits = person_one_data["predicted_bfi"]
        person_two_traits = person_two_data["predicted_bfi"]

        # Add structured data for both speakers
        structured_data.append({
            "index": index,
            "topic": topic,
            "speaker": "Person_One",
            **assigned_traits,
            **{f"predicted_{trait}": value for trait, value in person_one_traits.items()}
        })

        structured_data.append({
            "index": index,
            "topic": topic,
            "speaker": "Person_Two",
            **assigned_traits,
            **{f"predicted_{trait}": value for trait, value in person_two_traits.items()}
        })

        # Log progress every 10 instances
        if index % 10 == 0:
            print(f"✅ Successfully processed {index+1}/{len(data)} instances")

    except Exception as e:
        print(f"❌ ERROR at instance {index}: {str(e)}")
        error_instances.append({"index": index, "instance": instance, "error": str(e)})

# Convert structured data to DataFrame
df = pd.DataFrame(structured_data)

# Function to compute Exact Match Accuracy (EMA)
def compute_ema(df):
    """Computes Exact Match Accuracy per speaker."""
    ema_results = {}

    for speaker in ["Person_One", "Person_Two"]:
        speaker_df = df[df["speaker"] == speaker]
        if speaker_df.empty:
            ema_results[speaker] = None
            continue

        matches = (
            (speaker_df["Agreeableness"] == speaker_df["predicted_Agreeableness"]) &
            (speaker_df["Openness"] == speaker_df["predicted_Openness"]) &
            (speaker_df["Conscientiousness"] == speaker_df["predicted_Conscientiousness"]) &
            (speaker_df["Extraversion"] == speaker_df["predicted_Extraversion"]) &
            (speaker_df["Neuroticism"] == speaker_df["predicted_Neuroticism"])
        ).sum()

        total_instances = len(speaker_df)
        ema_results[speaker] = round((matches / total_instances) * 100, 2) if total_instances > 0 else None

    return ema_results

# Function to compute Per-Trait Accuracy (PTA)
def compute_pta(df):
    """Computes Per-Trait Accuracy per speaker."""
    pta_results = {}
    traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

    for speaker in ["Person_One", "Person_Two"]:
        speaker_df = df[df["speaker"] == speaker]
        if speaker_df.empty:
            pta_results[speaker] = None
            continue

        trait_accuracy = {}
        for trait in traits:
            matches = (speaker_df[trait] == speaker_df[f"predicted_{trait}"]).sum()
            total_instances = len(speaker_df)
            trait_accuracy[trait] = round((matches / total_instances) * 100, 2) if total_instances > 0 else None

        pta_results[speaker] = trait_accuracy

    return pta_results

# Function to compute Speaker-Level Consistency (SLC)
def compute_slc(df):
    """Computes Speaker-Level Consistency by comparing the consistency of predictions for both speakers."""
    slc_results = {}
    traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

    person_one = df[df["speaker"] == "Person_One"]
    person_two = df[df["speaker"] == "Person_Two"]

    if len(person_one) != len(person_two):
        return {"Error": "Mismatch in speaker data"}

    trait_consistency = {}
    for trait in traits:
        matches = (person_one[f"predicted_{trait}"].values == person_two[f"predicted_{trait}"].values).sum()
        total_instances = len(person_one)
        trait_consistency[trait] = round((matches / total_instances) * 100, 2) if total_instances > 0 else None

    slc_results["Overall"] = trait_consistency
    return slc_results

# Compute Metrics
ema_results = compute_ema(df)
pta_results = compute_pta(df)
slc_results = compute_slc(df)

# Store all results in structured JSON format
metrics_results = {
    "Exact Match Accuracy (EMA)": ema_results,
    "Per-Trait Accuracy (PTA)": pta_results,
    "Speaker-Level Consistency (SLC)": slc_results,
    "description": {
        "EMA": "Measures how often all five predicted traits exactly match the assigned traits.",
        "PTA": "Measures accuracy per individual personality trait.",
        "SLC": "Measures how similarly models assign personality traits to both speakers in a discourse."
    }
}

# Save results as JSON
output_file = "PCM_gpt_eval_4ovllama.json"
with open(output_file, "w") as f:
    json.dump(metrics_results, f, indent=4)

# Save problematic instances
if error_instances:
    with open("error_gpt_eval_4ovllama.json", "w") as error_file:
        json.dump(error_instances, error_file, indent=4)
    print(f"🔍 Saved problematic instances to 'error_instances.json'")

print(f"✅ Metrics saved to {output_file}")
print(f"✅ Successfully processed {len(df)} complete instances into DataFrame")

# Save results as JSON
#fiel saving format is: NameofEvaluation_ModelthatEvaluated_eval_DiscorseBetweenModelsName.json
# output_file = "PCM_4o_eval_4ov4omini.json"
# with open(output_file, "w") as f:
#     json.dump(metrics_results, f, indent=4)

# print(f"Metrics saved to {output_file}")
