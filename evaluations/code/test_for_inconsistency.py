import json
import pandas as pd
import pdb


# Load dataset
file_path = "parse_json/final_gptvs_gpt_discorse/bfi_analysis_gpt4o_gptvsgpt.json"  # Replace with actual path
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize lists for structured data and error tracking
structured_data = []
error_instances = []  # Store problematic instances
processed_instances = []  # Log successfully processed instances

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

        # Log successful instance processing
        processed_instances.append({"index": index, "topic": topic, "status": "Processed Successfully"})

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

# Save problematic instances for further inspection
if error_instances:
    with open("error_instances.json", "w") as error_file:
        json.dump(error_instances, error_file, indent=4)
    print(f"🔍 Saved problematic instances to 'error_instances.json'")

# Save successfully processed instances log
if processed_instances:
    with open("PCM_4o_eval_4ov4omini.json", "w") as success_file:
        json.dump(processed_instances, success_file, indent=4)
    print(f"📄 Logged all successfully processed instances in 'processed_instances.json'")

# Convert structured data to DataFrame
df = pd.DataFrame(structured_data)

# Final confirmation
print(f"✅ Successfully processed {len(df)} complete instances into DataFrame")

# Save results as JSON
#fiel saving format is: NameofEvaluation_ModelthatEvaluated_eval_DiscorseBetweenModelsName.json
# output_file = "PCM_4o_eval_4ov4omini.json"
# with open(output_file, "w") as f:
#     json.dump(metrics_results, f, indent=4)

# print(f"Metrics saved to {output_file}")
