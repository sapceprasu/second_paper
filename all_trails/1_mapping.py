import json
import pandas as pd

# List of model judgment files
model_files = {
    "GPT-4o": "bfi_analysis_gpt4o.json",
    "LLaMA3.2": "bfi_analysis_gpt4o.json",
    "LLaMA3.3": "bfi_analysis_gpt4o.json",
    "Qwen": "bfi_analysis_gpt4o.json",
    "DeepSeek": "bfi_analysis_gpt4o.json"
}

# Load all model judgments
judgments = {}
for model, file_path in model_files.items():
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            judgments[model] = json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Could not load {file_path}: {str(e)}")
        judgments[model] = []

# Initialize structured data
structured_data = []

# Process each model's judgment and align data
for model, instances in judgments.items():
    for instance in instances:
        try:
            topic = instance.get("topic", "Unknown Topic")
            traits = instance.get("traits", {})  # Assigned personality settings
            unique_id = f"{topic} | Traits: {traits}"

            # Ensure analysis exists
            analysis = instance.get("analysis", None)
            if not analysis:
                print(f"⚠️ Skipping {topic} (Missing 'analysis')")
                continue

            # Ensure both Person One and Person Two exist
            person_one_data = analysis.get("Person_One", None)
            person_two_data = analysis.get("Person_Two", None)
            if not person_one_data or not person_two_data:
                print(f"⚠️ Skipping {topic} (Missing 'Person_One' or 'Person_Two')")
                continue

            # Extract predicted traits safely
            person_one_traits = person_one_data.get("predicted_bfi", {})
            person_two_traits = person_two_data.get("predicted_bfi", {})

            structured_data.append({
                "topic": topic,
                "unique_id": unique_id,
                "model": model,
                **{f"P1_{trait}": person_one_traits.get(trait, "Missing") for trait in ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]},
                **{f"P2_{trait}": person_two_traits.get(trait, "Missing") for trait in ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]}
            })

            print(f"✅ Mapped topic: {topic} | Traits: {traits} | Judge: {model}")

        except Exception as e:
            print(f"❌ ERROR mapping topic {topic} for model {model}: {str(e)}")

# Convert to DataFrame
df = pd.DataFrame(structured_data)

# Print first few rows for verification
print("\n📌 First few rows of the structured dataset:")
print(df.head())

# Save structured data for further processing
df.to_csv("updated_mapped_model_data.csv", index=False)

print("✅ Data mapping complete. Saved as 'updated_mapped_model_data.csv'.")
