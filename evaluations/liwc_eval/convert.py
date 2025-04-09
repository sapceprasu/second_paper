import json
import pandas as pd

# Load the dataset (Modify the file path as needed)
input_file = "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/final_discorse_gptvsdeepseek.json"  # Replace with your actual file path

# Load the dataset (Modify the file path as needed)
# input_file = "bfi_analysis_gptvsgpt.json"  # Replace with your actual file
output_file = "liwc_input_gptvdeepseek.csv"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# List to store structured data for LIWC analysis
liwc_data = []

# Iterate through each debate topic
for entry in data:
    topic = entry["topic"]
    traits = entry["traits"]  # Assigned personality traits for this topic
    
    # Collect speaker-wise discourse
    speaker_texts = {"Person_One": [], "Person_Two": []}
    
    if "discourse" in entry:
        for message in entry["discourse"]:
            if "gpt-4o:" in message or "DeepSeek:" in message or "LLaMA-3:" in message:  # Identify speaker 1
                speaker_texts["Person_One"].append(message.split(":", 1)[1].strip())
            elif "gpt-4o-mini:" in message or "GPT-4o:" in message or "GPT-4:" in message :  # Identify speaker 2
                speaker_texts["Person_Two"].append(message.split(":", 1)[1].strip())

    # Append structured data for LIWC
    for speaker, messages in speaker_texts.items():
        if messages:  # Ensure there's valid text
            liwc_data.append({
                "Topic": topic,
                "Speaker": speaker,
                "Text": " ".join(messages),
                "Agreeableness": traits["Agreeableness"],
                "Openness": traits["Openness"],
                "Conscientiousness": traits["Conscientiousness"],
                "Extraversion": traits["Extraversion"],
                "Neuroticism": traits["Neuroticism"]
            })

# Convert to DataFrame
df_liwc = pd.DataFrame(liwc_data)

# Save structured text data for LIWC analysis
df_liwc.to_csv(output_file, index=False, encoding="utf-8")

print("✅ Data structured and saved as 'liwc_input.csv' for LIWC analysis.")

# Save structured text data for LIWC analysis
# df_liwc.to_csv("liwc_input_gptvgpt.csv", index=False, encoding="utf-8")



# Iterate through each debate instance and collect messages per speaker