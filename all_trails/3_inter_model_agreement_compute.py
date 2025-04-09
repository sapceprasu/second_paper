import pandas as pd
import json

# Load the inter-model agreement results
with open("inter_model_agreement.json", "r") as f:
    ima_results = json.load(f)

# Convert JSON data to DataFrame
df_ima = pd.DataFrame(ima_results)

# Expand the IMA scores into separate columns for traits
ima_scores_df = df_ima["ima_scores"].apply(pd.Series)
df_ima = pd.concat([df_ima.drop(columns=["ima_scores"]), ima_scores_df], axis=1)

# Check unique discourse instances (topics + traits)
print("\n📌 Unique topics count:", df_ima["topic"].nunique())
print("📌 Unique personality settings count:", df_ima["unique_id"].nunique())

# Display first few rows for verification
print("\n🔍 First few rows of the IMA dataset before aggregation:")
print(df_ima.head())

# Check if any unique topics are missing
missing_topics = set(df_ima["topic"]) - set(df_ima["unique_id"])
print("\n⚠️ Missing Topics:", missing_topics if missing_topics else "None")

# Save this intermediate debug file
df_ima.to_csv("ima_debug.csv", index=False)

print("✅ Debug file saved as 'ima_debug.csv'. Check for inconsistencies.")
