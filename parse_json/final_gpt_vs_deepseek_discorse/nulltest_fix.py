import json
from collections import defaultdict

# Sample JSON data structure (assuming data is loaded from a file)
with open("/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_gpt40_gptvsldeepseek.json", 'r') as f:
    data = json.load(f)

# Initialize counters for original trait distributions and classified traits
trait_counts = defaultdict(lambda: {"High": 0, "Low": 0})
classified_counts_p1 = defaultdict(lambda: {"High": 0, "Low": 0, "Other": 0})
classified_counts_p2 = defaultdict(lambda: {"High": 0, "Low": 0, "Other": 0})

# Allowed trait values
valid_traits = {"High", "Low"}

# Process each topic in the dataset
for topic in data:
    # Count original trait distributions
    for trait, value in topic["traits"].items():
        if value in valid_traits:
            trait_counts[trait][value] += 1
        else:
            print(f"⚠️ Unexpected original trait value '{value}' for {trait}")

    # Count classified traits for Person One
    for trait, value in topic["analysis"]["Person_One"]["predicted_bfi"].items():
        if value in valid_traits:
            classified_counts_p1[trait][value] += 1
        else:
            classified_counts_p1[trait]["Other"] += 1  # Handle unexpected values
            print(f"⚠️ Unexpected classified value '{value}' for {trait} (Person One)")

    # Count classified traits for Person Two
    for trait, value in topic["analysis"]["Person_Two"]["predicted_bfi"].items():
        if value in valid_traits:
            classified_counts_p2[trait][value] += 1
        else:
            classified_counts_p2[trait]["Other"] += 1  # Handle unexpected values
            print(f"⚠️ Unexpected classified value '{value}' for {trait} (Person Two)")

# Print results
print("\nOriginal Trait Distribution:")
for trait, counts in trait_counts.items():
    print(f"{trait}: High - {counts['High']}, Low - {counts['Low']}")

print("\nClassified Traits for Person One:")
for trait, counts in classified_counts_p1.items():
    print(f"{trait}: High - {counts['High']}, Low - {counts['Low']}, Other - {counts['Other']}")

print("\nClassified Traits for Person Two:")
for trait, counts in classified_counts_p2.items():
    print(f"{trait}: High - {counts['High']}, Low - {counts['Low']}, Other - {counts['Other']}")