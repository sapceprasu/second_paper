import json

# Required structure
EXPECTED_KEYS = {"Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"}
VALID_TRAIT_VALUES = {"High", "Low"}
VALID_CONSISTENCY_VALUES = {"Yes", "No"}

def validate_entry(entry, index):
    errors = []

    # Ensure "analysis" exists
    if "analysis" not in entry:
        errors.append(f"Missing 'analysis' section.")
        return errors  # No need to check further

    # Ensure both Person_One and Person_Two exist
    for person in ["Person_One", "Person_Two"]:
        if person not in entry["analysis"]:
            errors.append(f"Missing '{person}' in analysis.")
            continue

        person_data = entry["analysis"][person]

        # Ensure "predicted_bfi" exists
        if "predicted_bfi" not in person_data:
            errors.append(f"'{person}' is missing 'predicted_bfi'.")
            continue

        predicted_bfi = person_data["predicted_bfi"]

        # Check all personality traits
        for trait in EXPECTED_KEYS:
            if trait not in predicted_bfi:
                errors.append(f"'{person}' is missing '{trait}' trait.")
            elif predicted_bfi[trait] not in VALID_TRAIT_VALUES:
                errors.append(f"'{person}' has invalid value '{predicted_bfi[trait]}' for '{trait}'. Should be 'High' or 'Low'.")

        # Ensure "consistent_with_traits" exists and has valid values
        if "consistent_with_traits" not in person_data:
            errors.append(f"'{person}' is missing 'consistent_with_traits'.")
        elif person_data["consistent_with_traits"] not in VALID_CONSISTENCY_VALUES:
            errors.append(f"'{person}' has invalid 'consistent_with_traits' value '{person_data['consistent_with_traits']}'. Should be 'Yes' or 'No'.")

    return errors

def validate_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    invalid_entries = 0
    for i, entry in enumerate(data):
        errors = validate_entry(entry, i)
        if errors:
            invalid_entries += 1
            print(f"\n❌ Entry {i+1} - {entry.get('topic', 'Unknown Topic')} has errors:")
            for err in errors:
                print(f"   - {err}")

    print(f"\n✅ Validation complete. {invalid_entries} entries did not conform to the required format.")

# Run validation on dataset file
validate_json("/media/data_4tb/pranav/llms_gen_eval/parse_json/bfi_analysis_deepseek_gptvsllama.json")  # Replace with actual JSON filename
