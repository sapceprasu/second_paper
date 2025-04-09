# Step 1: Reload the CSV file for debugging
df_debug = pd.read_csv("updated_mapped_model_data.csv")

# Step 2: Check for missing values in trait columns
missing_values = df_debug.isnull().sum()

# Step 3: Verify that models in the CSV match the expected model list
csv_models = df_debug["model"].unique()
expected_models = list(model_files.keys())

# Step 4: Check if trait values are identical across all models for each topic
traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]
consistent_across_models = {}

for trait in traits:
    p1_values = df_debug.groupby("unique_id")[f"P1_{trait}"].nunique()
    p2_values = df_debug.groupby("unique_id")[f"P2_{trait}"].nunique()

    consistent_across_models[trait] = {
        "P1_Same_Across_Models": (p1_values == 1).sum(),
        "P2_Same_Across_Models": (p2_values == 1).sum(),
        "Total_Unique_Instances": len(p1_values)
    }

# Convert results to DataFrame for readability
df_consistency_check = pd.DataFrame.from_dict(consistent_across_models, orient="index")

# Step 5: Identify duplicate entries for unique_id and model
duplicate_check = df_debug[df_debug.duplicated(subset=["unique_id", "model"], keep=False)]

# Step 6: Print Diagnostic Report
report_lines = []

report_lines.append("\n🔍 **Diagnostic Report for Inter-Model Agreement Calculation:**\n")
report_lines.append("➡️ **Missing Values in CSV:**\n")
report_lines.append(str(missing_values))

report_lines.append("\n➡️ **Unique Models in CSV vs Expected Models:**\n")
report_lines.append(f"CSV Models: {csv_models}")
report_lines.append(f"Expected Models: {expected_models}")

report_lines.append("\n➡️ **Trait Consistency Check Across Models:** (Are traits identical across all models?)\n")
report_lines.append(str(df_consistency_check))

report_lines.append("\n➡️ **Duplicate Unique_ID & Model Entries:**\n")
report_lines.append(str(duplicate_check))

# Save Diagnostic Report to File
diagnostic_report_path = "inter_model_agreement_diagnostic_report.txt"
with open(diagnostic_report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"✅ Diagnostic report saved as '{diagnostic_report_path}'")
