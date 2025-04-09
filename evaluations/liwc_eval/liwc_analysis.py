import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================
# Step 1: Load and Process LIWC Data
# ====================================
# liwc_file = "/m/edia/data_4tb/pranav/llms_gen_eval/evaluations/liwc_eval/completed/liwc_analysis_gptvgpt.csv"


# liwc_file = "/media/data_4tb/pranav/llms_gen_eval/evaluations/liwc_eval/completed/liwc_analysis_gptvllama.csv"


# liwc_file = "/media/data_4tb/pranav/llms_gen_eval/evaluations/liwc_eval/completed/liwc_analysis_gptvdeepseek.csv"

# Define file paths for different model comparisons
liwc_files = {
    # "gptvgpt": "liwc_analysis_gptvgpt.csv",
    # "gptvllama": "liwc_analysis_gptvllama.csv",
    # "gptvdeepseek": "liwc_analysis_gptvdeepseek.csv"
"gptvgpt": "/media/data_4tb/pranav/llms_gen_eval/evaluations/liwc_eval/completed/liwc_analysis_gptvgpt.csv",
"gptvllama" :"/media/data_4tb/pranav/llms_gen_eval/evaluations/liwc_eval/completed/liwc_analysis_gptvllama.csv",
"gptvdeepseek" :"/media/data_4tb/pranav/llms_gen_eval/evaluations/liwc_eval/completed/liwc_analysis_gptvdeepseek.csv"
}

# Define custom colors for each model comparison
bar_colors_map = {
    "gptvgpt": ["#66c2a4", "#fc8d59"],  # Colors for GPT vs GPT
    "gptvllama": ["#4eb3d3", "#66c2a4"],  # Colors for GPT vs LLaMA
    "gptvdeepseek": ["#9e9ac8", "#66c2a4"]  # Colors for GPT vs DeepSeek
}

# Trait mappings
trait_mappings = {
    "Agreeableness": ["Social", "prosocial", "polite", "moral"],
    "Openness": ["cogproc", "insight", "certitude", "differ"],
    "Conscientiousness": ["work", "achieve", "certitude"],
    "Extraversion": ["Social", "Conversation", "assent", "Exclam"],
    "Neuroticism": ["emo_anx", "emo_anger", "emo_sad", "negate"]
}

# Short name mapping for better visualization
trait_short_names = {
    "Agreeableness": "Ag",
    "Openness": "Op",
    "Conscientiousness": "Co",
    "Extraversion": "Ex",
    "Neuroticism": "Ne"
}

fig_size = (13, 8)

# Process each file and generate figures
for model_name, liwc_file in liwc_files.items():
    print(f"Processing {model_name}...")

    # Load actual data
    df = pd.read_csv(liwc_file)

    # Preserve original traits
    original_traits = df[list(trait_mappings.keys())].copy()

    # Compute mean values for each trait using the correct mapped features
    for trait, features in trait_mappings.items():
        df[trait] = df[features].mean(axis=1)  # Correct averaging of features

    # Convert into "High" or "Low" based on median threshold
    for trait in trait_mappings.keys():
        threshold = df[trait].median()  # Correct threshold calculation
        df[f"Predicted_{trait}"] = np.where(df[trait] >= threshold, "High", "Low")
        df[f"Original_{trait}"] = original_traits[trait]  # Retain original values

    # Select and save processed trait data
    bfi_columns = ["Topic", "Speaker"] + [f"Predicted_{t}" for t in trait_mappings.keys()] + [f"Original_{t}" for t in trait_mappings.keys()]
    df_bfi_results = df[bfi_columns]
    bfi_filename = f"bfi_from_liwc_with_original_traits_{model_name}.csv"
    df_bfi_results.to_csv(bfi_filename, index=False)

    print(f"✅ BFI extraction complete. Results saved to '{bfi_filename}'.")

    # Load processed data for accuracy calculation
    df = pd.read_csv(bfi_filename)

    # Split dataset into Person 1 and Person 2
    df_person_1 = df[df["Speaker"] == "Person_One"]
    df_person_2 = df[df["Speaker"] == "Person_Two"]

    # Store accuracy results
    accuracy_results = {}

    for trait in trait_mappings.keys():
        df_person_1[f"Match_{trait}"] = df_person_1[f"Predicted_{trait}"] == df_person_1[f"Original_{trait}"]
        df_person_2[f"Match_{trait}"] = df_person_2[f"Predicted_{trait}"] == df_person_2[f"Original_{trait}"]

        accuracy_results[trait] = {
            "Person_One_Accuracy": df_person_1[f"Match_{trait}"].mean() * 100,  # Correct accuracy calculation
            "Person_Two_Accuracy": df_person_2[f"Match_{trait}"].mean() * 100
        }

    # Convert results to DataFrame and save
    accuracy_df = pd.DataFrame(accuracy_results).T
    accuracy_filename = f"trait_accuracy_results_fixed_{model_name}.csv"
    accuracy_df.to_csv(accuracy_filename)

    print(f"✅ Accuracy calculation completed. Results saved to '{accuracy_filename}'.")
    legend_properties = {'weight':'bold', 'size':'35'}

    # 📊 Bar Chart - Accuracy Comparison
    plt.figure(figsize=fig_size)
    ax = accuracy_df.plot(
        kind="bar",
        figsize=fig_size,
        color=bar_colors_map[model_name],
        edgecolor="black",
        linewidth=2
    )

    # Apply short trait names
    ax.set_xticklabels([trait_short_names[trait] for trait in accuracy_df.index], rotation=0, fontsize=14, fontweight="bold")

    # Labels and formatting
    plt.ylabel("Accuracy (%)", fontsize=40, fontweight="bold")
    plt.xlabel("Personality Traits", fontsize=40, fontweight="bold")
    plt.xticks(rotation=0, fontsize=35, fontweight="bold")
    plt.ylim(0, 100)
    plt.yticks(rotation=0, fontsize=35, fontweight="bold")
    plt.legend(["Participant 1", "Participant 2"], loc="upper right", frameon=True, prop = legend_properties)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save figure
    bar_chart_filename = f"trait_accuracy_bar_chart_{model_name}.pdf"
    plt.savefig(bar_chart_filename,  format="pdf", bbox_inches="tight")
    plt.close()
    
    # 🔥 Heatmap - Accuracy Visualization
    plt.figure(figsize=fig_size)
    sns.heatmap(accuracy_df, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5, annot_kws={"fontsize": 14})

    # Title and labels
    plt.title(f"Accuracy Heatmap of Trait Predictions - {model_name}", fontsize=22, fontweight="bold")
    plt.xlabel("Prediction Accuracy (%)", fontsize=28, fontweight="bold")
    plt.ylabel("Personality Traits", fontsize=28, fontweight="bold")

    # Save heatmap
    heatmap_filename = f"trait_accuracy_heatmap_{model_name}.pdf"
    plt.savefig(heatmap_filename, format="pdf", dpi=350, bbox_inches="tight")
    plt.close()

    print(f"✅ Figures saved: '{bar_chart_filename}', '{heatmap_filename}'.")

print("✅ All models processed successfully! All figures generated.")
