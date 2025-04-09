
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths, judges, models, and custom color maps for each dataset
datasets = [
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_gpt4o_gptvsgpt.json",
        "judge": "GPT-4o",
        "models": "GPT vs GPT",
        "cmap": ["#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45"] #gpt judge colors
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_gpt4omini_gptvsgpt.json",
        "judge": "GPT-4o-mini",
        "models": "GPT vs GPT",
        "cmap": ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801"]
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_llama_gptvsgpt.json",
        "judge": "LLaMA",
        "models": "GPT vs GPT",
        "cmap": ["#d0d1e6", "#a6bddb", "#a6bddb", "#3690c0", "#0570b0"] #llama juge colors
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/bfi_analysis_qwen_act_gptvsgpt.json",
        "judge": "Qwen",
        "models": "GPT vs GPT",
        "cmap": ["#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3"] #llama juge colors
    },
    ### gptvsllama
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_gpt4o_gptvllama.json",
        "judge": "GPT-4o",
        "models": "GPT vs Llama",
        "cmap": ["#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45"] #gpt judge colors
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_gpt4omini_gptvllama.json",
        "judge": "GPT-4o-mini",
        "models": "GPT vs Llama",
        "cmap": ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801"]
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_llama_gptvsllama.json",
        "judge": "LLaMA",
        "models": "GPT vs Llama",
        "cmap": ["#d0d1e6", "#a6bddb", "#a6bddb", "#3690c0", "#0570b0"] #llama juge colors
    },
        {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/bfi_analysis_qwen_gptvsllama.json",
        "judge": "Qwen",
        "models": "GPT vs Llama",
        "cmap": ["#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3"] #llama juge colors
    },

    ##### gpt vs deepseek!!
    {

        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_gpt40_gptvsldeepseek.json",
        "judge": "GPT-4o",
        "models": "GPT vs Deepseek",
        "cmap": ["#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45"] #gpt judge colors
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_gpt40mini_gptvsdeepseek.json",
        "judge": "GPT-4o-mini",
        "models": "GPT vs Deepseek",
        "cmap": ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801"] #gpt40mini judge colors 
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_llama_gptvsdeepseek.json",
        "judge": "LLaMA",
        "models": "GPT vs Deepseek",
        "cmap": ["#d0d1e6", "#a6bddb", "#a6bddb", "#3690c0", "#0570b0"] #llama juge colors
    },
    {
        "path": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/bfi_analysis_qwen_act_gptvdeepseek.json",
        "judge": "Qwen",
        "models": "GPT vs Deepseek",
        "cmap": ["#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3"] #llama juge colors
    },
]

# Define personality traits
traits = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# Define mapping for short trait names
trait_short_names = {
    "Agreeableness": "Ag",
    "Openness": "Op",
    "Conscientiousness": "Co",
    "Extraversion": "Ex",
    "Neuroticism": "Ne"
}

# Process each dataset
for dataset in datasets:
    try:
        # Load the dataset
        with open(dataset["path"], "r", encoding="utf-8") as f:
            data = json.load(f)

        # Initialize storage for confusion metric totals
        confusion_totals = {trait: {"CCH_P1": 0, "CCL_P1": 0, "MCH_P1": 0, "MCL_P1": 0,
                                    "CCH_P2": 0, "CCL_P2": 0, "MCH_P2": 0, "MCL_P2": 0} for trait in traits}

        # Process each topic and accumulate counts
        for topic in data:
            assigned_traits = topic["traits"]
            analysis = topic["analysis"]

            for trait in traits:
                base_value = assigned_traits[trait]  # The assigned High/Low value

                pred_p1 = analysis["Person_One"]["predicted_bfi"].get(trait, None)
                pred_p2 = analysis["Person_Two"]["predicted_bfi"].get(trait, None)

                if pred_p1 is None or pred_p2 is None:
                    continue  # Skip if predictions are missing

                # Update Confusion Metrics for Person 1
                confusion_totals[trait]["CCH_P1"] += int(base_value == "High" and pred_p1 == "High")
                confusion_totals[trait]["CCL_P1"] += int(base_value == "Low" and pred_p1 == "Low")
                confusion_totals[trait]["MCH_P1"] += int(base_value == "Low" and pred_p1 == "High")
                confusion_totals[trait]["MCL_P1"] += int(base_value == "High" and pred_p1 == "Low")

                # Update Confusion Metrics for Person 2
                confusion_totals[trait]["CCH_P2"] += int(base_value == "High" and pred_p2 == "High")
                confusion_totals[trait]["CCL_P2"] += int(base_value == "Low" and pred_p2 == "Low")
                confusion_totals[trait]["MCH_P2"] += int(base_value == "Low" and pred_p2 == "High")
                confusion_totals[trait]["MCL_P2"] += int(base_value == "High" and pred_p2 == "Low")

        # Compute Accuracy Metrics AFTER accumulating all topic counts
        confusion_metrics = []
        for trait, counts in confusion_totals.items():
            CCH_P1, CCL_P1, MCH_P1, MCL_P1 = counts["CCH_P1"], counts["CCL_P1"], counts["MCH_P1"], counts["MCL_P1"]
            CCH_P2, CCL_P2, MCH_P2, MCL_P2 = counts["CCH_P2"], counts["CCL_P2"], counts["MCH_P2"], counts["MCL_P2"]

            # Compute Accuracy Metrics for P1 & P2
            HTA_P1 = (CCH_P1 / (CCH_P1 + MCL_P1) * 100) if (CCH_P1 + MCL_P1) > 0 else 0
            LTA_P1 = (CCL_P1 / (CCL_P1 + MCH_P1) * 100) if (CCL_P1 + MCH_P1) > 0 else 0
            HTA_P2 = (CCH_P2 / (CCH_P2 + MCL_P2) * 100) if (CCH_P2 + MCL_P2) > 0 else 0
            LTA_P2 = (CCL_P2 / (CCL_P2 + MCH_P2) * 100) if (CCL_P2 + MCH_P2) > 0 else 0

            # Compute Total Classification Accuracy (TCC)
            TCC_P1 = ((CCH_P1 + CCL_P1) / (CCH_P1 + CCL_P1 + MCH_P1 + MCL_P1) * 100) if (CCH_P1 + CCL_P1 + MCH_P1 + MCL_P1) > 0 else 0
            TCC_P2 = ((CCH_P2 + CCL_P2) / (CCH_P2 + CCL_P2 + MCH_P2 + MCL_P2) * 100) if (CCH_P2 + CCL_P2 + MCH_P2 + MCL_P2) > 0 else 0

            confusion_metrics.append([trait, HTA_P1, LTA_P1,HTA_P2, LTA_P2])

        # Convert to DataFrame
        df_confusion = pd.DataFrame(confusion_metrics, columns=["Trait", "HTA_P1", "LTA_P1",  "HTA_P2", "LTA_P2"])

        # Convert DataFrame to heatmap format
        df_heatmap = df_confusion.set_index("Trait")

        # Rename index for the heatmap DataFrame to short trait names
        df_heatmap.index = df_heatmap.index.map(trait_short_names)

        # Plot heatmap with dataset-specific colors
        plt.figure(figsize=(9, 4))
        ax = sns.heatmap(df_heatmap, annot=True, cmap=dataset["cmap"], fmt=".1f", linewidths=0.7,
                         annot_kws={"fontsize": 18, "fontweight": "bold"})

        ax.set_xlabel("Metrics(%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Personality Traits", fontsize=14, fontweight="bold")
        # plt.title(f"Heatmap - {dataset['judge']} (Judging {dataset['models']})", fontsize=16, fontweight="bold")

        # Make tick labels bold and increase font size
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, fontweight="bold")

        # Save figure
        filename = f"heatmap_{dataset['judge'].replace(' ', '_')}_{dataset['models'].replace(' ', '_')}.pdf"
        plt.savefig(filename, format="pdf" ,dpi=400, bbox_inches="tight")
        print(f"✅ Heatmap saved as '{filename}'")

    except Exception as e:
        print(f"❌ Error processing {dataset['path']}: {e}")
