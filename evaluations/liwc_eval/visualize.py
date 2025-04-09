import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb


# liwc_file = "/media/data_4tb/pranav/llms_gen_eval/evaluations/liwc_eval/completed/trait_accuracy_results_fixed.csv"
# accuracy_results= pd.read_csv(liwc_file)


# # Convert results to DataFrame and save
# accuracy_df = pd.DataFrame(accuracy_results)
accuracy_df.to_csv("trait_accuracy_results_fixed_gptvdeepseek.csv")

print("✅ Accuracy calculation completed. Results saved to 'trait_accuracy_results_fixed_gptvdeepseek.csv'.")
print(accuracy_df)

# ====================================
# Step 3: Visualization & Save Figures
# ====================================
fig_size = (10, 6)
bar_colors = ["#9e9ac8", "#66c2a4"]  # Customizable colors

# 📊 Bar Chart - Accuracy Comparison
plt.figure(figsize=fig_size)
accuracy_df.plot(kind="bar", figsize=fig_size, color=bar_colors)
# plt.title("Trait Demonstration Accuracy for Person 1 & 2")
plt.ylabel("Accuracy (%)",fontsize=14, fontweight="bold")
plt.xlabel("Personality Traits",fontsize=14, fontweight="bold")
plt.xticks(rotation=0,fontsize=12, fontweight="bold")
plt.ylim(0, 100)
plt.legend(["Particpant 1", "Particpant 2"], fontsize=14,loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("trait_accuracy_bar_chart_gptvdeepseek.pdf", format="pdf")
plt.show()

# # 🔥 Heatmap - Accuracy Visualization
# plt.figure(figsize=fig_size)
# sns.heatmap(accuracy_df, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5)
# plt.title("Accuracy Heatmap of Trait Predictions")
# plt.xlabel("Prediction Accuracy (%)")
# plt.ylabel("Personality Traits")
# plt.savefig("trait_accuracy_heatmap.pdf", format="pdf")
# plt.show()

print("✅ Figures saved as PDFs: 'trait_accuracy_bar_chart_gptvgpt.pdf' & 'trait_accuracy_heatmap.pdf'.")
