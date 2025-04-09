import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load summary data
df_summary = pd.read_csv("ima_summary.csv")

# Convert data into long format for better visualization
df_stacked = df_summary.melt(id_vars=["topic"], 
                              value_vars=["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"], 
                              var_name="Trait", value_name="Agreement")

# Compute overall mean agreement per topic
df_topic_mean = df_stacked.groupby("topic")["Agreement"].mean()

# Select top 10 and bottom 10 topics
df_top10 = df_topic_mean.nlargest(10).reset_index()
df_bottom10 = df_topic_mean.nsmallest(10).reset_index()

# Shorten topic names for better visualization
df_top10["short_topic"] = ["T"+str(i+1) for i in range(len(df_top10))]
df_bottom10["short_topic"] = ["B"+str(i+1) for i in range(len(df_bottom10))]

# Convert data into long format for stacked visualization
df_selected_stacked_top = df_stacked[df_stacked["topic"].isin(df_top10["topic"])].pivot_table(index="topic", columns="Trait", values="Agreement")
df_selected_stacked_bottom = df_stacked[df_stacked["topic"].isin(df_bottom10["topic"])].pivot_table(index="topic", columns="Trait", values="Agreement")

# Map shortened topic names
df_selected_stacked_top = df_selected_stacked_top.rename(index=dict(zip(df_top10["topic"], df_top10["short_topic"])))
df_selected_stacked_bottom = df_selected_stacked_bottom.rename(index=dict(zip(df_bottom10["topic"], df_bottom10["short_topic"])))

# Set up side-by-side stacked bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

# Plot Top 10 Topics
df_selected_stacked_top.plot(kind="bar", stacked=True, colormap="Blues", ax=axes[0])
axes[0].set_title("Top 10 Topics with Highest IMA")
axes[0].set_xlabel("Topics")
axes[0].set_ylabel("Agreement (%)")
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title="Personality Traits", bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot Bottom 10 Topics
df_selected_stacked_bottom.plot(kind="bar", stacked=True, colormap="Greens", ax=axes[1])
axes[1].set_title("Bottom 10 Topics with Lowest IMA")
axes[1].set_xlabel("Topics")
axes[1].set_ylabel("")
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title="Personality Traits", bbox_to_anchor=(1.05, 1), loc="upper left")

# Adjust layout and save
plt.tight_layout()
plt.savefig("ima_stacked_top_bottom_10_side_by_side.png", dpi=300, bbox_inches="tight")

print("✅ Side-by-side stacked bar chart saved as 'ima_stacked_top_bottom_10_side_by_side.png'.")
