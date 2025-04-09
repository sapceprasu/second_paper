import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv("/media/data_4tb/pranav/llms_gen_eval/evaluations/pcm_eval/personality_consistency_gpt4o_vs_mini_with_TCC.csv")

# Melt and reshape data
melted = df.melt(id_vars="Trait", var_name="Metric", value_name="Value")
melted[["Category","Group"]] = melted["Metric"].str.extract(r'(.+)_(P\d+)')
pivot_df = melted.pivot_table(index=["Trait","Group"], columns="Category", values="Value").reset_index()

# Create custom hover text
def create_hover(row):
    return (
        f"<b>{row['Group']}</b><br>"
        f"High Trait Correct: {row['CCH']} ({row['HTA']:.1f}%)<br>"
        f"High Trait Misclassified: {row['MCH']}<br>"
        f"Low Trait Correct: {row['CCL']} ({row['LTA']:.1f}%)<br>"
        f"Low Trait Misclassified: {row['MCL']}<br>"
        f"Total Accuracy (TCC): {row['TCC']:.1f}%"
    )

pivot_df["hover_text"] = pivot_df.apply(create_hover, axis=1)

# Create plot
fig = px.bar(
    pivot_df,
    x="Trait",
    y=["CCH", "MCH", "CCL", "MCL"],
    color_discrete_sequence=["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"],
    facet_col="Group",
    barmode="stack",
    hover_name="Trait",
    custom_data=["hover_text", "HTA", "LTA", "TCC"]
)

# Add TCC reference lines
for i, group in enumerate(["P1", "P2"]):
    fig.add_hline(
        y=df[f"TCC_{group}"].max()*1.1,  # Position above bars
        line_dash="dot",
        line_color="#3498db",
        annotation_text=f"TCC Benchmark",
        row=1,
        col=i+1
    )

# Add percentage annotations
fig.update_traces(
    texttemplate="<b>%{customdata[1]:.1f}%</b>",  # HTA
    textposition="outside",
    selector={"name": "CCH"}
)

fig.update_traces(
    texttemplate="<b>%{customdata[2]:.1f}%</b>",  # LTA
    textposition="outside",
    selector={"name": "CCL"}
)

# Customize layout
fig.update_layout(
    title="Classification Performance by Trait and Group",
    yaxis_title="Count",
    hoverlabel=dict(bgcolor="white"),
    legend_title="Classification",
    uniformtext_minsize=8,
    annotations=[
        dict(
            text="<span style='color:#3498db'>Blue dotted line = Total Classification Accuracy (TCC)</span>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False
        )
    ]
)

fig.show()