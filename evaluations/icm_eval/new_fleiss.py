import pandas as pd

import numpy as np

from statsmodels.stats.inter_rater import fleiss_kappa
 
# Load the CSV file
# gpt
df = pd.read_csv('/media/data_4tb/pranav/llms_gen_eval/evaluations/icm_eval/gpt_fleiss_kappa_input_data_gptvgpt.csv')
#llama
# df = pd.read_csv('/media/data_4tb/pranav/llms_gen_eval/evaluations/icm_eval/llama_fleiss_kappa_input_data_gptvgpt.csv') 
#deepseek
# df = pd.read_csv('/media/data_4tb/pranav/llms_gen_eval/evaluations/icm_eval/deepseek_fleiss_kappa_input_data_gptvgpt.csv')
# Split columns into P1 and P2 raters

p1_columns = [col for col in df.columns if col.endswith('_P1')]

p2_columns = [col for col in df.columns if col.endswith('_P2')]
 
def compute_fleiss_kappa(rater_columns):

    agreement_matrix = []

    for _, row in df.iterrows():

        counts = row[rater_columns].value_counts()

        high = counts.get('High', 0)

        low = counts.get('Low', 0)

        agreement_matrix.append([high, low])

    return fleiss_kappa(np.array(agreement_matrix))
 
# Calculate Kappa for P1 and P2 separately

kappa_p1 = compute_fleiss_kappa(p1_columns)

kappa_p2 = compute_fleiss_kappa(p2_columns)

print(f"Fleiss' Kappa for P1 raters: {kappa_p1:.4f}")

print(f"Fleiss' Kappa for P2 raters: {kappa_p2:.4f}\n")
 
# Calculate Kappa for each trait (domain) using all raters (P1 + P2)

traits = df['trait'].unique()

all_rater_columns = p1_columns + p2_columns
 
print("Fleiss' Kappa per domain:")

for trait in traits:

    trait_df = df[df['trait'] == trait]

    agreement_matrix = []

    for _, row in trait_df.iterrows():

        counts = row[all_rater_columns].value_counts()

        high = counts.get('High', 0)

        low = counts.get('Low', 0)

        agreement_matrix.append([high, low])

    if len(agreement_matrix) > 0:

        kappa = fleiss_kappa(np.array(agreement_matrix))

        print(f"- {trait}: {kappa:.4f}")
 