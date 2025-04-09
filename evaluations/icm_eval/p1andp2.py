import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
 
# df = pd.read_csv('/media/data_4tb/pranav/llms_gen_eval/evaluations/icm_eval/gpt_fleiss_kappa_input_data_gptvgpt.csv')
#llama
df = pd.read_csv('/media/data_4tb/pranav/llms_gen_eval/evaluations/icm_eval/llama_fleiss_kappa_input_data_gptvgpt.csv') 
#deepseek
# df = pd.read_csv('/media/data_4tb/pranav/llms_gen_eval/evaluations/icm_eval/deepseek_fleiss_kappa_input_data_gptvgpt.csv')
# Split columns into P1 and P2 raters
# Split columns into P1 and P2 raters


p1_columns = [col for col in df.columns if col.endswith('_P1')]
p2_columns = [col for col in df.columns if col.endswith('_P2')]
 
# Calculate Kappa for each trait (domain) and rater group (P1/P2)
traits = df['trait'].unique()
 
print("Fleiss' Kappa per domain and rater group:")
for trait in traits:
    trait_df = df[df['trait'] == trait]
    # Calculate for P1 raters
    agreement_matrix_p1 = []
    for _, row in trait_df.iterrows():
        counts = row[p1_columns].value_counts()
        high = counts.get('High', 0)
        low = counts.get('Low', 0)
        agreement_matrix_p1.append([high, low])
    # Calculate for P2 raters
    agreement_matrix_p2 = []
    for _, row in trait_df.iterrows():
        counts = row[p2_columns].value_counts()
        high = counts.get('High', 0)
        low = counts.get('Low', 0)
        agreement_matrix_p2.append([high, low])
    # Compute Kappa values
    kappa_p1 = fleiss_kappa(np.array(agreement_matrix_p1)) if len(agreement_matrix_p1) > 0 else None
    kappa_p2 = fleiss_kappa(np.array(agreement_matrix_p2)) if len(agreement_matrix_p2) > 0 else None
    print(f"\n- **{trait}**")
    print(f"  P1 Kappa: {kappa_p1:.4f}" if kappa_p1 is not None else "  P1 Kappa: N/A")
    print(f"  P2 Kappa: {kappa_p2:.4f}" if kappa_p2 is not None else "  P2 Kappa: N/A")