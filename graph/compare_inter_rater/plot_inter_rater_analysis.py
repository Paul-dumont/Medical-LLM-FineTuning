import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
DIR_PATH = '/media/luciacev/Data/Medical-LLM-FineTuning/graph/compare_inter_rater'
FILE_IAA = os.path.join(DIR_PATH, 'inter_annotator_agreement.xlsx')
FILE_MODEL_VS_HUMANS = os.path.join(DIR_PATH, 'model_vs_humans_evaluation.xlsx')

plt.style.use('default')

# ---------------------------------------------------------
# 1. Figure: Inter-Annotator Agreement (Cohen's Kappa)
# ---------------------------------------------------------
print("Loading Inter-Annotator Agreement Data...")
df_iaa = pd.read_excel(FILE_IAA)

def clean_val(val):
    if pd.isna(val) or val == '-':
        return np.nan
    if isinstance(val, str):
        return float(val.replace(',', '.'))
    return float(val)

df_iaa['Support %'] = df_iaa['Support %'].apply(clean_val)
df_iaa['Cohen Kappa'] = df_iaa['Cohen Kappa'].apply(clean_val)

# Clean and sort by Support (frequency) to get the most common features
df_iaa = df_iaa.dropna(subset=['Cohen Kappa'])
df_iaa = df_iaa.sort_values(by='Support %', ascending=False).head(15) # Top 15 features

fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor='white')

# Bar plot for Cohen's Kappa
bars1 = ax1.barh(df_iaa['Feature'][::-1], df_iaa['Cohen Kappa'][::-1], color='#4682b4', edgecolor='black', linewidth=1.2, height=0.6)

ax1.set_title('Inter-Annotator Agreement (Cohen\'s Kappa) - Top 15 Features', fontsize=15, fontweight='bold', pad=15)
ax1.set_xlabel("Cohen's Kappa Score", fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1.1)
ax1.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add text labels on bars
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')

# Guidelines for Kappa Interpretation
ax1.axvline(0.81, color='green', linestyle=':', linewidth=2, label='Almost Perfect (>0.81)')
ax1.axvline(0.61, color='orange', linestyle=':', linewidth=2, label='Substantial (>0.61)')
ax1.legend(loc='upper right')

plt.tight_layout()
out1 = os.path.join(DIR_PATH, 'inter_annotator_kappa.png')
plt.savefig(out1, dpi=300)
print(f"✓ Saved Figure 1: {out1}")

# ---------------------------------------------------------
# 2. Figure: Model Performance vs Humans
# ---------------------------------------------------------
print("Loading Model vs Humans Data...")
df_mvh = pd.read_excel(FILE_MODEL_VS_HUMANS)

df_mvh['Count %'] = df_mvh['Count %'].apply(clean_val)
df_mvh['F1 vs Hum1'] = df_mvh['F1 vs Hum1'].apply(clean_val)
df_mvh['F1 vs Hum2'] = df_mvh['F1 vs Hum2'].apply(clean_val)
df_mvh['F1 vs Either (Soft)'] = df_mvh['F1 vs Either (Soft)'].apply(clean_val)

# Sort by frequency
df_mvh = df_mvh.sort_values(by='Count %', ascending=False).head(15)

fig2, ax2 = plt.subplots(figsize=(14, 9), facecolor='white')

y = np.arange(len(df_mvh))
height = 0.25

# Plotting F1 vs Human 1, Human 2, and Either (Soft)
bars_h1 = ax2.barh(y - height, df_mvh['F1 vs Hum1'][::-1], height, label='Model vs Human 1', color='#1f4e79', edgecolor='black', linewidth=1)
bars_h2 = ax2.barh(y, df_mvh['F1 vs Hum2'][::-1], height, label='Model vs Human 2', color='#4682b4', edgecolor='black', linewidth=1)
bars_soft = ax2.barh(y + height, df_mvh['F1 vs Either (Soft)'][::-1], height, label='Model vs Either Human (Soft Match)', color='#9bc2e6', edgecolor='black', linewidth=1)

ax2.set_title('Model F1-Score compared to Individual Humans & Consensus', fontsize=15, fontweight='bold', pad=15)
ax2.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax2.set_yticks(y)
ax2.set_yticklabels(df_mvh['Feature'][::-1], fontsize=11)
ax2.set_xlim(0, 1.1)

ax2.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=11)

plt.tight_layout()
out2 = os.path.join(DIR_PATH, 'model_vs_humans_f1.png')
plt.savefig(out2, dpi=300)
print(f"✓ Saved Figure 2: {out2}")

print("Done! All figures are ready.")