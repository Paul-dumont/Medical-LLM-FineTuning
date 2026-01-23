import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Load JSON data from external file
with open('/media/luciacev/Data/Medical-LLM-FineTuning/graph/features_metrics.json', 'r') as f:
    json_data = json.load(f)

# Filter out entries with null f1_score values
filtered_data = [item for item in json_data if item['f1_score'] is not None]

# Extract data for DataFrame
data = {
    'Feature': [item['feature'] for item in filtered_data],
    'Count_Pct': [int(item['percentage'].rstrip('%')) for item in filtered_data],
    'F1': [item['f1_score'] for item in filtered_data],
    'SEM': [item['sem'] for item in filtered_data]
}

# Check data length before creating DataFrame
print("🔍 Data length verification:")
print(f"  Count_Pct: {len(data['Count_Pct'])} elements")
print(f"  F1: {len(data['F1'])} elements")
print(f"  SEM: {len(data['SEM'])} elements")

# Verify that all lengths are identical
lengths = [len(data['Count_Pct']), len(data['F1']), len(data['SEM'])]
if len(set(lengths)) != 1:
    print(f"\n❌ ERROR: Lengths don't match!")
    print(f"  Count_Pct: {len(data['Count_Pct'])}")
    print(f"  F1: {len(data['F1'])}")
    print(f"  SEM: {len(data['SEM'])}")
    exit(1)
else:
    print(f"\n✅ All data has the same length: {lengths[0]} elements (after filtering nulls)\n")

df = pd.DataFrame(data)
print(len(df))

# Identifier les 10 features les plus fréquentes
top_10_df = df.nlargest(10, 'Count_Pct').reset_index(drop=True)
top_10_indices = df.nlargest(10, 'Count_Pct').index

# Palette de couleurs très distinctes (spectre complet, sans similitudes)
colors_palette = ['#FF0000', "#FF6600", "#740000", '#00CC00', '#0000FF',
                  '#4B0082', '#FF1493', '#00CED1', '#FFD700', "#00310F"]

# Créer les listes de couleurs pour chaque point
colors_f1 = []
colors_sem = []
for idx in df.index:
    if idx in top_10_indices:
        # Trouver la position du point dans le top 10
        position = list(top_10_indices).index(idx)
        colors_f1.append(colors_palette[position])
        colors_sem.append(colors_palette[position])
    else:
        colors_f1.append('#CCCCCC')  # Gris pour les autres
        colors_sem.append('#CCCCCC')

# Create figure with 2 subplots - augmenter la hauteur pour la légende
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: F1 vs Count %
ax1 = axes[0]
ax1.scatter(df['Count_Pct'], df['F1'], alpha=0.7, s=100, c=colors_f1, edgecolors='black', linewidth=0.5)
# Add trend line
z1 = np.polyfit(df['Count_Pct'], df['F1'], 1)
p1 = np.poly1d(z1)
x_smooth = np.linspace(df['Count_Pct'].min(), df['Count_Pct'].max(), 100)
ax1.plot(x_smooth, p1(x_smooth), "darkgray", linewidth=2, label='F1 Trend')
ax1.set_xlabel('Count % (Feature Frequency)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('F1 Score Evolution Based on Feature Frequency', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2, 105)
ax1.set_ylim(-0.05, 1.05)

# Plot 2: SEM vs Count %
ax2 = axes[1]
ax2.scatter(df['Count_Pct'], df['SEM'], alpha=0.7, s=100, c=colors_sem, edgecolors='black', linewidth=0.5)
# Add trend line
z2 = np.polyfit(df['Count_Pct'], df['SEM'], 1)
p2 = np.poly1d(z2)
ax2.plot(x_smooth, p2(x_smooth), "darkgray", linewidth=2, label='SEM Trend')
ax2.set_xlabel('Count % (Feature Frequency)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Semantic Score', fontsize=12, fontweight='bold')
ax2.set_title('Semantic Score Evolution Based on Feature Frequency', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, 105)
ax2.set_ylim(-0.05, 1.05)

# Créer la légende commune en dessous des graphiques
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_palette[i], edgecolor='black', 
                         label=top_10_df.iloc[i]['Feature']) 
                   for i in range(len(top_10_df))]
legend_elements.append(Patch(facecolor='#CCCCCC', edgecolor='black', label='Other Features'))

# Placer la légende en dessous des deux graphiques avec meilleure adaptation
fig.subplots_adjust(bottom=0.25)
fig.legend(handles=legend_elements, fontsize=11, loc='lower center', ncol=6, 
           bbox_to_anchor=(0.5, 0.01), frameon=True, edgecolor='black', fancybox=True)

plt.savefig('/media/luciacev/Data/Medical-LLM-FineTuning/graph/performance_vs_frequency.png', dpi=300)
print("✓ Graph saved: performance_vs_frequency.png")

# Statistics
print("\n📊 Global Statistics:")
print(f"  - Number of features: {len(df)}")
print(f"  - Average F1 Score: {df['F1'].mean():.3f}")
print(f"  - Average Semantic Score: {df['SEM'].mean():.3f}")
print(f"  - Average Count %: {df['Count_Pct'].mean():.1f}%")
print(f"\n  - F1 Score of frequent features (>50%): {df[df['Count_Pct'] > 50]['F1'].mean():.3f}")
print(f"  - SEM of frequent features (>50%): {df[df['Count_Pct'] > 50]['SEM'].mean():.3f}")
print(f"\n  - F1 Score of rare features (<10%): {df[df['Count_Pct'] < 10]['F1'].mean():.3f}")
print(f"  - SEM of rare features (<10%): {df[df['Count_Pct'] < 10]['SEM'].mean():.3f}")

plt.show()
