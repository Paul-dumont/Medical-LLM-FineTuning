import matplotlib.pyplot as plt
import pandas as pd

# Données fournies
data = {
    'Categorie': ['TOP 10', 'TOP 20', 'TOP 30', 'TOTAL GLOBAL'],
    'F114 Patients': [0.602, 0.382, 0.263, 0.087],
    '18 Patients': [0.556, 0.333, 0.233, 0.082],
    '22 Patients': [0.618, 0.396, 0.283, 0.101],
    '44 Patients': [0.701, 0.479, 0.354, 0.135]
}

# Création du DataFrame
df = pd.DataFrame(data)

# Correction : Dictionnaire pour associer les colonnes aux valeurs numériques exactes
# F114 correspond maintenant à 14 patients
patient_counts = {
    'F114 Patients': 14,
    '18 Patients': 18,
    '22 Patients': 22,
    '44 Patients': 44
}

plt.figure(figsize=(10, 6))

# Boucle pour tracer chaque ligne
for index, row in df.iterrows():
    label = row['Categorie']
    x_vals = []
    y_vals = []
    
    for col in df.columns:
        if col == 'Categorie':
            continue
        # Récupération de la valeur X correcte
        x_vals.append(patient_counts[col])
        y_vals.append(row[col])
    
    # Tri des points par nombre de patients (axe X)
    sorted_points = sorted(zip(x_vals, y_vals))
    x_sorted = [p[0] for p in sorted_points]
    y_sorted = [p[1] for p in sorted_points]
    
    # Tracé de la ligne avec des marqueurs
    plt.plot(x_sorted, y_sorted, marker='o', label=label)

# Configuration du graphique
plt.xlabel('Augmentation des patients (14, 18, 22, 44)')
plt.ylabel('Score')
plt.title('Score en fonction de l\'augmentation des patients')
plt.legend()
plt.grid(True)

# Affichage
plt.show()