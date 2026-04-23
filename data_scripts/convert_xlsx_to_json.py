#!/usr/bin/env python3
"""
Script rapide pour convertir un fichier Excel (.xlsx) en JSON
Usage: python convert_xlsx_to_json.py <input_file.xlsx> [output_file.json]
"""

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Erreur: pandas n'est pas installé. Installez-le avec: pip install pandas openpyxl")
    sys.exit(1)


def convert_xlsx_to_json(input_file, output_file=None):
    """
    Convertit un fichier Excel en JSON
    
    Args:
        input_file: Chemin du fichier Excel
        output_file: Chemin du fichier JSON de sortie (optionnel)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Erreur: Le fichier {input_file} n'existe pas")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.xlsx':
        print(f"Erreur: Le fichier doit être au format .xlsx")
        sys.exit(1)
    
    # Définir le fichier de sortie
    if output_file is None:
        output_file = input_path.stem + ".json"
    
    output_path = Path(output_file)
    
    try:
        # Lire le fichier Excel
        print(f"📖 Lecture du fichier: {input_file}")
        df = pd.read_excel(input_file)
        
        # Convertir en JSON
        print(f"🔄 Conversion en JSON...")
        json_data = df.to_dict(orient='records')
        
        # Nettoyer les valeurs NaN
        json_data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in json_data]
        
        # Sauvegarder le JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Fichier converti avec succès: {output_file}")
        print(f"   - Lignes: {len(json_data)}")
        print(f"   - Colonnes: {len(df.columns)}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_xlsx_to_json.py <input_file.xlsx> [output_file.json]")
        print("\nExemple:")
        print("  python convert_xlsx_to_json.py result_llama_no_prompt_5.xlsx")
        print("  python convert_xlsx_to_json.py result_llama_no_prompt_5.xlsx output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_xlsx_to_json(input_file, output_file)
