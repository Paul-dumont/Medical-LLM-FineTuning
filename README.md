# Medical-LLM-FineTuning

Dépôt pour le projet de fine-tuning d'un LLM médical.

Contenu:
- `generate_json.py`, `training.py` et fichiers de données d'entraînement.

Note:
- Les fichiers de données brutes (ex: gros fichiers `.xlsm`, `.jsonl`) peuvent être volumineux et ne devraient pas forcément être poussés sur GitHub. Si vous voulez, je peux ajouter des règles `.gitignore` pour les exclure.

Prochaines étapes recommandées:
- Choisir un nom de repo GitHub et visibilité (public/private) si vous voulez que je crée et pousse le repo distant.
- Ou créez le repo manuellement sur GitHub, puis donnez-moi l'URL pour que je l'ajoute en remote et pousse.

Commandes utiles (si vous avez la CLI GitHub `gh` installée):

```bash
# créer et pousser le repo (exemple public)
# Remplacez `my-repo-name` et choisissez --private si nécessaire
gh repo create my-repo-name --public --source=. --remote=origin --push
```

Commandes manuelles si vous préférez le site web GitHub:

```bash
# après création du repo sur GitHub, par exemple https://github.com/USERNAME/REPO.git
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```
