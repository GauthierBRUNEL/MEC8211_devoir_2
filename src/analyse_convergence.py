import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import os
import re 

# Chargement des données
dossier_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def trouver_fichiers_erreur(dossier):
    fichiers = os.listdir(dossier)
    fichiers_erreur = []
    
    for fichier in fichiers:
        match = re.match(r"erreur_L2_ (\d+) noeuds\.txt", fichier)
        if match:
            X = int(match.group(1))  # Extraction du nombre de nœuds X
            fichiers_erreur.append((X, fichier))
    
    if not fichiers_erreur:
        raise FileNotFoundError(f"Aucun fichier 'erreur_L2_X_noeuds.txt' trouvé dans {dossier}.")
    
    # Trier les fichiers par nombre de noeuds X décroissant (si nécessaire)
    fichiers_erreur.sort(reverse=True, key=lambda x: x[0])  # Trier par X décroissant
    return [os.path.join(dossier, fichier[1]) for fichier in fichiers_erreur]

# Trouver tous les fichiers correspondants
fichiers = trouver_fichiers_erreur(dossier_data)

# Fonction pour charger les données depuis le fichier texte
def charger_donnees(fichier):
    with open(fichier, 'r') as f:
        lignes = f.readlines()
        
        # Extraction des données : erreur et nombre de noeuds
        erreur = float(lignes[0].strip())  # Erreur de discretisation (1ère ligne)
        mesh_size = int(lignes[1].strip())  # Nombre de noeuds (2ème ligne)
        
    return pd.DataFrame({'error_L2': [erreur], 'mesh_size': [mesh_size]})

# Charger les données de chaque fichier dans un DataFrame
donnees = []
for fichier in fichiers:
    data = charger_donnees(fichier)
    donnees.append(data)

# Combiner tous les DataFrames dans un seul
data_combinee = pd.concat(donnees, ignore_index=True)

# Extraire les valeurs dans des variables
mesh_sizes = data_combinee["mesh_size"].values
erreurs = data_combinee["error_L2"].values

# Régression linéaire pour estimer l'ordre de convergence
slope, intercept, r_value, p_value, std_err = linregress(np.log(mesh_sizes), np.log(erreurs))

# Tracé du graphe de convergence avec des axes log
plt.figure(figsize=(8,6))
plt.plot(mesh_sizes, erreurs, 'o-', label=f"Ordre de convergence : {slope:.2f}")
plt.xscale('log')  # Axe des x en échelle logarithmique
plt.yscale('log')  # Axe des y en échelle logarithmique
plt.xlabel("Taille du maillage")
plt.ylabel("Erreur L2")
plt.title("Analyse de convergence")
plt.legend()
plt.grid(True, which="both", ls="--")  # Grille pour les deux axes (logarithmiques)
plt.savefig(os.path.join(dossier_data, "convergence_plot.png"), dpi=300)
plt.show()

# Affichage des résultats
print(f"Ordre de convergence estimé : {slope:.2f}")