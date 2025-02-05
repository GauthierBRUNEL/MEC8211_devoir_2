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
        match = re.match(r"erreur_ (\d+) noeuds\.txt", fichier)
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

def charger_donnees(fichier):
    with open(fichier, 'r') as f:
        lignes = f.readlines()
        
        # Extraction des erreurs et du nombre de noeuds
        erreur_L1 = float(lignes[0].split(":")[1].strip())  # Prendre la partie après le ":"
        erreur_L2 = float(lignes[1].split(":")[1].strip())  
        erreur_Linf = float(lignes[2].split(":")[1].strip())  
        mesh_size = int(lignes[3].split(":")[1].strip())  

    return pd.DataFrame({'error_L1': [erreur_L1], 'error_L2': [erreur_L2], 'error_Linf': [erreur_Linf], 'mesh_size': [mesh_size]})


# Charger les données de chaque fichier dans un DataFrame
donnees = []
for fichier in fichiers:
    data = charger_donnees(fichier)
    donnees.append(data)

# Combiner tous les DataFrames dans un seul
data_combinee = pd.concat(donnees, ignore_index=True)

# Extraire les valeurs dans des variables
mesh_sizes = data_combinee["mesh_size"].values
# Liste des normes à analyser
normes = ["error_L1", "error_L2", "error_Linf"]

for norme in normes:
    erreurs = data_combinee[norme].values

    # Sélection des points avec mesh_size > 15 pour l'interpolation
    mask = mesh_sizes > 15 # Possibilité de choisir la sélection des points
    mesh_sizes_interp = mesh_sizes[mask]
    erreurs_interp = erreurs[mask]

    # Régression linéaire sur les points sélectionnés uniquement
    if len(mesh_sizes_interp) > 1:  # Vérification pour éviter une erreur si pas assez de points
        slope, intercept, r_value, p_value, std_err = linregress(np.log(mesh_sizes_interp), np.log(erreurs_interp))
        label_slope = f"Ordre de convergence : {slope:.4f}"
    else:
        slope, intercept = None, None
        label_slope = "Pas assez de points pour interpolation"

    # Tracé du graphe de convergence avec des axes log
    plt.figure(figsize=(8,6))
    plt.xscale('log')  # Axe des x en échelle logarithmique
    plt.yscale('log')  # Axe des y en échelle logarithmique

    # Affichage des points non utilisés pour l'interpolation
    plt.plot(mesh_sizes[~mask], erreurs[~mask], 'ro', label="Non utilisé pour interpolation")  # Points en rouge

    # Affichage des points utilisés pour l'interpolation
    plt.plot(mesh_sizes_interp, erreurs_interp, 'bo-', label=label_slope)  # Points en bleu

    plt.xlabel("Taille du maillage")
    plt.ylabel(f"Erreur {norme}")
    plt.title(f"Analyse de convergence ({norme})")
    plt.legend()
    plt.grid(True, which="both", ls="--")  # Grille pour les deux axes (logarithmiques)
    plt.savefig(os.path.join(dossier_data, f"convergence_plot_{norme}.png"), dpi=300)
    plt.show()

    # Affichage des résultats
    if slope is not None:
        print(f"Ordre de convergence estimé pour {norme} : {slope:.4f}")
    else:
        print(f"Pas assez de points avec mesh_size > 15 pour interpoler {norme}")

## PLOT pour toutes les courbes ensemble 

# Couleurs associées aux erreurs
couleurs = {"error_L1": "b", "error_L2": "g", "error_Linf": "r"}
styles = {"error_L1": "o", "error_L2": "s", "error_Linf": "^"}  # Marqueurs différents

plt.figure(figsize=(8,6))
plt.xscale('log')
plt.yscale('log')

for norme in normes:
    erreurs = data_combinee[norme].values
    
    # Séparer les points qui seront utilisés pour l'interpolation
    mask = mesh_sizes > 15
    mesh_sizes_interp = mesh_sizes[mask]
    erreurs_interp = erreurs[mask]

    # Tracer **tous les points**
    plt.scatter(mesh_sizes, erreurs, color=couleurs[norme], marker=styles[norme], label=f"{norme}")

    # Régression et tracé de la droite SEULEMENT pour mesh_size > 15
    if len(mesh_sizes_interp) > 1:
        slope, intercept, _, _, _ = linregress(np.log(mesh_sizes_interp), np.log(erreurs_interp))
        mesh_fit = np.linspace(min(mesh_sizes_interp), max(mesh_sizes_interp), 100)  # Création d'un axe pour la droite
        erreur_fit = np.exp(intercept) * mesh_fit**slope  # Reconstruction de la droite en échelle normale
        plt.plot(mesh_fit, erreur_fit, '--', color=couleurs[norme], label=f"{norme} (ordre: {slope:.4f})")

plt.xlabel("Taille du maillage")
plt.ylabel("Erreurs")
plt.title("Comparaison des erreurs de convergence")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(dossier_data, "convergence_comparative.png"), dpi=300)
plt.show()
