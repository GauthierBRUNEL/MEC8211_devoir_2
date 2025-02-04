import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import os

# Chargement des données
dossier_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
chemin_fichier = os.path.join(dossier_data, "convergence.csv")

def charger_donnees(fichier):
    if not os.path.exists(fichier):
        raise FileNotFoundError(f"Le fichier {fichier} est introuvable.")
    
    data = pd.read_csv(fichier)
    return data

data = charger_donnees(chemin_fichier)
mesh_sizes = data["mesh_size"].values
erreurs = data["error_L2"].values

# Transformation logarithmique
log_mesh = np.log(mesh_sizes)
log_erreur = np.log(erreurs)

# Régression linéaire pour estimer l'ordre de convergence
slope, intercept, r_value, p_value, std_err = linregress(log_mesh, log_erreur)

# Tracé du graphe de convergence
plt.figure(figsize=(8,6))
plt.plot(log_mesh, log_erreur, 'o-', label=f"Ordre de convergence : {slope:.2f}")
plt.xlabel("ln(Taille du maillage)")
plt.ylabel("ln(Erreur L2)")
plt.title("Analyse de convergence")
plt.legend()
plt.grid()
plt.savefig(os.path.join(dossier_data, "convergence_plot.png"), dpi=300)
plt.show()

# Affichage des résultats
print(f"Ordre de convergence estimé : {slope:.2f}")