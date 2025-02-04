# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:42:39 2025

@author: fgley
"""

import numpy as np
import matplotlib.pyplot as plt 
import os

# Création des dossiers si inexistants
chemin_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
chemin_resultats = os.path.join(chemin_base, "results")
chemin_data = os.path.join(chemin_base, "data")

# Paramètres du problème
R = 0.5  # Rayon du pilier (m)
Ce = 20  # Concentration en surface (mol/m³)
Deff = 1e-10  # Coefficient de diffusion (m²/s)
S = 2e-8  # Terme source constant (mol/m³/s)
N = 25  # Nombre de nœuds
dr = R / (N - 1)  # Pas spatial

# Discrétisation spatiale
r = np.linspace(0, R, N)

# Construction du système linéaire matriciel Ax = b
A = np.zeros((N, N))
b = np.zeros(N)

# Remplissage des équations internes
for i in range(1, N-1):
    A[i, i-1] = 1 / dr**2
    A[i, i] = -2 / dr**2 - 1/(r[i]*dr)
    A[i, i+1] = 1 / dr**2 + 1 / (r[i] * dr)
    b[i] = S / Deff

# Condition aux limites à r = 0 (symétrie)
A[0, 0] = -1
A[0, 1] = 1
b[0] = 0  # dC/dr = 0 en r=0

# Condition aux limites à r = R (C = Ce)
A[N-1, N-1] = 1
b[N-1] = Ce

# Résolution du système linéaire
C_numerique = np.linalg.solve(A, b)

# Solution analytique
C_analytique = (S / (4 * Deff) * R**2) * ((r**2 / R**2) - 1) + Ce

# Tracé des résultats
plt.figure(figsize=(8, 6))
plt.plot(r, C_numerique, 'o-', label="Solution numérique")
plt.plot(r, C_analytique, 'o-', label="Solution analytique")
plt.xlabel("Rayon (m)")
plt.ylabel("Concentration (mol/m³)")
plt.legend()
plt.grid()
plt.title("Profil de concentration du sel dans le pilier")
plt.show()

# Sauvegarde de la figure dans le bon dossier
plt.savefig(os.path.join(chemin_resultats, f"profil_concentration pour maillage avec {N} noeuds.png"), dpi=300)
plt.show()

# Calcul de l'erreur L2
erreur_L2 = np.sqrt(np.sum((C_numerique - C_analytique) ** 2) * dr)

# Affichage de l'erreur
print(f"Erreur de discrétisation (norme L2) : {erreur_L2:.6e}")

# Sauvegarde de l'erreur dans un fichier texte
with open(os.path.join(chemin_data, f"erreur_L2_ {N} noeuds.txt"), "w") as f:
    f.write(f"{erreur_L2:.6e} \n")
    f.write(f"{N} \n")