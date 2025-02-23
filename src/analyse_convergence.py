# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 09:28:36 2025

@author: fgley

Script pour calculer et tracer les erreurs entre les solutions numériques et la solution manufacturée.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Chemins des dossiers
chemin_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
chemin_data = os.path.join(chemin_base, "data")
chemin_resultats = os.path.join(chemin_base, "results")

# Paramètres du problème
R = 0.5  # Rayon du pilier (m)
N_list = [5, 10, 25, 50, 100]  # Nombre de nœuds spatiaux
dt_list = [7 * 24 * 3600, 14 * 24 * 3600, 30 * 24 * 3600, 60 * 24 * 3600]  # Pas de temps (1 semaine, 2 semaines, 1 mois, 2 mois)

# Fonction pour calculer les normes d'erreur
def calculer_erreurs(C_num, C_hat, dr):
    erreur_L1 = np.sum(np.abs(C_num - C_hat)) * dr  # Norme L1
    erreur_L2 = np.sqrt(np.sum((C_num - C_hat)**2) * dr)  # Norme L2
    erreur_Linf = np.max(np.abs(C_num - C_hat))  # Norme Linf
    return erreur_L1, erreur_L2, erreur_Linf

# Dictionnaires pour stocker les erreurs
erreurs_euler = {"L1": [], "L2": [], "Linf": []}
erreurs_cn = {"L1": [], "L2": [], "Linf": []}
dr_list = []  # Liste des dr correspondant à N_list

# Boucle pour lire les fichiers et calculer les erreurs
for N in N_list:
    dr = R / (N - 1)  # Calcul de dr
    dr_list.append(dr)
    
    for dt in dt_list:
        # Charger les fichiers
        C_euler = np.loadtxt(os.path.join(chemin_data, f"C_euler_N{N}_dt{dt}.txt"))
        C_crank_nicholson = np.loadtxt(os.path.join(chemin_data, f"C_crank_nicholson_N{N}_dt{dt}.txt"))
        C_hat = np.loadtxt(os.path.join(chemin_data, f"C_hat_N{N}_dt{dt}.txt"))
        
        # Calculer les erreurs pour Euler
        erreur_L1, erreur_L2, erreur_Linf = calculer_erreurs(C_euler[-1, :], C_hat[-1, :], dr)
        erreurs_euler["L1"].append(erreur_L1)
        erreurs_euler["L2"].append(erreur_L2)
        erreurs_euler["Linf"].append(erreur_Linf)
        
        # Calculer les erreurs pour Crank-Nicholson
        erreur_L1, erreur_L2, erreur_Linf = calculer_erreurs(C_crank_nicholson[-1, :], C_hat[-1, :], dr)
        erreurs_cn["L1"].append(erreur_L1)
        erreurs_cn["L2"].append(erreur_L2)
        erreurs_cn["Linf"].append(erreur_Linf)

# Convertir les listes en tableaux numpy pour faciliter les calculs
erreurs_euler = {key: np.array(value) for key, value in erreurs_euler.items()}
erreurs_cn = {key: np.array(value) for key, value in erreurs_cn.items()}
dt_list = np.array(dt_list) / (24 * 3600)  # Convertir dt en jours
dr_list = np.array(dr_list)

# Tracer les erreurs en fonction de dt (convergence temporelle)
plt.figure(figsize=(12, 6))
for i, norme in enumerate(["L1", "L2", "Linf"]):
    plt.subplot(1, 3, i + 1)
    plt.loglog(dt_list, erreurs_euler[norme].reshape(len(N_list), len(dt_list)).T, 'o-', label="Euler")
    plt.loglog(dt_list, erreurs_cn[norme].reshape(len(N_list), len(dt_list)).T, 's-', label="Crank-Nicholson")
    plt.xlabel("dt (jours)")
    plt.ylabel(f"Erreur {norme}")
    plt.title(f"Convergence temporelle ({norme})")
    plt.grid(True, which="both", ls="--")
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(chemin_resultats, "Convergence_temporelle.png"))
# plt.close()

# Tracer les erreurs en fonction de dr (convergence spatiale)
plt.figure(figsize=(12, 6))
for i, norme in enumerate(["L1", "L2", "Linf"]):
    plt.subplot(1, 3, i + 1)
    plt.loglog(dr_list, erreurs_euler[norme].reshape(len(N_list), len(dt_list)).mean(axis=1), 'o-', label="Euler")
    plt.loglog(dr_list, erreurs_cn[norme].reshape(len(N_list), len(dt_list)).mean(axis=1), 's-', label="Crank-Nicholson")
    plt.xlabel("dr (m)")
    plt.ylabel(f"Erreur {norme}")
    plt.title(f"Convergence spatiale ({norme})")
    plt.grid(True, which="both", ls="--")
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(chemin_resultats, "Convergence_spatiale.png"))
# plt.close()