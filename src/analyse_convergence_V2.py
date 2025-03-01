"""
Script d'analyse de convergence des méthodes numériques de diffusion réactive

Ce script permet de :
    - Lire les fichiers de résultats générés par les simulations (méthodes Euler, Crank-Nicholson et la solution analytique C_hat)
    - Calculer les erreurs de convergence (normes L1, L2 et L∞) en comparant les solutions numériques aux solutions analytiques (si disponibles)
    - Tracer et sauvegarder les courbes de convergence spatiale (pour un dt fixe) et temporelle (pour un dr fixe)

Les résultats sont récupérés depuis le dossier "data" et les graphiques générés sont sauvegardés dans le dossier "results".

Auteur : fgley
Date de création : Fri Feb 28 13:21:04 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Définition des chemins
chemin_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
chemin_data = os.path.join(chemin_base, "data")
chemin_resultats = os.path.join(chemin_base, "results")

def lire_donnees(N_list, dt_list):
    data = []
    
    for N in N_list:
        for dt in dt_list:
            for methode, nom_fichier in zip(["Euler", "CrankNicholson", "C_hat"],
                                            [f"Euler_N{N}_dt{dt}.txt", f"CrankNicholson_N{N}_dt{dt}.txt", f"C_hat_N{N}_dt{dt}.txt"]):
                fichier = os.path.join(chemin_data, nom_fichier)
                
                if os.path.exists(fichier):
                    C = np.loadtxt(fichier, delimiter=",")
                    
                    data.append({
                        "N": N,
                        "dt": dt,
                        "methode": methode,
                        "concentration": C
                    })
                else:
                    print(f"Fichier manquant : {fichier}")
    
    return pd.DataFrame(data)

def calculer_erreurs(df):
    erreurs = []
    
    for (N, dt), group in df.groupby(["N", "dt"]):
        C_hat = group[group["methode"] == "C_hat"]["concentration"].values
        
        if len(C_hat) == 0:
            continue
        C_hat = C_hat[0]
        
        for methode in ["Euler", "CrankNicholson"]:
            C_num = group[group["methode"] == methode]["concentration"].values
            
            if len(C_num) == 0:
                continue
            C_num = C_num[0]
            
            dr = 0.5 / (N - 1)  # Calcul de dr avec R = 0.5
            
            erreur_L1 = np.sum(np.abs(C_num - C_hat)) * dr * dt
            erreur_L2 = np.sqrt(np.sum((C_num - C_hat) ** 2) * dr * dt)
            erreur_Linf = np.max(np.abs(C_num - C_hat))  # Pas d'intégration pour Linfini, c'est juste le max

            
            erreurs.append({
                "N": N,
                "dt": dt,
                "dr": dr,
                "methode": methode,
                "L1": erreur_L1,
                "L2": erreur_L2,
                "Linf": erreur_Linf
            })
    
    return pd.DataFrame(erreurs)

import scipy.stats as stats  # Pour la régression linéaire

def tracer_convergence(df_erreurs):
    dr_values = sorted(df_erreurs["dr"].unique())
    dt_values = sorted(df_erreurs["dt"].unique())
    figsize = (12, 6)  # Taille des figures

    # Convergence spatiale (fixe dt)
    for dt in dt_values:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        subset = df_erreurs[df_erreurs["dt"] == dt]
    
        for i, (norme, ax) in enumerate(zip(["L1", "L2", "Linf"], axes)):
            for methode in ["Euler", "CrankNicholson"]:
                valeurs = subset[subset["methode"] == methode].sort_values(by="dr")
    
                # Calcul de la pente pour les deux derniers points
                if len(valeurs) >= 2:
                    x = np.log(valeurs["dr"].values[-2:])
                    y = np.log(valeurs[norme].values[-2:])
                    pente, _, _, _, _ = stats.linregress(x, y)
                    label = f"{methode} - {norme} (pente = {pente:.4f})"
                    
                    # Points utilisés pour la pente (en rouge et carré)
                    ax.loglog(valeurs["dr"].values[-2:], valeurs[norme].values[-2:], 'sr', markersize=8)
                else:
                    label = f"{methode} - {norme}"
    
                # Tracé général
                ax.loglog(valeurs["dr"], valeurs[norme], 'o-', label=label)
    
                ax.set_xlabel("dr (m)")
                ax.set_title(f"Erreur {norme}")
                ax.grid(True, which="both", linestyle="--")
    
            ax.set_ylabel("Erreur") if i == 0 else None
    
        fig.suptitle(f"Convergence spatiale - dt={dt}", fontsize=12)
        fig.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(chemin_resultats, f"Convergence_spatiale_dt{dt}.png"))
        plt.close()

    # Convergence temporelle (fixe dr)
    for dr in dr_values:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        subset = df_erreurs[df_erreurs["dr"] == dr]
    
        for i, (norme, ax) in enumerate(zip(["L1", "L2", "Linf"], axes)):
            for methode in ["Euler", "CrankNicholson"]:
                valeurs = subset[subset["methode"] == methode].sort_values(by="dt")
    
                # Calcul de la pente pour les deux premiers points
                if len(valeurs) >= 2:
                    x = np.log(valeurs["dt"].values[:2])
                    y = np.log(valeurs[norme].values[:2])
                    pente, _, _, _, _ = stats.linregress(x, y)
                    label = f"{methode} - {norme} (pente = {pente:.4f})"
                    
                    # Points utilisés pour la pente (en rouge et carré)
                    ax.loglog(valeurs["dt"].values[:2], valeurs[norme].values[:2], 'sr', markersize=8)
                else:
                    label = f"{methode} - {norme}"
    
                # Tracé général
                ax.loglog(valeurs["dt"], valeurs[norme], 'o-', label=label)
    
                ax.set_xlabel("dt (s)")
                ax.set_title(f"Erreur {norme}")
                ax.grid(True, which="both", linestyle="--")
    
            ax.set_ylabel("Erreur") if i == 0 else None
    
        fig.suptitle(f"Convergence temporelle - dr={dr:.5f}", fontsize=12)
        fig.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(chemin_resultats, f"Convergence_temporelle_dr{dr:.5f}.png"))
        plt.close()


# Exemple d'utilisation
N_list = [4, 5, 6, 7, 8, 9,  10, 11, 20,  25,50,  100, 200, 500]
dt_list = [ 7.884e8, 1.5768e8, 3.1536e7, 1.5768e7,(1.5768e7/6) ] 
df_donnees = lire_donnees(N_list, dt_list)
df_erreurs = calculer_erreurs(df_donnees)
tracer_convergence(df_erreurs)