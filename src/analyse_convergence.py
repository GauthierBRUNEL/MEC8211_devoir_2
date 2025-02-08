# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:42:39 2025

@author: fgley, gbrun, cflor

Objectif :
Ce script charge des fichiers contenant des erreurs numériques pour différents maillages
et réalise une analyse de convergence. Il effectue une régression linéaire sur les erreurs
en fonction de la taille du maillage (dr) en échelle logarithmique et génère des graphiques
montrant l'évolution des erreurs (L1, L2, Linf) pour différents types d'approximation.

Résultats produits :
- Calcul des ordres de convergence estimés pour chaque norme d'erreur.
- Graphiques de convergence individuels et comparatifs enregistrés au format PNG.

Auteur : [Ton Nom]
Date : [Date de création ou de dernière modification]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import os
import re

# Chargement des données
dossier_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def trouver_fichiers_erreur(dossier, type_approx):
    fichiers = os.listdir(dossier)
    fichiers_erreur = []

    for fichier in fichiers:
        match = re.match(rf"erreur_(\d+)_noeuds_{type_approx}\.txt", fichier)
        if match:
            X = int(match.group(1))  # Extraction du nombre de nœuds X
            fichiers_erreur.append((X, fichier))

    if not fichiers_erreur:
        raise FileNotFoundError(f"Aucun fichier 'erreur_L2_X_noeuds_{type_approx}.txt' trouvé dans {dossier}.")

    # Trier les fichiers par nombre de noeuds X décroissant (si nécessaire)
    fichiers_erreur.sort(reverse=True, key=lambda x: x[0])  # Trier par X décroissant
    return [os.path.join(dossier, fichier[1]) for fichier in fichiers_erreur]

# Types d'approximation
types_approximation = ["premier_type", "second_type"]

for type_approx in types_approximation:
    # Trouver tous les fichiers correspondants
    fichiers = trouver_fichiers_erreur(dossier_data, type_approx)

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

    # Calcul des tailles de maille (dr)
    R = 0.5  # Rayon du pilier (m)  (Assurez-vous que cette valeur est correcte !)
    drs = R / (mesh_sizes - 1)  # Calcul de dr pour chaque N

    for norme in normes:
        erreurs = data_combinee[norme].values

        # Sélection des points avec mesh_size > 15 pour l'interpolation
        # On travaille maintenant avec dr, donc la condition doit être adaptée.
        # Un dr plus petit signifie un maillage plus fin.  Il faut donc adapter la condition.
        # Par exemple, on pourrait dire que dr doit être inférieur à une certaine valeur.
        mask = drs < (R / 15)   # Sélection des points pour lesquels N > 15 (dr < R/15)

        mesh_sizes_interp = mesh_sizes[mask]  # Conserver (pour le tracé)
        erreurs_interp = erreurs[mask]
        drs_interp = drs[mask]  # Utiliser dr pour la régression

        # Régression linéaire sur les points sélectionnés uniquement
        if len(drs_interp) > 1:  # Vérification pour éviter une erreur si pas assez de points
            # Régression linéaire AVEC dr
            slope, intercept, r_value, p_value, std_err = linregress(np.log(drs_interp), np.log(erreurs_interp))
            label_slope = f"Ordre de convergence : {slope:.6f}"
        else:
            slope, intercept = None, None
            label_slope = "Pas assez de points pour interpolation"

        # Tracé du graphe de convergence avec des axes log
        plt.figure(figsize=(8,6))
        plt.xscale('log')  # Axe des x en échelle logarithmique
        plt.yscale('log')  # Axe des y en échelle logarithmique

        # Affichage des points non utilisés pour l'interpolation
        plt.plot(drs[~mask], erreurs[~mask], 'ro', label="Non utilisé pour interpolation")  # Points en rouge

        # Affichage des points utilisés pour l'interpolation
        plt.plot(drs_interp, erreurs_interp, 'bo-', label=label_slope)  # Points en bleu

        plt.xlabel("Taille du maillage (dr)") # Changer le label
        plt.ylabel(f"Erreur {norme}")
        plt.title(f"Analyse de convergence ({norme}, {type_approx})")
        plt.legend()
        plt.grid(True, which="both", ls="--")  # Grille pour les deux axes (logarithmiques)
        plt.savefig(os.path.join(dossier_data, f"convergence_plot_{norme}_{type_approx}.png"), dpi=300)
        #plt.show()

        # Affichage des résultats
        if slope is not None:
            print(f"Ordre de convergence estimé pour {norme}, {type_approx} : {slope:.6f}")
        else:
            print(f"Pas assez de points avec mesh_size > 15 pour interpoler {norme}, {type_approx}")

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
        mask = drs < (R / 15) # Utiliser dr
        mesh_sizes_interp = mesh_sizes[mask] # Conserver pour tracer (si besoin)
        erreurs_interp = erreurs[mask]
        drs_interp = drs[mask] # Utiliser dr

        # Tracer **tous les points**
        plt.scatter(drs, erreurs, color=couleurs[norme], marker=styles[norme], label=f"{norme}")  # Utiliser dr

        # Régression et tracé de la droite SEULEMENT pour mesh_size > 15
        if len(drs_interp) > 1:
            slope, intercept, _, _, _ = linregress(np.log(drs_interp), np.log(erreurs_interp))  # Utiliser dr
            mesh_fit = np.linspace(min(drs_interp), max(drs_interp), 100)  # Création d'un axe pour la droite
            erreur_fit = np.exp(intercept) * mesh_fit**slope  # Reconstruction de la droite en échelle normale
            plt.plot(mesh_fit, erreur_fit, '--', color=couleurs[norme], label=f"{norme} (ordre: {slope:.6f})")

    plt.xlabel("Taille du maillage (dr)") # Changer le label
    plt.ylabel("Erreurs")
    plt.title(f"Comparaison des erreurs de convergence ({type_approx})")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dossier_data, f"convergence_comparative_{type_approx}.png"), dpi=300)
    plt.show()