# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:42:39 2025

@author: fgley, gbrun, cflor

Ce script résout numériquement l'équation de diffusion stationnaire d'un soluté 
dans un pilier cylindrique en utilisant la méthode des différences finies. 

Objectif :
- Comparer deux schémas d'approximation des dérivées secondes pour discrétiser 
  l'équation de diffusion.
- Étudier l'influence du nombre de nœuds (N) sur la précision de la solution numérique.
- Comparer la solution numérique obtenue avec la solution analytique.
- Évaluer l'erreur de discrétisation à l'aide des normes L1, L2 et Linf.

Le script génère :
- Des graphiques montrant l'évolution de la concentration du soluté en fonction du rayon.
- Des fichiers de données contenant les erreurs de discrétisation pour chaque valeur de N.
- Une sauvegarde des figures dans un dossier de résultats.

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

# Définir un vecteur de valeurs pour N (nombre de nœuds)
vecteur_N = [5, 10, 20, 50, 100, 200, 300, 500]  # Exemple de valeurs

# Types d'approximation
types_approximation = {
    1: "premier_type",
    2: "second_type"
}

for case, type_approx in types_approximation.items():
    for N in vecteur_N:
        dr = R / (N - 1)  # Pas spatial

        # Discrétisation spatiale (pour la solution numérique)
        r_numerique = np.linspace(0, R, N)

        # Construction du système linéaire matriciel Ax = b
        A = np.zeros((N, N))
        b = np.zeros(N)

        # Remplissage des équations internes
        if case == 1:  # QUESTION C
            for i in range(1, N-1):
                A[i, i-1] = 1 / dr**2
                A[i, i] = -2 / dr**2 - 1/(r_numerique[i]*dr)
                A[i, i+1] = 1 / dr**2 + 1 / (r_numerique[i] * dr)
                b[i] = S / Deff
                
            # Condition aux limites à r = 0 (symétrie)
            A[0, 0] = -1
            A[0, 1] = 1
            b[0] = 0  # dC/dr = 0 en r=0
            
        elif case == 2:  # QUESTION E
            for i in range(1, N-1):
                A[i, i-1] = 1 / dr**2 - 1 / (2*r_numerique[i] * dr)
                A[i, i] = -2 / dr**2
                A[i, i+1] = 1 / dr**2 + 1 / (2*r_numerique[i] * dr)
                b[i] = S / Deff

            # Condition aux limites à r = 0 (symétrie)
            A[0, 0] = -3
            A[0, 1] = 4
            A[0, 2] = -1
            b[0] = 0  # dC/dr = 0 en r=0

        # Condition aux limites à r = R (C = Ce)
        A[N-1, N-1] = 1
        b[N-1] = Ce

        # Résolution du système linéaire
        C_numerique = np.linalg.solve(A, b)

        # Création du maillage combiné pour la solution analytique
        points_par_segment = 5  # Nombre de points entre les points du maillage numérique
        r_analytique = []
        for i in range(len(r_numerique) - 1):
            r_segment = np.linspace(r_numerique[i], r_numerique[i+1], points_par_segment + 1)
            r_analytique.extend(r_segment[:-1])  # Exclure le dernier point pour éviter les doublons
        r_analytique.append(r_numerique[-1])  # Ajouter le dernier point

        r_analytique = np.array(r_analytique)

        # Solution analytique (évaluée sur le vecteur combiné)
        C_analytique = (S / (4 * Deff) * R**2) * ((r_analytique**2 / R**2) - 1) + Ce

        # Tracé des résultats
        plt.figure(figsize=(8, 6))
        plt.plot(r_numerique, C_numerique, 'o', markersize=3, label="Solution numérique")  # Points pour la solution numérique
        plt.plot(r_analytique, C_analytique, '-', linewidth=1, label="Solution analytique")  # Ligne continue pour la solution analytique
        plt.xlabel("Rayon (m)")
        plt.ylabel("Concentration (mol/m³)")
        plt.legend()
        plt.grid()
        plt.title(f"Profil de concentration du sel dans le pilier (N={N}, {type_approx})")
        #plt.show()

        # Sauvegarde de la figure dans le bon dossier
        plt.savefig(os.path.join(chemin_resultats, f"profil_concentration_pour_maillage_avec_{N}_noeuds_{type_approx}.png"), dpi=300)
        plt.close('all')

        # Calcul de l'erreur L1, L2, Linf (sans interpolation !)
        C_analytique_numerique_points = (S / (4 * Deff) * R**2) * ((r_numerique**2 / R**2) - 1) + Ce
        erreur_L2 = np.sqrt(np.sum((C_numerique - C_analytique_numerique_points) ** 2) * dr)
        erreur_L1 = np.sum(np.abs(C_numerique - C_analytique_numerique_points)) * dr
        erreur_Linf = np.max(np.abs(C_numerique - C_analytique_numerique_points))

        # Affichage de l'erreur
        print(f"Erreur de discrétisation (norme L1) pour N={N}, {type_approx} : {erreur_L1:.7e}")
        print(f"Erreur de discrétisation (norme L2) pour N={N}, {type_approx} : {erreur_L2:.7e}")
        print(f"Erreur de discrétisation (norme Linfini) pour N={N}, {type_approx} : {erreur_Linf:.7e}")

        # Sauvegarde de l'erreur dans un fichier texte
        with open(os.path.join(chemin_data, f"erreur_{N}_noeuds_{type_approx}.txt"), "w") as f:
            f.write(f"Erreur L1 : {erreur_L1:.7e} \n")
            f.write(f"Erreur L2 : {erreur_L2:.7e} \n")
            f.write(f"Erreur Linfini : {erreur_Linf:.7e} \n")
            f.write(f"Nombre de nœuds : {N} \n")