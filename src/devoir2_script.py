# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 08:57:48 2025

@author: fgley
"""

import numpy as np
import matplotlib.pyplot as plt

# Paramètres du problème
R = 0.5  # Rayon du pilier (m)
Ce = 20  # Concentration en surface (mol/m³)
Deff = 1e-10  # Coefficient de diffusion (m²/s)
k = 4e-9  # Constante de réaction (s⁻¹)

# Discrétisation
N = 5  # Nombre de nœuds spatiaux
dt = 1e6  # Pas de temps (s)
t_final = 4e9  # Temps final (s)

r = np.linspace(0, R, N)
dr = R / (N - 1)
C = np.zeros(N)  # Condition initiale : C(r, 0) = 0

# Matrice du schéma d'Euler implicite
A = np.zeros((N, N))
b = np.zeros(N)

def construire_matrice():
    global A
    for i in range(1, N-1):
        A[i, i-1] = Deff / dr**2 - Deff / (2 * r[i] * dr)
        A[i, i] = -2 * Deff / dr**2 - k - 1/dt
        A[i, i+1] = Deff / dr**2 + Deff / (2 * r[i] * dr)
    
    # Condition aux limites : symétrie à r=0
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    # Condition aux limites : Dirichlet à r=R
    A[N-1, N-1] = 1

def avancer_temps():
    global C
    plt.figure(figsize=(8,6))
    for t in range(int(t_final / dt)):
        b[1:N-1] = -C[1:N-1] / dt  # Suppression du terme source S, remplacé par k * C
        b[0] = 0  # Condition de symétrie
        b[N-1] = Ce  # Condition de Dirichlet
        
        C = np.linalg.solve(A, b)  # Résolution du système linéaire
        
        if t % 1000 == 0:
            print(f"Avancement temporel : {t*dt/t_final*100:.2f}%")
            plt.plot(r, C, label=f"t = {t*dt:.2e} s")
    
    # Affichage final
    plt.xlabel("Rayon (m)")
    plt.ylabel("Concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la concentration en fonction du temps")
    plt.show()

# Exécution
construire_matrice()
avancer_temps()