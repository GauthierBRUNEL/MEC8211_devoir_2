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
N = 50  # Nombre de nœuds spatiaux
dt = 30 * 24 * 3600  # Pas de temps (1 mois en secondes)
t_final = 12 * dt  # Temps final (1 an en secondes)

r = np.linspace(0, R, N)
dr = R / (N - 1)
C = np.zeros(N)  # Condition initiale : C(r, 0) = 0

# Matrice du schéma d'Euler implicite
A = np.zeros((N, N))
b = np.zeros(N)

# Matrice du schéma de Crank-Nicholson
A_CN = np.zeros((N, N))
B_CN = np.zeros((N, N))

MMS = False

def construire_matrices():
    global A, A_CN, B_CN
    for i in range(1, N-1):
        A[i, i-1] = Deff / dr**2 - Deff / (2 * r[i] * dr)
        A[i, i] = -2 * Deff / dr**2 - k - 1/dt
        A[i, i+1] = Deff / dr**2 + Deff / (2 * r[i] * dr)
        
        A_CN[i, i-1] = -0.5 * Deff / dr**2 + 0.5 * Deff / (2 * r[i] * dr)
        A_CN[i, i] = 1/dt + k + Deff / dr**2
        A_CN[i, i+1] = -0.5 * Deff / dr**2 - 0.5 * Deff / (2 * r[i] * dr)
        
        B_CN[i, i-1] = 0.5 * Deff / dr**2 - 0.5 * Deff / (2 * r[i] * dr)
        B_CN[i, i] = 1/dt - k - Deff / dr**2
        B_CN[i, i+1] = 0.5 * Deff / dr**2 + 0.5 * Deff / (2 * r[i] * dr)
    
    # Condition aux limites : symétrie à r=0
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    A_CN[0, 0] = -3
    A_CN[0, 1] = 4
    A_CN[0, 2] = -1
    
    # Condition aux limites : Dirichlet à r=R
    A[N-1, N-1] = 1
    A_CN[N-1, N-1] = 1

def avancer_temps_euler():
    global C
    plt.figure(figsize=(8,6))
    for t in range(int(t_final / dt)):
        b[1:N-1] = -C[1:N-1] / dt  # Suppression du terme source S, remplacé par k * C
        b[0] = 0  # Condition de symétrie
        b[N-1] = Ce  # Condition de Dirichlet
        
        if MMS == True : 
            S = (Ce / R**2) * (r**2 * np.pi / t_final * np.cos(np.pi * t * dt / t_final) + np.sin(np.pi * t * dt / t_final) * (k * r**2 - 4 * Deff))
            b[1:N-1] += S[1:N-1]  # Ajout du terme source
            b[N-1] = Ce * np.sin(np.pi * t * dt / t_final)  # Nouvelle condition de Dirichlet

        
        C = np.linalg.solve(A, b)  # Résolution du système linéaire
        
        if t % 1 == 0:  # Affichage pour chaque mois
            print(f"Avancement temporel : {t*dt/t_final*100:.2f}%")
            plt.plot(r, C, label=f"Euler t = {t*dt/3600/24:.0f} jours")
    
    plt.xlabel("Rayon (m)")
    plt.ylabel("Concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la concentration avec Euler (1 an)")
    plt.show()

def avancer_temps_crank_nicholson():
    global C
    plt.figure(figsize=(8,6))
    for t in range(int(t_final / dt)):
        b_CN = B_CN @ C
        b_CN[0] = 0  # Condition de symétrie
        b_CN[N-1] = Ce  # Condition de Dirichlet
        
        if MMS == True : 
            S = (Ce / R**2) * (r**2 * np.pi / t_final * np.cos(np.pi * t * dt / t_final) + np.sin(np.pi * t * dt / t_final) * (k * r**2 - 4 * Deff))
            b_CN[1:N-1] += S[1:N-1]  # Ajout du terme source
            b_CN[N-1] = Ce * np.sin(np.pi * t * dt / t_final)  # Nouvelle condition de Dirichlet

        
        C = np.linalg.solve(A_CN, b_CN)  # Résolution du système linéaire
        
        if t % 1 == 0:  # Affichage pour chaque mois
            print(f"Avancement temporel : {t*dt/t_final*100:.2f}%")
            plt.plot(r, C, label=f"Crank-Nicholson t = {t*dt/3600/24:.0f} jours")
    
    plt.xlabel("Rayon (m)")
    plt.ylabel("Concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la concentration avec Crank-Nicholson (1 an)")
    plt.show()

# Exécution
construire_matrices()
avancer_temps_euler()
avancer_temps_crank_nicholson()
