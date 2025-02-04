# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:42:39 2025

@author: fgley
"""

import numpy as np
import matplotlib.pyplot as plt 

# Paramètres du problème
R = 0.5  # Rayon du pilier (m)
Ce = 20  # Concentration en surface (mol/m³)
Deff = 1e-10  # Coefficient de diffusion (m²/s)
S = 2e-8  # Terme source constant (mol/m³/s)
N = 5  # Nombre de nœuds
dr = R / (N - 1)  # Pas spatial

# Discrétisation spatiale
r = np.linspace(0, R, N)

# Construction du système linéaire matriciel Ax = b
A = np.zeros((N, N))
b = np.zeros(N)

# Remplissage des équations internes
for i in range(1, N-1):
    A[i, i-1] = 1 / dr**2 - 1 / (r[i] * 2 * dr)
    A[i, i] = -2 / dr**2
    A[i, i+1] = 1 / dr**2 + 1 / (r[i] * 2 * dr)
    b[i] = -S / Deff

# Condition aux limites à r = 0 (symétrie)
A[0, 0] = -3 / dr**2
A[0, 1] = 4 / dr**2
A[0, 2] = -1 / dr**2
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
plt.plot(r, C_analytique, 'r--', label="Solution analytique")
plt.xlabel("Rayon (m)")
plt.ylabel("Concentration (mol/m³)")
plt.legend()
plt.grid()
plt.title("Profil de concentration du sel dans le pilier")
plt.show()
