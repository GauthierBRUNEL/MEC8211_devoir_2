# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 13:21:53 2025

@author: User
"""

import scipy.sparse as spl
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import sympy as sp
import os

#Définition des chemins
chemin_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
chemin_resultats = os.path.join(chemin_base, "results")
chemin_data = os.path.join(chemin_base, "data")

#Création des dossiers si inexistants
os.makedirs(chemin_resultats, exist_ok=True)
os.makedirs(chemin_data, exist_ok=True)

def sauvegarder_resultats(nom_fichier, data):
    chemin_fichier = os.path.join(chemin_data, nom_fichier)
    np.savetxt(chemin_fichier, data, delimiter=',')
    print(f"Résultats sauvegardés dans {chemin_fichier}")

def sauvegarder_plot(nom_fichier):
    chemin_fichier = os.path.join(chemin_resultats, nom_fichier)
    plt.savefig(chemin_fichier)
    print(f"Plot sauvegardé dans {chemin_fichier}")

def construire_matrice(N, dr, dt, Deff, k, schema="euler"):
    A = np.zeros((N, N))
    B = np.zeros((N, N)) if schema == "crank" else None
    
    if schema == "euler":
        for i in range(1, N - 1):
            r = i * dr
            A[i, i - 1] = dt * Deff * (1 / (2 * r * dr) - 1 / dr**2)
            A[i, i]     = 1 + dt * Deff * (2 / dr**2) + k*dt
            A[i, i + 1] = dt * Deff * (-1 / (2 * r * dr) - 1 / dr**2)
    
    elif schema == "crank":
        for i in range(1, N - 1):
            r = i * dr
            alpha = (Deff * dt) / (2 * dr**2)
            beta = (Deff * dt) / (4 * r * dr)
            
            A[i, i - 1] = -alpha + beta
            A[i, i]     = 1 + 2 * alpha + (k * dt) / 2
            A[i, i + 1] = -alpha - beta
            
            B[i, i - 1] = alpha - beta
            B[i, i]     = 1 - 2 * alpha - (k * dt) / 2
            B[i, i + 1] = alpha + beta
    
    # Condition limite en r = R (Dirichlet)
    A[N-1, :] = 0
    A[N-1, N-1] = 1
    if schema == "crank":
        B[N-1, :] = 0
        B[N-1, N-1] = 1
    
    # Condition limite en r = 0 (Neumann)
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    if schema == "crank":
        B[0, 0] = -3
        B[0, 1] = 4
        B[0, 2] = -1
    
    return (A, B) if schema == "crank" else A


def rendre_matrice_creuse(A):
    """ Transforme la matrice A en format creux (CSR) pour une meilleure efficacité numérique. """
    A_sparse = spl.csr_matrix(A)  # Conversion en matrice creuse au format CSR
    return A_sparse

def preconditionner_matrice(A_sparse):
    """ Crée un préconditionneur diagonal basé sur la matrice A_sparse """
    diagonale = A_sparse.diagonal()
    diagonale[diagonale == 0] = 1  # Évite les divisions par zéro
    M_inv = spl.diags(1 / diagonale)  # Matrice diagonale inverse
    return M_inv

def calcul_terme_source(Ce, R, Deff, k, tf):
    t, r = sp.symbols('t r')
    C_hat = Ce * ((1 - sp.exp(-t/tf)) * (1 - r**2/R**2) + r**2/R**2)
    
    dCdt = sp.diff(C_hat, t)
    dCdr = sp.diff(C_hat, r)
    d2Cdr2 = sp.diff(dCdr, r)
    
    source = dCdt - Deff * (1/r * dCdr + d2Cdr2) + k * C_hat
    
    return sp.lambdify((r, t), source, 'numpy'), sp.lambdify((r, t), C_hat, 'numpy')



def euler_implicite(A, N, dt, t_final, Ce, R, Deff, k, MMS):
    C = np.zeros(N)
    B = np.zeros(N)
    t = 0
    iterations = []

    if MMS:
        tf = t_final
        source_func, C_hat_func = calcul_terme_source(Ce, R, Deff, k, tf)
        C = Ce * (np.linspace(0, R, N) ** 2 / R**2)
    
    A_sparse = rendre_matrice_creuse(A)
    M_inv = preconditionner_matrice(A_sparse)  

    while t < t_final:
        B[:] = C
        if MMS:
            r_values = np.linspace(0, R, N)
            B[1:] += source_func(r_values[1:], t) * dt

        B[0] = 0  
        B[N-1] = Ce  

        # Résolution avec préconditionneur
        C, exit_code = spla.bicgstab(A_sparse, B, M=M_inv)

        if exit_code != 0:
            print(f" Problème de convergence pour t={t/(12*30*24*3600):.1f} ans")

        t += dt
        iterations.append(C.copy())

    return iterations, C_hat_func

def crank_nicholson(A, B, N, dt, t_final, Ce, R, Deff, k, MMS):
    C = np.zeros(N)
    B_vector = np.zeros(N)
    t = 0
    iterations = []

    if MMS:
        tf = t_final
        source_func, C_hat_func = calcul_terme_source(Ce, R, Deff, k, tf)
        C = Ce * (np.linspace(0, R, N) ** 2 / R**2)
    
    A_sparse = rendre_matrice_creuse(A)
    M_inv = preconditionner_matrice(A_sparse)

    while t < t_final:
        B_vector[:] = C
        if MMS:
            r_values = np.linspace(0, R, N)
            B_vector[1:] += source_func(r_values[1:], t) * dt / 2 + source_func(r_values[1:], t-dt) * dt / 2

        B_vector[0] = 0
        B_vector[N-1] = Ce

        # Résolution avec préconditionneur
        C, exit_code = spla.bicgstab(A_sparse, B_vector, M=M_inv)

        if exit_code != 0:
            print(f" Problème de convergence pour t={t/(12*30*24*3600):.1f} ans")

        t += dt
        iterations.append(C.copy())

    return iterations

def tracer_concentration(iterations, R, N, MMS, C_hat_func, t_final, dt, methode):
    r_values = np.linspace(0, R, N)
    plt.figure()
    for i, C_iter in enumerate(iterations):
        temps_annees = ((i+1) * dt) / (12 * 30 * 24 * 3600)  # Conversion en années
        plt.plot(r_values, C_iter)
        plt.plot(r_values, C_iter, label=f"t={int(temps_annees)} ans")

        
        if MMS and C_hat_func is not None:
            C_hat_values = C_hat_func(r_values, i * dt)
            plt.plot(r_values, C_hat_values, '--')

    
    plt.xlabel("Rayon (m)")
    plt.ylabel("Concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la concentration au fil du temps")
    
    # Sauvegarde de l'image
    nom_fichier = f"{methode}_N{N}_dt{dt}.png"
    sauvegarder_plot(nom_fichier)

    plt.close()  # Fermer la figure pour éviter d'empiler trop de plots en mémoire
    
 
    
if __name__ == "__main__":
    # Paramètres du problème
    R       = 0.5  # Rayon du pilier (m)
    Ce      = 20  # Concentration en surface (mol/m³)
    Deff    = 1e-10  # Coefficient de diffusion (m²/s)
    k       = 4e-9  # Constante de réaction (s⁻¹)
    t_final = 12 * 30 * 24 * 3600 * 150  # Temps final (150 ans en secondes)
    
    # Mode MMS activé ou non
    MMS = True  # Passer en mode MMS
    
    # Listes de valeurs à tester
    N_list = [4, 5, 6, 7, 8, 9,  10, 11, 20,  25,50,  100, 200, 500]
    dt_list = [3.1536e8, 7.884e8, 1.5768e8, 3.1536e7, 1.5768e7,(1.5768e7/6) ] #10 ans, 25 ans, 5 ans, 1 ans, 6 mois 

    for N in N_list:
        dr = R / (N - 1)
        for dt in dt_list:
            print(f"Test avec N={N}, dt={dt/(12*30*24*3600)} ans")
            dr = R / (N - 1)
            # Euler implicite
            A = construire_matrice(N, dr, dt, Deff, k)
            eigvals = np.linalg.eigvals(A)
            #print(f"Valeurs propres de A (Euler) pour N={N} :", eigvals)
            print(f"Conditionnement de A (Euler) pour N={N} : {np.linalg.cond(A):.2e}")
            iterations_euler, C_hat_func = euler_implicite(A, N, dt, t_final, Ce, R, Deff, k, MMS)
            tracer_concentration(iterations_euler, R, N, MMS, C_hat_func, t_final, dt, "Euler")
            nom_fichier_euler = f"Euler_N{N}_dt{dt}.txt"
            sauvegarder_resultats(nom_fichier_euler, iterations_euler)
            
            # Crank-Nicholson
            A_crank, B_crank = construire_matrice(N, dr, dt, Deff, k, schema="crank")
            iterations_crank = crank_nicholson(A_crank, B_crank, N, dt, t_final, Ce, R, Deff, k, MMS)
            tracer_concentration(iterations_crank, R, N, MMS, C_hat_func, t_final, dt, "CrankNicholson")
            print(f"Shape de iterations_crank: {np.array(iterations_crank).shape}")
            nom_fichier_cn = f"CrankNicholson_N{N}_dt{dt}.txt"
            sauvegarder_resultats(nom_fichier_cn, iterations_crank)
            
            # Solution analytique si MMS activé
            if MMS:
                r_values = np.linspace(0, R, N)
                C_hat_values = np.array([C_hat_func(r_values, i * dt) for i in range(len(iterations_euler))])
                nom_fichier_chat = f"C_hat_N{N}_dt{dt}.txt"
                sauvegarder_resultats(nom_fichier_chat, C_hat_values)
    
    print("Simulation terminée.")