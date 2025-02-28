# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 08:57:48 2025

@author: fgley
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Chemins des dossiers
chemin_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
chemin_resultats = os.path.join(chemin_base, "results")
chemin_data = os.path.join(chemin_base, "data")

# Paramètres du problème
R = 0.5  # Rayon du pilier (m)
Ce = 20  # Concentration en surface (mol/m³)
Deff = 1e-10  # Coefficient de diffusion (m²/s)
k = 4e-9  # Constante de réaction (s⁻¹)

# Discrétisation
N_list = [5]  # Nombre de nœuds spatiaux
dt_list = [12 * 12 * 30 * 24 * 3600]  # Pas de temps : 1 mois, 6 mois, 1 an, 10 ans 
# N_list = [5, 10, 25, 50, 100, 200]  # Nombre de nœuds spatiaux
# dt_list = [6 * 30 * 24 * 3600, 12 * 30 * 24 * 3600, 10 * 12 * 30 * 24 * 3600]  # Pas de temps : 1 mois, 6 mois, 1 an, 10 ans 
t_final = 12 * 30 * 24 * 3600 * 150  # Temps final (150 an en secondes)

Save_Data = True
Save_Plot = True
MMS       = True

#%% Fonctions 

def MMS_Source_S(t,r,dt) : 
    S = (Ce / R**2) * np.exp((t+1)*dt / t_final)*((r**2 + R**2) / t_final  + (k * r**2 - 4 * Deff))
    return S
    
def construire_matrices():
    global A, A_CN, B_CN
    for i in range(1, N-1):
        A[i, i-1] = -dt*Deff / dr**2 + dt*Deff / (2 * r[i] * dr)
        A[i, i] = 2 * dt*Deff / dr**2 + k*dt + 1
        A[i, i+1] = - dt*Deff / dr**2 - dt*Deff / (2 * r[i] * dr)
        
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
    
    if MMS == False :
        C = np.zeros(N)
    else : 
        C = np.full(N, Ce * ((r**2  + R**2)/ R**2))
        
    C_temps = np.zeros((int(t_final / dt), N))  # Stocke toute l'évolution
    plt.figure(figsize=(8,6))
    colormap = cm.viridis 
    colormap2 = cm.inferno 
    
    for t in range(int(t_final / dt)):
        b[1:N-1] = C[1:N-1]
        b[0] = 0  # Condition de symétrie
        b[N-1] = Ce  # Condition de Dirichlet
        
              
        if MMS == True : 
            S = MMS_Source_S(t,r,dt)
            b[1:N-1] += S[1:N-1]*dt  # Ajout du terme source
            b[N-1] = 2*Ce * np.exp(dt *(t+1)/ t_final)  # Nouvelle condition de Dirichlet
            C_hat = Ce*((r**2 + R**2)/R**2)*np.exp(dt*(t+1) / t_final)
            plt.scatter(r, C_hat, label=f"C_hat = {t*dt/3600/24/30 + 1:.0f} mois", color=colormap2(t / int(t_final / dt)))
        
        C = np.linalg.solve(A, b)  # Résolution du système linéaire
        C_temps[t, :] = C  # Stocke l'évolution
        print(C)
        if t % 1 == 0:  # Affichage pour chaque mois
            # print(f"Avancement temporel : {t*dt/t_final*100:.2f}%")
            plt.plot(r, C, label=f"Euler t = {t*dt/3600/24/30 + 1:.0f} mois", color=colormap(t / int(t_final / dt)))
    
    if Save_Plot == True: 
        plt.xlabel("Rayon (m)")
        plt.ylabel("Concentration (mol/m³)")
        plt.legend()
        plt.grid()
        plt.title(f"Évolution de la concentration avec Euler (N={N}, dt={dt//(24*3600)} jours)")
        if Save_Data == True : 
            nom_fichier = f"Euler_N{N}_dt{dt//(24*3600)}j.png"
            plt.savefig(os.path.join(chemin_resultats, nom_fichier))
    plt.close()
        
    return C_temps

def avancer_temps_crank_nicholson():

    if MMS == False :
        C = np.zeros(N)
    else : 
        C = np.full(N, Ce * ((r**2  + R**2)/ R**2))
        
    C_temps = np.zeros((int(t_final / dt), N))  # Stocke toute l'évolution
    plt.figure(figsize=(8,6))
    colormap = cm.viridis 
    colormap2 = cm.inferno 
    for t in range(int(t_final / dt)):
        b_CN = B_CN @ C
        b_CN[0] = 0  # Condition de symétrie
        b_CN[N-1] = Ce  # Condition de Dirichlet
        
        if MMS == True : 
            S = MMS_Source_S(t,r,dt)
            b_CN[1:N-1] += S[1:N-1] # Ajout du terme source
            b_CN[N-1] = 2*Ce * np.exp(dt*(t+1)/ t_final)  # Nouvelle condition de Dirichlet
            
            C_hat = Ce*((r**2 + R**2)/R**2)*np.exp(dt*(t+1) / t_final)
            plt.scatter(r, C_hat, label=f"C_hat = {t*dt/3600/24/30 + 1:.0f} mois", color=colormap2(t / int(t_final / dt)))

        
        C = np.linalg.solve(A_CN, b_CN)  # Résolution du système linéaire
        C_temps[t, :] = C  # Stocke l'évolution
        
        if t % 1 == 0:  # Affichage pour chaque mois
            # print(f"Avancement temporel : {t*dt/t_final*100:.2f}%")
            plt.plot(r, C, label=f"Crank-Nicholson t = {t*dt/3600/24/30 + 1:.0f} mois", color=colormap(t / int(t_final / dt)))
    
    if Save_Plot == True : 
        plt.xlabel("Rayon (m)")
        plt.ylabel("Concentration (mol/m³)")
        plt.legend()
        plt.grid()
        plt.title(f"Évolution de la concentration avec Crank-Nicholson (N={N}, dt={dt//(24*3600)} jours)")
        if Save_Data == True : 
            nom_fichier = f"CrankNicholson_N{N}_dt{dt//(24*3600)}j.png"
            plt.savefig(os.path.join(chemin_resultats, nom_fichier))
    plt.close()

    return C_temps

def MMS_Calcul():
    C_temps = np.zeros((int(t_final / dt), N))
    for t in range(int(t_final / dt)):
            C_hat = Ce*((r**2 + R**2)/R**2)*np.exp(dt*(t+1) / t_final)
            C_temps[t, :] = C_hat 
    return C_temps

#%% Exécution

# Dictionnaire pour stocker les résultats
results = {}

for N in N_list:
    r = np.linspace(0, R, N)
    dr = R / (N - 1)
    C = np.zeros(N)  # Condition initiale : C(r, 0) = 0
    
    # Matrice du schéma d'Euler implicite
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # Matrice du schéma de Crank-Nicholson
    A_CN = np.zeros((N, N))
    B_CN = np.zeros((N, N))
    
    for dt in dt_list:
        # Stocker les résultats sous forme (N, dt)
        key = f"N={N}, dt={dt/(24*3600)}j"
        results[key] = {}
        
        # Construire les matrices pour cette configuration
        construire_matrices()
        
        # Exécuter les calculs
        C_euler = avancer_temps_euler()
        C_crank_nicholson = avancer_temps_crank_nicholson()
        C_hat = MMS_Calcul()
        
        # Sauvegarde des résultats
        results[key]["Euler"] = C_euler
        results[key]["Crank-Nicholson"] = C_crank_nicholson
        results[key]["C_hat"] = C_hat
        
        nom_fichier_euler = f"Euler_N{N}_dt{dt}.txt"
        nom_fichier_cn = f"CrankNicholson_N{N}_dt{dt}.txt"
        nom_fichier_chat = f"C_hat_N{N}_dt{dt}.txt"
    
        np.savetxt(os.path.join(chemin_data, nom_fichier_euler), C_euler, delimiter=",",comments="")
        np.savetxt(os.path.join(chemin_data, nom_fichier_cn), C_crank_nicholson, delimiter=",",  comments="")
        np.savetxt(os.path.join(chemin_data, nom_fichier_chat), C_hat, delimiter=",", comments="")
        # np.savetxt(os.path.join(chemin_data, nom_fichier_euler), C_euler, delimiter=",", header="Évolution temporelle de la concentration (Euler)", comments="")
        # np.savetxt(os.path.join(chemin_data, nom_fichier_cn), C_crank_nicholson, delimiter=",", header="Évolution temporelle de la concentration (Crank-Nicholson)", comments="")
        # np.savetxt(os.path.join(chemin_data, nom_fichier_chat), C_hat, delimiter=",", header="Évolution temporelle de la concentration (C_hat)", comments="")


#%% PLOT DIFF Euler & CN

Diff = False

if Diff == True : 
    # Calcul de la différence entre les deux solutions sur toute la période
    diff = np.abs(C_euler - C_crank_nicholson)  # Matrice (temps, espace)
    
    plt.figure(figsize=(8,6))
    colormap = cm.viridis 
    # Tracer chaque ligne de diff comme une courbe
    for i in range(0, diff.shape[0], max(1, diff.shape[0]//10)):  # Environ 10 courbes espacées
        plt.plot(r, diff[i, :], label=f"dt_{i}", color=colormap(i / diff.shape[0]))
    
    plt.xlabel("Rayon (m)")
    plt.ylabel("Écart de concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la différence entre Euler et Crank-Nicholson")
    plt.show()

#%% PLOT DIFF Euler & CN & MMS

if MMS == True : 
    C_hat=MMS_Calcul()
    
    diff_Euler_MMS = np.abs(C_euler - C_hat)  # Matrice (temps, espace)
    diff_CN_MMS    = np.abs(C_crank_nicholson - C_hat)  # Matrice (temps, espace)
    
    plt.figure(figsize=(8,6))
    colormap = cm.viridis 
    # Tracer chaque ligne de diff comme une courbe
    for i in range(0, diff_Euler_MMS.shape[0], max(1, diff_Euler_MMS.shape[0]//10)):  # Environ 10 courbes espacées
        plt.plot(r, diff_Euler_MMS[i, :], label=f"dt_{i}", color=colormap(i / diff_Euler_MMS.shape[0]))
    
    plt.xlabel("Rayon (m)")
    plt.ylabel("Écart de concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la différence entre Euler et la solution MMS")
    plt.show()
    
    plt.figure(figsize=(8,6))
    colormap = cm.viridis 
    # Tracer chaque ligne de diff comme une courbe
    for i in range(0, diff_CN_MMS.shape[0], max(1, diff_CN_MMS.shape[0]//10)):  # Environ 10 courbes espacées
        plt.plot(r, diff_CN_MMS[i, :], label=f"dt_{i}", color=colormap(i / diff_CN_MMS.shape[0]))
    
    plt.xlabel("Rayon (m)")
    plt.ylabel("Écart de concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la différence entre Crank-Nicholson et la solution MMS")
    plt.show()