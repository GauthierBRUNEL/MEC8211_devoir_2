import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

# Définition des chemins
chemin_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
chemin_data = os.path.join(chemin_base, "data")
chemin_resultats = os.path.join(chemin_base, "results")
os.makedirs(chemin_resultats, exist_ok=True)

# Paramètres du problème
R = 0.5  # Rayon du pilier (m)
N_list = [5, 10, 25, 100, 200, 500]
dt_list = [12 * 30 * 24 * 3600 * 1, 2 * 30 * 24 * 3600 * 10, 2 * 30 * 24 * 3600 * 25, 12 * 30 * 24 * 3600 * 0.5]  # Pas de temps

# Fonction de calcul des erreurs
def calculer_erreurs(C_num, C_hat, dr):
    erreur_L1 = np.sum(np.abs(C_num - C_hat)) * dr
    erreur_L2 = np.sqrt(np.sum((C_num - C_hat)**2) * dr)
    erreur_Linf = np.max(np.abs(C_num - C_hat))
    return erreur_L1, erreur_L2, erreur_Linf

# Initialisation des structures de stockage
erreurs_euler = {"L1": [], "L2": [], "Linf": []}
erreurs_cn = {"L1": [], "L2": [], "Linf": []}
dr_list = []

# Lecture des fichiers et calcul des erreurs
for N in N_list:
    dr = R / (N - 1)
    dr_list.append(dr)
    
    for dt in dt_list:
        C_euler = np.loadtxt(os.path.join(chemin_data, f"Euler_N{N}_dt{dt}.txt"), delimiter=",")
        C_crank_nicholson = np.loadtxt(os.path.join(chemin_data, f"CrankNicholson_N{N}_dt{dt}.txt"), delimiter=",")
        C_hat = np.loadtxt(os.path.join(chemin_data, f"C_hat_N{N}_dt{dt}.txt"), delimiter=",")

        erreur_L1, erreur_L2, erreur_Linf = calculer_erreurs(C_euler[-1, :], C_hat[-1, :], dr)
        erreurs_euler["L1"].append(erreur_L1)
        erreurs_euler["L2"].append(erreur_L2)
        erreurs_euler["Linf"].append(erreur_Linf)

        erreur_L1, erreur_L2, erreur_Linf = calculer_erreurs(C_crank_nicholson[-1, :], C_hat[-1, :], dr)
        erreurs_cn["L1"].append(erreur_L1)
        erreurs_cn["L2"].append(erreur_L2)
        erreurs_cn["Linf"].append(erreur_Linf)

# Conversion en tableaux numpy
erreurs_euler = {key: np.array(value) for key, value in erreurs_euler.items()}
erreurs_cn = {key: np.array(value) for key, value in erreurs_cn.items()}
dt_list_jours = np.array(dt_list) / (24 * 3600)
dr_list = np.array(dr_list)

# Tracé des erreurs en fonction de dr (convergence spatiale)
plt.figure(figsize=(12, 6))
for i, norme in enumerate(["L1", "L2", "Linf"]):
    plt.subplot(1, 3, i + 1)
    
    erreurs_euler_norme = erreurs_euler[norme].reshape(len(N_list), len(dt_list)).mean(axis=1)
    erreurs_cn_norme = erreurs_cn[norme].reshape(len(N_list), len(dt_list)).mean(axis=1)
    
    plt.loglog(dr_list, erreurs_euler_norme, 'o-', label="Euler")
    plt.loglog(dr_list, erreurs_cn_norme, 's-', label="Crank-Nicholson")
    
    plt.xlabel("dr (m)")
    plt.ylabel(f"Erreur {norme}")
    plt.title(f"Convergence spatiale ({norme})")
    plt.grid(True, which="both", ls="--")
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(chemin_resultats, "Convergence_spatiale.png"))
plt.close()

# Calcul des ordres de convergence
# for norme in ["L1", "L2", "Linf"]:
#     erreurs_euler_norme = erreurs_euler[norme].reshape(len(N_list), len(dt_list)).mean(axis=1)
#     erreurs_cn_norme = erreurs_cn[norme].reshape(len(N_list), len(dt_list)).mean(axis=1)

#     # Régression linéaire en échelle log-log
#     slope_euler, intercept_euler, _, _, _ = linregress(np.log(dr_list), np.log(erreurs_euler_norme))
#     slope_cn, intercept_cn, _, _, _ = linregress(np.log(dr_list), np.log(erreurs_cn_norme))

#     print(f"Ordre de convergence (Euler) pour {norme} : {slope_euler:.6f}")
#     print(f"Ordre de convergence (Crank-Nicholson) pour {norme} : {slope_cn:.6f}")

# Tracé des erreurs en fonction de dt (convergence temporelle)
plt.figure(figsize=(12, 6))
for i, norme in enumerate(["L1", "L2", "Linf"]):
    plt.subplot(1, 3, i + 1)

    erreurs_euler_norme = erreurs_euler[norme].reshape(len(N_list), len(dt_list)).T
    erreurs_cn_norme = erreurs_cn[norme].reshape(len(N_list), len(dt_list)).T

    plt.loglog(dt_list_jours, erreurs_euler_norme, 'o-', label="Euler")
    plt.loglog(dt_list_jours, erreurs_cn_norme, 's-', label="Crank-Nicholson")

    plt.xlabel("dt (jours)")
    plt.ylabel(f"Erreur {norme}")
    plt.title(f"Convergence temporelle ({norme})")
    plt.grid(True, which="both", ls="--")
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(chemin_resultats, "Convergence_temporelle.png"))
plt.close()
