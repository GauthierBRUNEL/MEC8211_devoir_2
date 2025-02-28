import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import sympy as sp

def construire_matrice(N, dr, dt, Deff, k):
    A = np.zeros((N, N))
    for i in range(1, N - 1):
        r = i * dr
        A[i, i - 1] = dt * Deff * (1 / (2 * r * dr) - 1 / dr**2)
        A[i, i]     = 1 + dt * Deff * (2 / dr**2) + k*dt
        A[i, i + 1] = dt * Deff * (-1 / (2 * r * dr) - 1 / dr**2)
    
    # Condition limite en r = R
    A[N-1, :] = 0
    A[N-1, N-1] = 1
    
    # Condition limite en r = 0 (dérivée nulle)
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    return A

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
    
    
    C_hat_func = None  # Définition par défaut pour éviter l'erreur

    if MMS:
        tf = t_final
        source_func, C_hat_func = calcul_terme_source(Ce, R, Deff, k, tf)
        C = Ce * (np.linspace(0, R, N) ** 2 / R**2)  # Nouvelle condition initiale

    while t < t_final:
        B[:] = C
        if MMS:
            r_values = np.linspace(0, R, N)
            B[1:] += source_func(r_values[1:], t) * dt  # On évite r = 0
        
        B[0] = 0  # Condition de Neumann à r = 0
        B[N-1] = Ce  # Condition de Dirichlet
        
        C = la.solve(A, B)  # Résolution du système
        t += dt
        iterations.append(C.copy())
       
    
    return iterations, C_hat_func


def tracer_concentration(iterations, R, N, MMS, C_hat_func, t_final, dt):
    r_values = np.linspace(0, R, N)
    plt.figure()
    for i, C_iter in enumerate(iterations):
        temps_annees = (i * dt) / (12 * 30 * 24 * 3600)  # Conversion en années
        plt.plot(r_values, C_iter, label=f"t={int(temps_annees)} ans")

        
        if MMS and C_hat_func is not None:
            C_hat_values = C_hat_func(r_values, i * dt)
            plt.plot(r_values, C_hat_values, '--')

    
    plt.xlabel("Rayon (m)")
    plt.ylabel("Concentration (mol/m³)")
    plt.legend()
    plt.grid()
    plt.title("Évolution de la concentration au fil du temps")
    plt.show()

if __name__ == "__main__":
    # Paramètres du problème
    R       = 0.5  # Rayon du pilier (m)
    Ce      = 20  # Concentration en surface (mol/m³)
    Deff    = 1e-10  # Coefficient de diffusion (m²/s)
    k       = 4e-9  # Constante de réaction (s⁻¹)
    
    # Discrétisation
    N       = 50  # Nombre de nœuds spatiaux
    dr = R / (N - 1)
    dt      = 12 * 30 * 24 * 3600 * 12# Pas de temps (1 an en secondes)
    t_final = 12 * 30 * 24 * 3600 * 150  # Temps final (150 ans en secondes)
    
    # Mode MMS activé ou non
    MMS = False  # Passer en mode MMS
    
    # Construction de la matrice et résolution
    A = construire_matrice(N, R / (N - 1), dt, Deff, k)
    iterations, C_hat_func = euler_implicite(A, N, dt, t_final, Ce, R, Deff, k, MMS)
    tracer_concentration(iterations, R, N, MMS, C_hat_func, t_final, dt)
    
    # Affichage du résultat final
    print("Concentration finale:", iterations[-1])
