import numpy as np
import matplotlib.pyplot as plt

# Paramètres
N = 250  # Nombre de points
R = 0.5  # Domaine spatial
S = 2e-8
Deff = 1e-10
Ce = 20

# Génération d'un maillage non uniforme (raffiné en r=0)
t = np.linspace(0, 1, N)  # Vecteur linéaire entre 0 et 1
r = R * (np.exp(t) - 1) / (np.exp(1) - 1)  # Transformation exponentielle, ajustée pour [0, R]
dr = np.diff(r)  # Calcul des pas locaux

# Initialisation des matrices
A = np.zeros((N, N))
b = np.zeros(N)

# Condition de Neumann en r = 0 (dC/dr = 0)
A[0, 0] = 1
A[0, 1] = -1 
b[0] = 0  # Flux nul

# Remplissage du système pour i = 1 à N-2
case = 2

if case == 1 :
    for i in range(1, N-1):
     
        A[i, i-1] = 1/((r[i]-r[i-1])*(r[i+1]-r[i]))
        A[i, i]   = -2/((r[i]-r[i-1])*(r[i+1]-r[i])) - 1/(r[i]*(r[i+1]-r[i]))
        A[i, i+1] = 1/((r[i]-r[i-1])*(r[i+1]-r[i])) + 1/(r[i]*(r[i+1]-r[i]))
        
        b[i] = S/Deff
elif case == 2 : 
    for i in range(1, N-1):
     
        A[i, i-1] = 1/((r[i]-r[i-1])*(r[i+1]-r[i])) - 1/(2*r[i]*(r[i+1]-r[i-1]))
        A[i, i]   = -2/((r[i]-r[i-1])*(r[i+1]-r[i])) 
        A[i, i+1] = 1/((r[i]-r[i-1])*(r[i+1]-r[i])) + 1/(2*r[i]*(r[i+1]-r[i-1]))
        
        b[i] = S/Deff

# Condition de Dirichlet en r = R (C = 20)
A[N-1, N-1] = 1
b[N-1] = Ce

# Résolution du système linéaire
C = np.linalg.solve(A, b)

C_analytique = (S / (4 * Deff) * R**2) * ((r**2 / R**2) - 1) + Ce

# Affichage des résultats
plt.plot(r, C, '^', linewidth=2)
plt.plot(r, C_analytique, 'r-', linewidth=2)
plt.xlabel('r (m)')
plt.ylabel('Concentration C')
plt.title('Distribution de C en fonction de r (maillage non uniforme)')
plt.grid()
plt.show()
