# Diffusion Equation with source term in 1D using FVM

import numpy as np
import matplotlib.pyplot as plt

"""
***Known data***
Base Temp = 200 deg C
Ambient Temp = 20 deg C
n^2 = hP/(kA) = 25/m^2

Analytical solution
(T-T_amb)/(T_base-T_amb) = cosh(n*(L-x))/cosh(n*L)
"""

# BC
T_amb = 20
T_base = 100		# K
n_sq = 25
L = 1				# m
n_cells = 5			# More cells results in instability (dx would be too small)
dx = L/n_cells

x = np.arange(0, L+(1.5*dx),dx)
x = x-dx/2
x[0] = 0
x[n_cells+1] = L
print('x = ', x)

# Initializing
a_W = (1/dx)*np.ones(n_cells)
a_E = (1/dx)*np.ones(n_cells)
Sp = -n_sq*dx*np.ones(n_cells)
Su = n_sq*dx*T_amb*np.ones(n_cells)

# Boundary conditions
# Left node
a_W[0] = 0
Sp[0] = -n_sq*dx -2/dx
Su[0] = n_sq*dx*T_amb +(2/dx)*T_base

# Right node
a_E[n_cells-1] = 0
Sp[n_cells-1] = -n_sq*dx
Su[n_cells-1] = n_sq*dx*T_amb

print('Su = ', Su)
print('Sp = ', Sp)
print('a_W = ', a_W)
print('a_E = ', a_E)

# Computing a_P
a_P = np.zeros(n_cells)
for i in range(n_cells):	
	a_P[i] = a_W [i]+ a_E[i] - Sp[i]
print('a_P = ', a_P)

# Initializing
T = np.array([0]*(n_cells+2))
T_old = T.copy()
error = 1
tol = 1e-4
n_iter = 0

# Convergence loop
while error>tol:
	# Nodal loop
	for i in range(1,n_cells+1):
		T[i] = (a_W[i-1]*T_old[i-1] + a_E[i-1]*T_old[i+1] + Su[i-1])/a_P[i-1]

	error = np.max(np.abs(T[1:n_cells]-T_old[1:n_cells]))
	n_iter = n_iter+1
	T_old = T.copy()

# Updating boundary surface values
T[0] = T_base
T[n_cells+1] = T[n_cells]

# Analytical Solution
T_exact = np.zeros(n_cells+2)
for i in range(0,n_cells+2):
	T_exact[i] = T_amb + (T_base-T_amb)*np.cosh(np.sqrt(n_sq)*(L-x[i]))/(np.cosh(np.sqrt(n_sq)*L))

print('T_exact = ',T_exact)
print('T = ',T)
print('n_iter = ', n_iter)
print('error = ', error)

plt.plot(x,T)
plt.plot(x,T_exact,'*')
plt.xlabel('Domain')
plt.ylabel('Temperature')
plt.legend(['Approximation','Exact Solution'])
plt.title('FVM Solution for 1D Steady Heat Equation- Heat Transfer along a Fin')
plt.show()
