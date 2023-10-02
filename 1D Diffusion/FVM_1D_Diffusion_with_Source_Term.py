# Diffusion Equation with source term in 1D using FVM

import numpy as np
import matplotlib.pyplot as plt
# BC
Ta = 100		# K
Tb = 200		# K
k = 0.5			# W/m.K
q = 1000*1000	# W/m^3
A = 1			# m^2
L = 0.02		# m
n_cells = 5		# More cells results in instability (dx would be too small)
dx = L/n_cells

x = np.arange(0,L+(1.5*dx),dx)
x = x-dx/2
x[0] = 0
x[n_cells+1] = L
print(x)

# Initializing
a_W = (k*A/dx)*np.ones(n_cells)
a_E = (k*A/dx)*np.ones(n_cells)
Sp = np.zeros(n_cells)
Su = (q*A*dx)*np.ones(n_cells)

# Boundary conditions
# Left node
a_W[0] = 0
Sp[0] = -2*k*A/dx
Su[0] = q*A*dx + (2*k*A/dx)*Ta

# Right node
a_E[n_cells-1] = 0
Sp[n_cells-1] = -2*k*A/dx
Su[n_cells-1] = q*A*dx + (2*k*A/dx)*Tb

print(Su)
print(Sp)
# Computing a_P
a_P = np.zeros(n_cells)
for i in range(n_cells):	
	a_P[i] = a_W [i]+ a_E[i] - Sp[i]

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
T[0] = Ta
T[n_cells+1] = Tb
T_exact = np.zeros(n_cells+2)
for i in range(0,n_cells+2):
	T_exact[i] = ((Tb-Ta)/L + (q/(2*k))*(L-x[i]))*x[i] + Ta

print('T_exact = ',T_exact)
print('T = ',T)
print('n_iter = ', n_iter)
print('error = ', error)

plt.plot(x,T)
plt.plot(x,T_exact,'*')
plt.xlabel('Domain')
plt.ylabel('Temperature')
plt.legend(['Approximation','Exact Solution'])
plt.title('FVM Solution for 1D Steady Heat Equation')
plt.show()
