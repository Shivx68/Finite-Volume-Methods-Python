# Diffusion Equation in 1D using FVM

import numpy as np
import matplotlib.pyplot as plt
# BC
Ta = 100
Tb = 500
k = 1000
A = 1e-2
L = 0.5
n_cells = 5
dx = 0.1

x = np.arange(0,L+2*dx,dx)
x = x-dx/2
x[0] = 0
x[6] = L

# Initializing
a_W = [k*A/dx]*n_cells
a_E = [k*A/dx]*n_cells
Sp = [0]*n_cells
Su = [0]*n_cells

# Boundary conditions
# Left node
a_W[0] = 0
Sp[0] = -2*k*A/dx
Su[0] = (2*k*A/dx)*Ta

# Right node
a_E[n_cells-1] = 0
Sp[n_cells-1] = -2*k*A/dx
Su[n_cells-1] = (2*k*A/dx)*Tb

# Computing a_P
a_P = [0]*n_cells
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
# Exact solution
T_exact = 800*x+100

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