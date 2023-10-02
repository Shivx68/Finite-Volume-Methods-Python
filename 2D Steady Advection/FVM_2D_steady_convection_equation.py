# 2D steady convection equation

import sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

L = 1
u = 2
v = 2
n_cells = 50

phi = 0.1*np.ones([n_cells,n_cells])

# Boundary Conditions
phi[:,-1] = 100
phi[0,:] = 100
phi[0,-1] = 50
phi[-1,0] = 50

error = 1;
tol = 1e-4;
phi_old = phi.copy()

while error>tol:
	for j in np.arange(1,n_cells-1):
		for i in range(1,n_cells-1):			
			phi[i,j] = (u + v)/(u/phi_old[i-1,j] + v/phi_old[i,j+1])

	error = np.max(np.abs(phi[1:n_cells] - phi_old[1:n_cells]));
	print(error)
	phi_old = phi.copy()


#print(phi)
plt.figure(1)
plt.contourf(phi,20)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('FVM Solution for 2D Steady Convection Equation')
plt.show()
