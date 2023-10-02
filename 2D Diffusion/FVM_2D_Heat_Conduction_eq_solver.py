# Heat Diffusion Equation in 2D using FVM

import numpy as np
import matplotlib.pyplot as plt

# BC
T_left = 900	# Appears on the bottom in the Contour plot
T_right = 600	# Appears on the top
T_top = 800		# Apprears on the right
T_bottom = 400	# Appears on the left

# Assumming uniform values for the following quantities in both x and y
k = 100
A = 1e-2
L = 1
n_cells = 20
dx = L/n_cells

x = np.arange(0,L+2*dx,dx)
x = x-dx/2
x[0] = 0
x[6] = L
y = np.copy(x)
dy = np.copy(dx)

# Initializing
a_W = [k*A/dx]*np.ones(n_cells)
a_E = [k*A/dx]*np.ones(n_cells)
a_S = [k*A/dy]*np.ones(n_cells)
a_N = [k*A/dy]*np.ones(n_cells)
Sp_x = np.zeros(n_cells)
Su_x = np.zeros(n_cells)
Sp_y = np.copy(Sp_x)
Su_y = np.copy(Sp_y)

# Boundary conditions
# Left boundary
a_W[0] = 0
Sp_x[0] = -2*k*A/dx
Su_x[0] = (2*k*A/dx)*T_left

# Right boundary
a_E[n_cells-1] = 0
Sp_x[n_cells-1] = -2*k*A/dx
Su_x[n_cells-1] = (2*k*A/dx)*T_right

# Bottom boundary
a_S[0] = 0
Sp_y[0] = -2*k*A/dy
Su_y[0] = (2*k*A/dy)*T_bottom

# Top boundary
a_N[n_cells-1] = 0
Sp_y[n_cells-1] = -2*k*A/dy
Su_y[n_cells-1] = (2*k*A/dy)*T_top

#print(Sp_x)
#print(Sp_y)
# Computing a_P
a_P = np.zeros([n_cells,n_cells])
for i in range(n_cells):
	for j in range(n_cells):	
		a_P[i][j] = a_W[i] + a_E[i] - Sp_x[i] + a_N[j] + a_S[j] - Sp_y[j]

#print(a_P)

# Initializing
T = np.zeros([n_cells+2,n_cells+2])
T_old = T.copy()
error = 1
tol = 1e-4
n_iter = 0

# Convergence loop
while error>tol:
	# Nodal loop
	for i in range(1,n_cells+1):
		for j in range(1,n_cells+1):
			T[i][j] = (a_W[i-1]*T_old[i-1][j] + a_E[i-1]*T_old[i+1][j] + Su_x[i-1] + a_S[j-1]*T_old[i][j-1] + a_N[j-1]*T_old[i][j+1] + Su_y[j-1])/a_P[i-1][j-1]

	error = np.max(np.max(np.abs(T[1:n_cells][1:n_cells]-T_old[1:n_cells][1:n_cells])))
	n_iter = n_iter+1
	T_old = T.copy()
#print(T)

# Updating boundary surface values
T[:,0] = T_bottom
T[:,-1] = T_top
T[0,:] = T_left
T[-1,:] = T_right

print('T = ',T)
print('n_iter = ', n_iter)
print('error = ', error)

plt.figure(1)
plt.contourf(T,20)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('FVM Solution for 2D Steady Heat Equation')
plt.show()
