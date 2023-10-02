# 1D Steady Convection-Diffusion Equation using Upwind scheme for interpolation

import numpy as np
import matplotlib.pyplot as plt

""" 
Case 1: u = 0.1 m/s;	n_cells = 5
Case 2: u = 2.5 m/s;	n_cells = 5
Case 3: u = 2.5 m/s;	n_cells = 20
"""

u = 0.01;	# m/s
n_cells = 5;
L = 1		# m
rho = 1		# kg/m^3
Gamma = 0.1	# kg/m.s
dx = L/n_cells

# Domain
x = np.arange(0, L+(1.5*dx),dx)
x = x-dx/2
x[0] = 0
x[n_cells+1] = L

# Computing F and D
F = rho*u
D = Gamma/dx

# Initializing and Applying BC for Phi
Phi = np.zeros(n_cells+2);
Phi[0] = 1;
Phi[-1] = 0;

# Initializing coefficients
a_W = (D+F)*np.ones(n_cells)
a_E = (D)*np.ones(n_cells)
Sp = np.zeros(n_cells)
Su = np.zeros(n_cells)

# Applying Boundary Conditions for coefficeints
# First Node
a_W[0] = 0;
Sp[0] = -(2*D+F);
Su[0] = (2*D+F)*Phi[0];

# Last Node
a_E[-1] = 0;
Sp[-1] = -(2*D);
Su[-1] = (2*D)*Phi[-1];

a_P = a_W + a_E - Sp;


# Solving Implicit Equations
error = 1;
tol = 1e-4;
Phi_old = Phi.copy();
n_iter = 0;

while error>tol:
	for i in range(1,n_cells+1):
		Phi[i] = (a_W[i-1]*Phi_old[i-1] + a_E[i-1]*Phi_old[i+1] + Su[i-1])/a_P[i-1];

	error = np.max(np.abs(Phi[1:n_cells] - Phi_old[1:n_cells]));
	Phi_old = Phi.copy();
	n_iter = n_iter+1;

# Analytical Solution
Phi_ana = np.zeros(n_cells+2)
for i in range(n_cells+2):
	Phi_ana[i] = Phi[0] + (Phi[-1] - Phi[0])*(np.exp(F*x[i]/Gamma)-1)/(np.exp(F*L/Gamma)-1);

# Plot
print('F/D ratio = ', F/D)
print('Phi_analytical = ',Phi_ana)
print('Phi = ',Phi)
print('n_iter = ', n_iter)
print('error = ', error)

plt.plot(x,Phi)
plt.plot(x,Phi_ana)
plt.xlabel('Domain')
plt.ylabel('Phi')
plt.legend(['Numerical Solution','Analytical Solution'])
plt.title('FVM Solution for 1D Steady Convection-Diffusion Equation')
plt.show()	