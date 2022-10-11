#Code for solving Allen-Cahn equation and calculating error (L2-norm) using Chebyshev Spectral Method.

import numpy as np
import scipy.linalg as spl
import scipy.integrate as ingrt
import matplotlib.pyplot as plt
import scipy.interpolate as intrp
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

#Function to generate Chebyshev nodes
def cheb(N):
    chebp = np.cos(np.pi*np.arange(0,N+1,1)/float(N))
    def c(i):
        if i == 0 or i == N:
            return 2
        else:
            return 1
    coeff = np.array([[float(np.power(-1,i+j)*c(i))/c(j) for j in range(N+1)] for i in range(N+1)])
    X = np.tile(chebp,(N+1,1)).T
    dX = X - X.T
    D = np.true_divide(coeff,dX + np.eye(N+1))
    D = D - np.diag(np.sum(D.T,axis=0))
    return D,chebp

#Initial conditions
n = 64
errorpoint = np.array([])
eps = 0.01
t = 0
dt = 0.01
tmax = 100
#z = np.linspace(-1,1,40)
npoints = int(tmax/dt)
tpoints = np.linspace(0,tmax,npoints)

def f(t,u):
    return eps*np.dot(D2,u)+u-np.power(u,3)

#The following I ran in a loop
rint k
D,x = cheb(k)
D2 = np.dot(D,D)
D2[0,:] = 0.; D2[k,:] = 0.;
v = x #.53*x + .47*np.sin(-1.5*np.pi*x)
#v = np.random.uniform(low = -0.01, high = 0.01, size = (n+1))
sol = ingrt.solve_ivp(f,[0,tmax],v,method='RK45',t_eval= tpoints, rtol=1e-5)
phis = sol.y.T[npoints-1,:]
#phi = intrp.barycentric_interpolate(x,phis,z)
error = spl.norm(np.tanh(x/np.sqrt(2*eps)) - phis)
errorpoint = np.append(errorpoint,error)
    
#Code for Space and time plot
"""
xxx, yyy = np.meshgrid(x,tpoints)
fig=plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(30,210) 
ax.plot_wireframe(xxx,yyy,sol.y.T)
#ax.plot_surface(xxx,yyy,sol.y.T,rstride=1, cstride=1,cmap=cm.Set2)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.savefig('plot/1DACevol32.pdf')
plt.show()
"""

#Code for time plot
"""
plt.plot(z,phi,'-o')
#plt.plot(z,np.tanh(z/np.sqrt(2*eps)))
plt.xlabel('z', size=10)
plt.xticks(size=7)
plt.ylabel(r'$\psi$', size=10)
plt.yticks(size=7)
#plt.savefig('plot/1DAC.pdf')
plt.show()
"""

#Code for error plot

plt.semilogy(np.arange(2,100,1),errorpoint,'-o')
plt.xlabel('n', size=10)
plt.xticks(size=7)
plt.ylabel('Error', size=10)
plt.yticks(size=7)
#plt.savefig('plot/1DACerror.pdf')
plt.show()

