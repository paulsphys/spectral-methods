#Code to compare the accuracy of Chebyshev Spectral Methods and 2nd order finite difference for the solution of the Cahn-Hilliard equation in 1D.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import scipy.integrate as ingrt
import scipy.linalg as spl
import scipy.optimize as opt

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

errorpointspec = np.array([])
errorpointfdm = np.array([])
eps = 0.2
t = 0
dt = 1e-4
tmax = 0.5
npoints = int(tmax/dt)
tpoints = np.linspace(0,tmax,npoints)


def f(t,v):
    mu = np.power(v,3) - v - eps*eps*np.dot(D2fd,v)
    return np.dot(D2fd,mu)

def fsp(t,v):
    mu = np.power(v,3) - v - eps*eps*np.dot(D2,v)
    return np.dot(D2,mu)

def fun(y,u,dt):
    return y - dt*0.5*(f(t,u) +f(t,y)) - u

def funsp(y,u,dt):
    return y - dt*0.5*(fsp(t,u) +fsp(t,y)) - u

def jac(y,u,dt):
    lam = np.diag(3*np.power(y,2))
    return I + 0.5*dt*eps*eps*np.dot(D2fd,D2fd) - 0.5*dt*np.dot(D2fd,I-lam)

def jacsp(y,u,dt):
    lam = np.diag(3*np.power(y,2))
    return I + 0.5*dt*eps*eps*np.dot(D2,D2) - 0.5*dt*np.dot(D2,I-lam)

"""
def BDF4(fun,t,v,u1,u2,u3,dt,jac):
    i = v + (dt/24.0)*(55*f(t,v)-59*f(t,u3)+37*f(t,u2)-9*f(t,u1))
    sol = opt.root(fun,i,args=(u1,u2,u3,v,dt),method='hybr', jac=jac)
    u1 = u2
    u2 = u3
    u3 = v
    v = sol.x
    return (v,u1,u2,u3)    
"""

for n in range(4,65):
    print n
    D,xspec = cheb(n)
    D2 = np.dot(D,D)    
    D2[0,:] = 0.; D2[n,:] = 0.;
    I = np.eye(n+1)
    xfdm,h = np.linspace(-1,1,n+1,retstep=True)
    D2fd = (1.0/h**2)*diags([1,-2,1],[-1,0,1],shape=(n+1,n+1)).toarray()
    D2fd[0,:] = 0.; D2fd[n,:] = 0.;
    
    vfdm = xfdm #.53*x + .47*np.sin(-1.5*np.pi*x)
    u1fdm = xfdm; 
    vspec = xspec
    
    t = 0
    while t < tmax:
        t += dt
        i = vfdm + f(t,vfdm)*dt  
        #u = newtiter(fun,i,jac,prec,u,dt,1e-8)
        sol = opt.root(fun,i,args=(vfdm,dt),method='hybr', jac=jac)
        vfdm = sol.x

        i = vspec + fsp(t,vspec)*dt  
        #u = newtiter(fun,i,jac,prec,u,dt,1e-8)
        sol = opt.root(funsp,i,args=(vspec,dt),method='hybr', jac=jacsp)
        vspec = sol.x
        

    phifdm = vfdm
    phispec = vspec
    
    errorfdm = spl.norm(np.tanh(xfdm/(np.sqrt(2)*eps)) - phifdm)
    errorpointfdm = np.append(errorpointfdm,errorfdm)
    errorspec = spl.norm(np.tanh(xspec/(np.sqrt(2)*eps)) - phispec)   
    errorpointspec = np.append(errorpointspec,errorspec)
    print errorspec, errorfdm

fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.65, 0.6, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.semilogy(np.arange(4,65,1),errorpointfdm,'-o', color='red' ,label="2nd order FDM" )
ax1.semilogy(np.arange(4,65,1),errorpointspec,'-o', color = 'green', label = "Chebyshev Spectral method",)

ax2.plot(xfdm,phifdm,'-o', color='red')
ax2.plot(xspec,phispec,'-x', color = 'green')
ax2.plot(xfdm,np.tanh(xfdm/(np.sqrt(2)*eps)), color='black')

ax1.set_xlabel('n', size=10)
ax1.set_ylabel('Error', size=10)
ax1.legend(loc = 'best')
ax1.set_title('Comparison between 2nd order FDM and Chebyshev spectral method for Cahn-Hilliard equation using Crank Nicholson.')
#plt.savefig('plot/1DAC.pdf')
plt.show()

