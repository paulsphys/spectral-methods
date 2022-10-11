#Code to solve the 1D Cahn-Hilliard using Chebyshev Spectral Method

import numpy as np
import scipy.interpolate as intrp
import scipy.integrate as ingrt
import scipy.optimize as opt
import scipy.linalg as lin
import matplotlib.pyplot as plt
import time

#Chebyshev Spectral Derivative
def cheb(N):
    #To obatain x in the interval [a,b] instead of [-1,1] use 0.5*(a+b) + 0.5*(b-a)*np.cos(np.pi*np.arange(0,N+1,1)/float(N))
    #chebp = np.pi + np.pi*np.cos(np.pi*np.arange(0,N+1,1)/float(N))
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
n = 20
eps = 0.2
t = 0
tmax = 0.5
dt = 1e-4

D,x = cheb(n)
D2 = np.dot(D,D)
D2[0,:] = 0.; D2[n,:] = 0
I = np.eye(n+1)
z = np.linspace(-1,1,50)

def f(t,v):
    mu = np.power(v,3) - v - eps*eps*np.dot(D2,v)
    return np.dot(D2,mu)

def fun(y,u,dt):
    return y - dt*f(t,y) - u

def jac(y,u,dt):
    lam = np.diag(3*np.power(y,2))
    return I + dt*eps*eps*np.dot(D2,D2) - dt*np.dot(D2,I-lam)

def prec(y,u,dt):
    return I + dt*eps*eps*np.dot(D2,D2) - 2*dt*D2

#Initial condition
v = np.copy(x)
k = time.time()
#v = np.cos(2*x) + 0.01*np.exp(np.cos(x+0.1))
def RK4(v,t,dt,f):
    a = dt*f(t,v)
    b = dt*f(t-0.5*dt, v+0.5*a)
    c = dt*f(t-0.5*dt, v+0.5*b)
    d = dt*f(t-dt,     v+c)
    vprime = (a + 2*b + 2*c + d)/6.0
    return vprime

while t < tmax:
    
    #Backward Euler for time-stepping
    t += dt
    i = v + f(t,v)*dt  
    sol = opt.root(fun,i,args=(v,dt),method='hybr', jac=jac, tol=1e-8)
    v = sol.x
    """
    #RK4 for time-stepping
    v += RK4(v,t,dt,f)
    t += dt
    """
print time.time() - k
#phi = intrp.BarycentricInterpolator(x,v)(z)
print lin.norm(v-np.tanh(x/(np.sqrt(2)*eps)))
plt.plot(x,v)
plt.plot(x,np.tanh(x/(np.sqrt(2)*eps)), color='black')
plt.show()
