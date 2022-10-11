#Code to solve the Cahn-Hilliard equation in 2D.

import numpy as np
import scipy.interpolate as intrp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg as lin

def newtiter(f,initial,jac,prec,u,dt,tol):
    e = 1
    guess = initial
    i = 0
    while e > tol:
        prevg = guess
        precinv = lin.inv(prec(guess,u,dt))
        A = np.dot(jac(guess,u,dt),precinv)
        B = -f(guess,u,dt)
        y = lin.solve(A,B)
        guess = prevg + lin.solve(prec(guess,u,dt),y)
        e = lin.norm(guess - prevg)
        i += 1
    return guess

def cheb(N):
    chebp = 0.4 + 0.4*np.cos(np.pi*np.arange(0,N+1,1)/float(N))
    #chebp = np.cos(np.pi*np.arange(0,N+1,1)/float(N))
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
n = 16
eps = 0.002
dt = 1e-7
tmax = 0.08
D,x = cheb(n)
D2 = np.dot(D,D)
D2[0,:] = 0.; D2[n,:] = 0
I = np.eye(n+1)
Iprime = np.eye(np.power(n+1,2))
L2 = np.kron(D2,I) + np.kron(I,D2)

xxx,yyy = np.meshgrid(x,x)
xxx = np.reshape(xxx.T,(pow(n+1,2),1))
yyy = np.reshape(yyy.T,(pow(n+1,2),1))
z = np.linspace(0,0.8,40)
xx,yy = np.meshgrid(z,z)

def f(t,v):
    mu = np.power(v,3) - v - eps*np.dot(L2,v)
    return np.dot(L2,mu)

def fun(y,u,dt):
    return y - dt*f(y) - u

def jac(y,u,dt):
    lam = np.diag(3*np.power(y,2))
    return Iprime + dt*eps*np.dot(L2,L2) - dt*np.dot(L2,Iprime-lam)

def prec(y,u,dt):
    return Iprime + dt*eps*np.dot(L2,L2) - 2*dt*L2

#Initial conditions
#v = 0.05*(np.power(np.cos(3*xxx)*np.sin(4*yyy) + np.cos(4*xxx)*np.sin(3*yyy),2) + np.cos(xxx-5*yyy)*np.cos(2*xxx-yyy))
#v = 0.1*np.cos(2*np.pi*xxx/0.8)*np.cos(2*np.pi*yyy/0.8)

t = 0

for j in range(0,5):
    t = 0
    v = np.random.uniform(low = -0.01, high = 0.01, size = (pow(n+1,2),1))
    while t <= tmax:
        
        #Backward Euler for time stepping
        """
        i = v + f(v)*dt
        #v = newtiter(fun,i,jac,prec,v,dt,1e-4)
        sol = opt.root(fun,i,args=(v,dt),method='krylov')
        v = sol.x
        """
        #RK4 for time-stepping
        a = dt*f(t,v)
        b = dt*f(t+0.5*dt, v+0.5*a)
        c = dt*f(v+0.5*dt, v+0.5*b)
        d = dt*f(t+dt,     v+c)
        v += (a + 2*b + 2*c + d)/6.0
    
        t += dt
        print t
        if np.isclose(t,3.0) or np.isclose(t,6.0) or np.isclose(t,9.0) or np.isclose(t,12.0) or np.isclose(t,15.0) or np.isclose(t,18.0) or np.isclose(t,21.0) or np.isclose(t,24.0) or np.isclose(t,27.0) or np.isclose(t,30.0):
            phis = np.reshape(v,(n+1,n+1),order='F')
            phio = intrp.interp2d(x,x,phis,kind='cubic')
            print phio(z,z)
            with open('data2/2DCHdata{0:02}{1:02}.txt'.format(j,t),'w') as fp:
                np.savetxt(fp,phio(z,z),fmt='%e')
    
"""
#Contour plot

plt.figure()
plt.contourf(xx, yy, phio(z,z), 100, cmap=cm.winter)
#plt.clim(-1,1)
plt.colorbar(ticks=[-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

#Profile plot
plt.figure()
plt.plot(z,phio(0.4,z))
plt.ylim([-1,1])
plt.savefig('plot/img{0:04}.svg'.format(t), format='svg', dpi=300)
plt.close()
"""
