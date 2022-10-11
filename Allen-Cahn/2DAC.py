#Code for solving the 2D Allen-Cahn Equation using Chebyshev Spectral Method

import numpy as np
import scipy.linalg as spl
import scipy.integrate as ingrt
import scipy.interpolate as intrp
import matplotlib.pyplot as plt
from matplotlib import cm

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
t = 0
dt = 0.01
tmax = 10
z = np.linspace(-1,1,64)
I = np.eye(n+1)
npoints = int(tmax/dt)
eps = 0.001

D,x = cheb(n)
D2 = np.dot(D,D)
D2[0,:] = 0.; D2[n,:] = 0.;    #For Boundary condition
xxx,yyy = np.meshgrid(x,x)
#xxx = np.reshape(xxx.T,(pow(n+1,2),1))
#yyy = np.reshape(yyy.T,(pow(n+1,2),1))
xx,yy = np.meshgrid(z,z)
L = np.kron(D2,I) + np.kron(I,D2) #Laplacian
#v = np.tanh((np.sqrt((xxx-np.pi)**2 + (yyy-np.pi)**2) - 2)/(np.sqrt(2*eps))) #Initial condition
#v = np.random.uniform(low = -0.01, high = 0.01, size = (pow(n+1,2),1)) #Random Inital Conditions
#vold = v - dt*(v - v**3 + eps*np.dot(L,v)) #Using Backward Euler to estimate at t = -dt

#Rearranging the equation in the form dU/dt = eps*K*U + T(U)
Ibar = np.eye(pow(n+1,2))
K = Ibar/eps + L
def T(u):
    return -np.power(u,3)

#Allen-Cahn Equation

def f(t,v):
    return v - v**3 + eps*np.dot(L,v)

#Time Stepping
for j in range(0,1):
    print j
    t = 0
    i = 0
    v = np.random.uniform(low = -0.01, high = 0.01, size = (pow(n+1,2),1))
    phis = np.reshape(v,(n+1,n+1),order='F')
    #phio = intrp.interp2d(x,x,phis,kind='cubic')
    vold = v - dt*(v - v**3 + eps*np.dot(L,v))
    print eps
    with open('data6/2DACdata{0:02}{1:.4f}.txt'.format(j,t),'w') as fp:
        np.savetxt(fp,phis,fmt='%e')

    ax = plt.figure()
    cf = plt.pcolormesh(xxx, yyy, phis, shading='gouraud', cmap=cm.seismic)
    plt.title('Time = {0:.2f}'.format(t))
    cbar = plt.colorbar()
    plt.savefig('image-dump2/img{0:04}.png'.format(i), format='png')
    plt.close()
    
    while t < tmax:
        #Semi-implicit method for time-stepping
        
        A = 3*Ibar - 2*dt*eps*K
        B = 4*v - vold + 2*dt*(2*T(v) - T(vold))
        vnew = np.linalg.solve(A,B)
        vold = v
        v = vnew
        """
        #RK4
        a = dt*f(t,v)
        b = dt*f(t+0.5*dt, v+0.5*a)
        c = dt*f(v+0.5*dt, v+0.5*b)
        d = dt*f(t+dt,     v+c)
        v += (a + 2*b + 2*c + d)/6.0
        """
        t += dt
        i += 1
        if i % 1 == 0:
            print j,t
            phis = np.reshape(v,(n+1,n+1),order='F')
            #phio = intrp.interp2d(x,x,phis,kind='cubic')
            with open('data6/2DACdata{0:02}{1:.4f}.txt'.format(j,t),'w') as fp:
                np.savetxt(fp,phis,fmt='%e')
            
            ax = plt.figure()
            cf = plt.pcolormesh(xxx, yyy, phis, shading='gouraud', cmap=cm.seismic)
            #plt.clim(-1,1)  
            plt.title('Time = {0:.3f}'.format(t))
            cbar = plt.colorbar()
            plt.savefig('image-dump/img{0:04}.png'.format(i), format='png')
            plt.close()
            
