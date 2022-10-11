#Code for comparing errors between Chebyshev Spectral Methods and 2nd order Finite difference for the 1D Allen-Cahn solution 

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import scipy.integrate as ingrt
import scipy.linalg as spl
import scipy.optimize as opt
import scipy.interpolate as intrp

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
errorpointint = np.array([])
errorpointfdmerr = np.array([])
errorpointspecerr = np.array([])
errorpointinterr = np.array([])
fdmpoint = np.array([])
specpoint = np.array([])
actp = np.array([])
fdmobj = np.array([])
specobj = np.array([])

maxn = 66
eps = 0.01
t = 0
dt = 0.01
tmax = 100
npoints = int(tmax/dt)
tpoints = np.linspace(0,tmax,npoints)

def f(t,u):
    return eps*np.dot(D2fd,u)+u-np.power(u,3)

def fsp(t,u):
    return eps*np.dot(D2,u)+u-np.power(u,3)

z = np.linspace(-1,1,50)
k = 0

for n in range(4,maxn):
    print n
    D,xspec = cheb(n)
    D2 = np.dot(D,D)    
    D2[0,:] = 0.; D2[n,:] = 0.;
    
    xfdm,h = np.linspace(-1,1,n+1,retstep=True)
    D2fd = (1.0/h**2)*diags([1,-2,1],[-1,0,1],shape=(n+1,n+1)).toarray()
    D2fd[0,:] = 0.; D2fd[n,:] = 0.;
    
    vfdm = .53*xfdm + .47*np.sin(-1.5*np.pi*xfdm) #.53*x + .47*np.sin(-1.5*np.pi*x)
    vspec = .53*xspec+ .47*np.sin(-1.5*np.pi*xspec)
    
    solfdm = ingrt.solve_ivp(f,[0,tmax],vfdm,method='RK45',t_eval= tpoints, rtol=1e-5)
    solspec = ingrt.solve_ivp(fsp,[0,tmax],vspec,method='RK45',t_eval= tpoints, rtol=1e-5)
    phisfdm = solfdm.y.T[npoints-1,:]
    phisspec = solspec.y.T[npoints-1,:]
    #fdmobj = np.append(fdmobj,intrp.interp1d(xfdm,phisfdm, kind='cubic'))
    #specobj = np.append(specobj,intrp.BarycentricInterpolator(xspec,phisspec))
    errfdm = np.tanh(xfdm/np.sqrt(2*eps))-phisfdm
    errorpointfdm = np.append(errorpointfdm,spl.norm(errfdm))
    errorpointfdmerr = np.append(errorpointfdmerr, np.std(errfdm))

    errspec = np.tanh(xspec/np.sqrt(2*eps))-phisspec
    errorpointspec = np.append(errorpointspec,spl.norm(errspec))
    errorpointspecerr = np.append(errorpointspecerr,np.std(errspec))

    errorint = np.tanh(z/np.sqrt(2*eps)) - intrp.BarycentricInterpolator(xspec,phisspec)(z)
    errorpointint = np.append(errorpointint,spl.norm(errorint))
    errorpointinterr = np.append(errorpointinterr,np.std(errorint))
    
    #Comparison of interface length
    """
    act = 2*np.sqrt(2*eps)*np.arctanh(0.9)
    actp = np.append(actp,act)

    phifdm = intrp.interp1d(xfdm,phisfdm, kind='cubic')
    r1 =  opt.brentq(lambda x: phifdm(x) - 0.9 ,-1,1)
    r2 =  opt.brentq(lambda x: phifdm(x) + 0.9 ,-1,1)
    deltafdm = r1-r2
    errorfdm = spl.norm(act - deltafdm)
    errorpointfdm = np.append(errorpointfdm,deltafdm)
    errorpointfdmerr = np.append(errorpointfdmerr, errorfdm)

    phispec = intrp.BarycentricInterpolator(xspec,phisspec)
    deltaspec = opt.root_scalar(lambda x: 0.9 - phispec(x), bracket=[-1,1]).root - opt.root_scalar(lambda x:-0.9 - phispec(x), bracket=[-1,1]).root
    errorspec = spl.norm(act - deltaspec)
    errorpointspec = np.append(errorpointspec,deltaspec)
    errorpointspecerr = np.append(errorpointspecerr,errorspec)
    """
    
#inset plotting

fig, ax1 = plt.subplots()

# These are in unitless percentages of th figure size. (0,0 is bottom left)
left, bottom, width, height = [0.17, 0.3, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.errorbar(np.arange(4,maxn,1),errorpointfdm, yerr = errorpointfdmerr, marker = 'x', ls = '' ,color='red' ,label="2nd order FDM" )
ax1.errorbar(np.arange(4,maxn,1),errorpointspec, yerr = errorpointspecerr, marker = 'o', ls ='', color = 'green', label = "Chebyshev Spectral method")
ax1.errorbar(np.arange(4,maxn,1),errorpointint, yerr = errorpointinterr, marker = '*', ls ='', color = 'black', label = "Barycentric Interpolation")

ax2.plot(xfdm,phisfdm,'-o', color='red')
ax2.plot(xspec,phisspec,'-o', color = 'green')

ax2.plot(xfdm,np.tanh(xfdm/np.sqrt(2*eps)), color='black', label='Analytic solution')

ax1.set_xlabel('n', size=10)
ax1.set_ylabel('Error', size=10)
ax1.set_yscale('log')
ax1.legend(loc = 'best')
ax2.legend(loc='upper left')
#ax1.set_title('Comparison between 2nd order FDM and Chebyshev spectral method for Allen-Cahn equation.')
#plt.savefig('plot/1DACSpFdcomp.pdf',dpi=400)
plt.show()

"""
plt.errorbar(np.arange(4,maxn,1),errorpointfdm, yerr = errorpointfdmerr, marker = 'x', ls = '' ,color='red' ,label="2nd order FDM" )
plt.errorbar(np.arange(4,maxn,1),errorpointspec, yerr = errorpointspecerr, marker = 'o', ls ='', color = 'green', label = "Chebyshev Spectral method",)
plt.plot(np.arange(4,maxn,1),actp, marker = '*', ls = '', color = 'black', label='Actual interface length')
plt.legend(loc='best')
plt.title('Comparison between 2nd order FDM and Chebyshev spectral method for Allen-Cahn equation.')
plt.yscale('log')
plt.show()

meanp = np.array([])
yerr = np.array([])
for m in z:
    point = np.array([fdmobj[0](m),fdmobj[1](m),fdmobj[2](m)])
    print point
    meanp = np.append(meanp,np.mean(point))
    yerr = np.append(yerr,np.std(point))   
   
plt.errorbar(z,meanp,yerr=yerr,marker='o')
plt.plot(z,np.tanh(z/np.sqrt(2*eps)))
plt.show()
"""
