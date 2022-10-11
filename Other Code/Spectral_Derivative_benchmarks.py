#Code to calculate the error in taking the derivative of various functions using Chebyshev nodes.

import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
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
    return D, chebp

#Operator list
"""
n = 4
I = np.eye(n+1)
D, x = cheb(n)
D2 = np.dot(D,D)
D3 = np.dot(D,np.dot(D,D))
Dx = np.kron(D,I)
Dy = np.kron(I,D)
D2x = np.kron(D2,I)
D2y = np.kron(I,D2)
DxDy = np.dot(Dx,Dy)
D2xDy = np.dot(D2x,Dy)
DxD2y = np.dot(Dx,D2y)
D2xD2y = np.dot(D2x,D2y)

L = Dx + Dy
L2 = D2x + D2y
xx,yy = np.meshgrid(x,x)
xx = np.reshape(xx.T,(pow(n+1,2),1))
yy = np.reshape(yy.T,(pow(n+1,2),1))
"""

#Error calculation
errorp1 = ([])
errorp2 = ([])
errorp3 = ([])
errorp4 = ([])
errorp5 = ([])
errorp6 = ([])
errorp1b = ([])
errorp2b = ([])
errorp3b = ([])
errorp4b = ([])
errorp5b = ([])
errorp6b = ([])
z = np.linspace(-1,1,20)
for k in range(1,65):
    print k
    I = np.eye(k+1)
    D, x = cheb(k)
    D2 = np.dot(D,D)
    D3 = np.dot(D,np.dot(D,D))
    Dx = np.kron(D,I)
    Dy = np.kron(I,D)
    D2x = np.kron(D2,I)
    D2y = np.kron(I,D2)
    DxDy = np.dot(Dx,Dy)
    D2xDy = np.dot(D2x,Dy)
    DxD2y = np.dot(Dx,D2y)
    D2xD2y = np.dot(D2x,D2y)

    L = Dx + Dy
    L2 = D2x + D2y
    xx,yy = np.meshgrid(x,x)
    xx = np.reshape(xx.T,(pow(k+1,2),1))
    yy = np.reshape(yy.T,(pow(k+1,2),1))

    phi1 = np.dot(D,x/(x**2+1))
    dphi1 = (1-x**2)/(x**2+1)**2
    phi2 = np.dot(D2,x*x*x)
    dphi2 = 6*x
    phi3 = np.dot(L,xx*xx*yy)
    dphi3 = 2*xx*yy + xx*xx
    phi4 = np.dot(L2,xx*xx*yy)
    dphi4 = 2*yy
    phi5 = np.dot(DxDy,xx*xx*yy)
    dphi5 = 2*xx
    phi6 = np.dot(D2xDy,xx*xx*yy*yy)
    dphi6 = 4*yy

    #cdphi1 = intrp.barycentric_interpolate(x,phi1,z)
    error1 = spl.norm(phi1-dphi1)
    errorp1 = np.append(errorp1, error1)
    error2 = spl.norm(phi2-dphi2)
    error2b = spl.norm(phi2-dphi2)
    errorp2 = np.append(errorp2, error2)
    error3 = spl.norm(phi3-dphi3)
    errorp3 = np.append(errorp3, error3)
    error4 = spl.norm(phi4-dphi4)
    errorp4 = np.append(errorp4, error4)
    error5 = spl.norm(phi5-dphi5)
    errorp5 = np.append(errorp5, error5)
    error6 = spl.norm(phi6-dphi6)
    errorp6 = np.append(errorp6, error6)

#Plotting of all the errors as subplots in one plot
"""
plt.figure(1)
plt.subplot(231)
plt.semilogy(np.arange(1,64,1),errorp1,'o')
plt.ylabel('Error')
plt.xlabel('n')
plt.title(r'$\frac{d}{dx}x^3$')
plt.subplot(232)
plt.semilogy(np.arange(1,64,1),errorp2,'o')
plt.ylabel('Error')
plt.xlabel('n')
plt.title(r'$\frac{d^2}{dx^2}x^3$')
plt.subplot(233)
plt.semilogy(np.arange(1,64,1),errorp3,'o')
plt.ylabel('Error')
plt.xlabel('n')
plt.title(r'$\left(\frac{\partial}{\partial x} + \frac{\partial}{\partial y}\right)x^2y$')
plt.subplot(234)
plt.semilogy(np.arange(1,64,1),errorp4,'o')
plt.ylabel('Error')
plt.xlabel('n')
plt.title(r'$\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right)x^2y$')
plt.subplot(235)
plt.semilogy(np.arange(1,64,1),errorp5,'o')
plt.ylabel('Error')
plt.xlabel('n')
plt.title(r'$\left(\frac{\partial^2}{\partial x \partial y}\right)x^2y$')
plt.subplot(236)
plt.semilogy(np.arange(1,64,1),errorp6,'o')
plt.ylabel('Error')
plt.xlabel('n')
plt.title(r'$\left(\frac{\partial^3}{\partial x^2 \partial y}\right)x^2y^2$')
plt.suptitle(r'Convergence of chebyshev spectral derivative') 
plt.show()
"""
#Plotting of single error
plt.plot(np.arange(1,65,1),errorp6,marker='o',ls='')
plt.ylabel('Error')
plt.xlabel('n')
#plt.title(r'$\frac{d}{dx}\left(\frac{x}{x^2+1}\right)$')
#plt.title(r'$\left(\frac{\partial^3}{\partial x^2 \partial y}\right)x^2y^2$', size=16)
plt.yscale('log')
plt.savefig('plot/Derivative/3D_mixedderiv.pdf', dpi=400)
plt.show()
