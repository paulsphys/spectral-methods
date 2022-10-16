# spectral-methods

Using Chebyshev spectral methods to solve the Allen-Cahn and Cahn-Hilliard equations. Instead of an equispaced grid we consider the Chebyshev points $x_j = cos(j pi/N)$. We implement the Chebyshev method as given in Trefethen. [1] We use it to first solve the Allen-Cahn equation in 1D and 2D: $$\frac{\partial\phi}{\partial t} = \Gamma(a\phi - b\phi^3 + c\nabla^2\phi) \label{ac}$$ with the no flux boundary condition, $\frac{\partial \phi}{\partial \textbf{n}} = 0$ and initial condition, $\phi(x,0) = \phi(x)$. We then proceed to try and solve the Cahn-Hilliard equation $$\frac{\partial\phi}{\partial t} = D\nabla^2(b\phi^3 - a\phi - c\nabla^2\phi) \label{ch}$$ with the same boundary and initial conditions.


[1] Lloyd N. Trefethen, Spectral Methods in MATLAB, Society for Industrial and Applied Mathematics, 2000
