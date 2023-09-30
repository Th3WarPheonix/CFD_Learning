
import numpy as np
import matplotlib.pyplot as plt
from Method_DB import solve_tridiagonal
import time

def stretch_mesh(dymin, ymax, numpts):
    """
    Implements Rakich Stretching function on pg 78 (section 5.3.1).
    Function returns points assuming a stretch from 0 to ymax

    Parameters
    ----------
    dymin : smallest y2-y1 distance from which point spacing will only
    increase
    ymax : distnace over which the stretch occurs
    numpts : number of points in stretch
    """
    ys = np.zeros(numpts)

    kold = 0
    knew = 1
    a = 1/(numpts-1)
    while abs(kold - knew) > 1e-6:
        kold = knew
        fcn = dymin - ymax*(np.exp(kold*a)-1)/(np.exp(kold)-1)
        dfcn = -ymax*((a-1)*np.exp(a*kold+kold)-a*np.exp(a*kold)+np.exp(kold))/(np.exp(kold)-1)**2
        knew = kold - fcn/dfcn

    for j in range(1, numpts):
        a = (j+1-1)/(numpts-1)
        ys[j] = ys[0] + ymax*(np.exp(kold*a)-1)/(np.exp(kold)-1)

    return ys

def point_jacobi(time_steps, Minf, xpts, ypts, phi, phi2, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=False):
    bigA = 1 - Minf**2

    startjac = time.time()
    residualjac = np.ones(time_steps)
    res = np.zeros_like(phi)
    for t in range(time_steps):
        
        for i in range(1, len(xpts)-1): # move across columns
            for j in range(1, len(ypts)-1): # move through columns
                help1a = bigA/(xpts[i+1]-xpts[i-1])*(1/(xpts[i+1] - xpts[i]) + (1/(xpts[i] - xpts[i-1])))
                help1b = 1/(ypts[j+1]-ypts[j-1])*(1/(ypts[j+1] - ypts[j]) + (1/(ypts[j] - ypts[j-1])))
                help1 = 1/(2*(help1a + help1b))
                help2 = bigA*(phi[j, i+1] + phi[j, i-1])/((xpts[i] - xpts[i-1])*(xpts[i+1] - xpts[i]))
                help3 = (phi[j+1, i] + phi[j-1, i])/((ypts[j] - ypts[j-1])*(ypts[j+1] - ypts[j]))
                phi2[j, i] = help1*(help2+help3)  
        
        # Zone 1 bottom boundary condition
        for k1 in range(1, len(xpts1)-1): # move across columns
            phi2[0, k1] = phi[1, k1]

        # Zone 2 bottom boundary condition
        for k2 in range(len(xpts2)): # move across columns
            k2a = k2 + len(xpts1)-1
            phi2[0, k2a] = phi[1, k2a] - Vinf*dymin*(chord/2-xpts2[k2])/np.sqrt(fairf**2-(xpts2[k2]-chord/2)**2)

        # Zone 3 bottom boundary condition
        for k3 in range(1, len(xpts3)-1): # move across columns
            phi2[0, k3] = phi[1, k3]

        phi = np.copy(phi2)

        # Calculate residual
        for i in range(1, len(xpts)-1): # move across columns
            for j in range(1, len(ypts)-1): # move through columns
                helpx1 = 2*bigA*((phi[j, i+1]-phi[j, i])/(xpts[i+1] - xpts[i]) - (phi[j, i]-phi[j, i-1])/(xpts[i] - xpts[i-1]))/(xpts[i+1] - xpts[i-1])
                helpy1 = 2*((phi[j+1, i]-phi[j, i])/(ypts[j+1] - ypts[j]) - (phi[j, i]-phi[j-1, i])/(ypts[j] - ypts[j-1]))/(ypts[j+1] - ypts[j-1])
                res[j, i] = helpx1 + helpy1
        residualjac[t] = np.max(np.max(res))

    elapsed_time = time.time() - startjac
    if timing:
        print('Point Jacobi Elapsed Time: {:.3}'.format(elapsed_time))
    return residualjac, phi

def point_guass_seidel(time_steps, Minf, xpts, ypts, phi, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=False):
    bigA = 1 - Minf**2

    startgs = time.time()
    residualgs = np.ones(time_steps)
    res = np.zeros_like(phi)
    for t in range(time_steps):
    
        for i in range(1, len(xpts)-1): # move across columns
            for j in range(1, len(ypts)-1): # move through columns
                help1a = bigA/(xpts[i+1]-xpts[i-1])*(1/(xpts[i+1] - xpts[i]) + (1/(xpts[i] - xpts[i-1])))
                help1b = 1/(ypts[j+1]-ypts[j-1])*(1/(ypts[j+1] - ypts[j]) + (1/(ypts[j] - ypts[j-1])))
                help1 = 1/(2*(help1a + help1b))
                help2 = bigA*(phi[j, i+1] + phi[j, i-1])/((xpts[i] - xpts[i-1])*(xpts[i+1] - xpts[i]))
                help3 = (phi[j+1, i] + phi[j-1, i])/((ypts[j] - ypts[j-1])*(ypts[j+1] - ypts[j]))
                phi[j, i] = help1*(help2+help3)  

        # Zone 1 bottom boundary condition
        for k1 in range(1, len(xpts1)-1): # move across columns
            phi[0, k1] = phi[1, k1]
        
        # Zone 2 bottom boundary condition
        for k2 in range(len(xpts2)): # move across columns
            k2a = k2 + len(xpts1)-1
            phi[0, k2a] = phi[1, k2a] - Vinf*dymin*(chord/2-xpts2[k2])/np.sqrt(fairf**2-(xpts2[k2]-chord/2)**2)

        # Zone 3 bottom boundary condition
        for k3 in range(1, len(xpts3)-1): # move across columns
            phi[0, k3] = phi[1, k3]

        # Calculate residual
        for i in range(1, len(xpts)-1): # move across columns
            for j in range(1, len(ypts)-1): # move through columns
                helpx1 = 2*bigA*((phi[j, i+1]-phi[j, i])/(xpts[i+1] - xpts[i]) - (phi[j, i]-phi[j, i-1])/(xpts[i] - xpts[i-1]))/(xpts[i+1] - xpts[i-1])
                helpy1 = 2*((phi[j+1, i]-phi[j, i])/(ypts[j+1] - ypts[j]) - (phi[j, i]-phi[j-1, i])/(ypts[j] - ypts[j-1]))/(ypts[j+1] - ypts[j-1])
                res[j, i] = helpx1 + helpy1
        residualgs[t] = np.max(np.max(res))

    elapsed_time = time.time() - startgs
    if timing:
        print('Point Gauss-Seidel Elapsed Time: {:.3}'.format(elapsed_time))
    return residualgs

def line_jacobi(time_steps, Minf, xpts, ypts, phi, phi2, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=False):
    bigA = 1 - Minf**2

    array_len = len(ypts)
    A = np.ones(array_len) # phi[i-1]
    B = np.ones(array_len) # phi[i]
    C = np.ones(array_len) # phi[i+1]
    D = np.ones(array_len) # known

    startljc = time.time()
    residualljc = np.ones(time_steps)
    res = np.zeros_like(phi)
    for t in range(time_steps):
        
        # Zone 1 in front of the airfoil
        for i in range(1, len(xpts1)-1): # move across columns
            helpx = bigA/(xpts[i+1]-xpts[i-1]) * (1/(xpts[i+1]-xpts[i]) + 1/(xpts[i]-xpts[i-1]))
            for j in range(1, len(ypts)-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/((ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]))
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[i+1]-xpts[i-1]) * (phi[j, i+1]/(xpts[i+1]-xpts[i]) + phi[j, i-1]/(xpts[i]-xpts[i-1]))

            C[0] = -1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, i] # top boundary condition
            phi2[:, i] = solve_tridiagonal(len(B), A, B, C, D)

        # Zone 2 on the airfoil
        for i in range(len(xpts2)): # move across columns
            k2 = i + len(xpts1)-1
            helpx = bigA/(xpts[k2+1]-xpts[k2-1]) * (1/(xpts[k2+1]-xpts[k2]) + 1/(xpts[k2]-xpts[k2-1]))
            for j in range(1, len(ypts)-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/((ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]))
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[k2+1]-xpts[k2-1]) * (phi[j, k2+1]/(xpts[k2+1]-xpts[k2]) + phi[j, k2-1]/(xpts[k2]-xpts[k2-1]))

            C[0] =-1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] =-Vinf*dymin*(chord/2-xpts2[i])/np.sqrt(fairf**2-(xpts2[i]-chord/2)**2) # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, k2] # top boundary condition
            phi2[:, k2] = solve_tridiagonal(len(B), A, B, C, D)

        # Zone 3 behind the airfoil
        for i in range(len(xpts3)-2): # move across columns
            k3 = i + len(xpts1) + len(xpts2) - 1
            helpx = bigA/(xpts[k3+1]-xpts[k3-1]) * (1/(xpts[k3+1]-xpts[k3]) + 1/(xpts[k3]-xpts[k3-1]))
            for j in range(1, len(ypts)-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/((ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]))
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[k3+1]-xpts[k3-1]) * (phi[j, k3+1]/(xpts[k3+1]-xpts[k3]) + phi[j, k3-1]/(xpts[k3]-xpts[k3-1]))

            C[0] =-1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, k3] # top boundary condition
            phi2[:, k3] = solve_tridiagonal(len(B), A, B, C, D)

        phi = np.copy(phi2)

        # Calculate residual
        for i in range(1, len(xpts)-1): # move across columns
            for j in range(1, len(ypts)-1): # move through columns
                helpx1 = 2*bigA*((phi[j, i+1]-phi[j, i])/(xpts[i+1] - xpts[i]) - (phi[j, i]-phi[j, i-1])/(xpts[i] - xpts[i-1]))/(xpts[i+1] - xpts[i-1])
                helpy1 = 2*((phi[j+1, i]-phi[j, i])/(ypts[j+1] - ypts[j]) - (phi[j, i]-phi[j-1, i])/(ypts[j] - ypts[j-1]))/(ypts[j+1] - ypts[j-1])
                res[j, i] = helpx1 + helpy1
        residualljc[t] = np.max(np.max(res))
        
    elapsed_time = time.time() - startljc
    if timing:
        print('Line Jacobi Elapsed Time: {:.3}'.format(elapsed_time))
    return residualljc, phi

def line_guass_seidel(time_steps, Minf, xpts, ypts, phi, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=False):
    bigA = 1 - Minf**2

    array_len = len(ypts)
    A = np.ones(array_len) # phi[i-1]
    B = np.ones(array_len) # phi[i]
    C = np.ones(array_len) # phi[i+1]
    D = np.ones(array_len) # known

    startlgs = time.time()
    residuallgs = np.ones(time_steps)
    res = np.zeros_like(phi)
    for t in range(time_steps):
        
        # Zone 1 in front of the airfoil
        for i in range(1, len(xpts1)-1): # move across columns
            helpx = bigA/(xpts[i+1]-xpts[i-1]) * (1/(xpts[i+1]-xpts[i]) + 1/(xpts[i]-xpts[i-1]))
            for j in range(1, len(ypts)-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/((ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]))
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[i+1]-xpts[i-1]) * (phi[j, i+1]/(xpts[i+1]-xpts[i]) + phi[j, i-1]/(xpts[i]-xpts[i-1]))

            C[0] = -1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, i] # top boundary condition
            phi[:, i] = solve_tridiagonal(len(B), A, B, C, D)

        # Zone 2 on the airfoil
        for i in range(len(xpts2)): # move across columns
            k2 = i + len(xpts1)-1
            helpx = bigA/(xpts[k2+1]-xpts[k2-1]) * (1/(xpts[k2+1]-xpts[k2]) + 1/(xpts[k2]-xpts[k2-1]))
            for j in range(1, len(ypts)-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/((ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]))
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[k2+1]-xpts[k2-1]) * (phi[j, k2+1]/(xpts[k2+1]-xpts[k2]) + phi[j, k2-1]/(xpts[k2]-xpts[k2-1]))

            C[0] =-1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] =-Vinf*dymin*(chord/2-xpts2[i])/np.sqrt(fairf**2-(xpts2[i]-chord/2)**2) # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, k2] # top boundary condition
            phi[:, k2] = solve_tridiagonal(len(B), A, B, C, D)

        # Zone 3 behind the airfoil
        for i in range(len(xpts3)-2): # move across columns
            k3 = i + len(xpts1) + len(xpts2) - 1
            helpx = bigA/(xpts[k3+1]-xpts[k3-1]) * (1/(xpts[k3+1]-xpts[k3]) + 1/(xpts[k3]-xpts[k3-1]))
            for j in range(1, len(ypts)-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/( (ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]) )
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[k3+1]-xpts[k3-1]) * (phi[j, k3+1]/(xpts[k3+1]-xpts[k3]) + phi[j, k3-1]/(xpts[k3]-xpts[k3-1]))

            C[0] =-1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, k3] # top boundary condition
            phi[:, k3] = solve_tridiagonal(len(B), A, B, C, D)

        # Calculate residual
        for i in range(1, len(xpts)-1): # move across columns
            for j in range(1, len(ypts)-1): # move through columns
                helpx1 = 2*bigA*((phi[j, i+1]-phi[j, i])/(xpts[i+1] - xpts[i]) - (phi[j, i]-phi[j, i-1])/(xpts[i] - xpts[i-1]))/(xpts[i+1] - xpts[i-1])
                helpy1 = 2*((phi[j+1, i]-phi[j, i])/(ypts[j+1] - ypts[j]) - (phi[j, i]-phi[j-1, i])/(ypts[j] - ypts[j-1]))/(ypts[j+1] - ypts[j-1])
                res[j, i] = helpx1 + helpy1
        residuallgs[t] = np.max(np.max(res))

    elapsed_time = time.time() - startlgs
    if timing:
        print('Line Gauss-Seidel Elapsed Time: {:.3}'.format(elapsed_time))
    return residuallgs

def ADI(time_steps, Minf, xpts, ypts, phi, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=False):
    """ADI : alternating direction implicit"""
    bigA = 1 - Minf**2

    nypts = len(ypts)
    nxpts = len(xpts)
    nxpts1 = len(xpts1)
    nxpts2 = len(xpts2)
    nxpts3 = len(xpts3)

    A = np.zeros(nypts) # phi[i-1]
    B = np.zeros(nypts) # phi[i]
    C = np.zeros(nypts) # phi[i+1]
    D = np.zeros(nypts) # known

    Ay = np.zeros(nxpts) # phi[j-1]
    By = np.zeros(nxpts) # phi[j]
    Cy = np.zeros(nxpts) # phi[j+1]
    Dy = np.zeros(nxpts) # known

    startadi = time.time()
    residualadi = np.ones(time_steps)
    res = np.zeros_like(phi)
    for t in range(time_steps):

        # Step 1
        # Zone 1 in front of the airfoil
        for i in range(1, nxpts1-1): # move across columns
            helpx = bigA/(xpts[i+1]-xpts[i-1]) * (1/(xpts[i+1]-xpts[i]) + 1/(xpts[i]-xpts[i-1]))
            for j in range(1, nypts-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/( (ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]) )
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[i+1]-xpts[i-1]) * (phi[j, i+1]/(xpts[i+1]-xpts[i]) + phi[j, i-1]/(xpts[i]-xpts[i-1]))

            C[0] = -1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, i] # top boundary condition
            phi[:, i] = solve_tridiagonal(len(B), A, B, C, D)

        # Zone 2 on the airfoil
        for i in range(nxpts2): # move across columns
            k2 = i + nxpts1-1
            helpx = bigA/(xpts[k2+1]-xpts[k2-1]) * (1/(xpts[k2+1]-xpts[k2]) + 1/(xpts[k2]-xpts[k2-1]))
            for j in range(1, nypts-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/( (ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]) )
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[k2+1]-xpts[k2-1]) * (phi[j, k2+1]/(xpts[k2+1]-xpts[k2]) + phi[j, k2-1]/(xpts[k2]-xpts[k2-1]))

            C[0] =-1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] =-Vinf*dymin*(chord/2-xpts2[i])/np.sqrt(fairf**2-(xpts2[i]-chord/2)**2) # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, k2] # top boundary condition
            phi[:, k2] = solve_tridiagonal(len(B), A, B, C, D)

        # Zone 3 behind the airfoil
        for i in range(nxpts3-2): # move across columns
            k3 = i + nxpts1 + nxpts2 - 1
            helpx = bigA/(xpts[k3+1]-xpts[k3-1]) * (1/(xpts[k3+1]-xpts[k3]) + 1/(xpts[k3]-xpts[k3-1]))
            for j in range(1, nypts-1): # move through columns
                helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
                A[j-1] = -1/((ypts[j]-ypts[j-1]) * (ypts[j+1]-ypts[j-1]))
                B[j] = helpx + helpy
                C[j] = -1/((ypts[j+1]-ypts[j]) * (ypts[j+1]-ypts[j-1]))
                D[j] = bigA/(xpts[k3+1]-xpts[k3-1]) * (phi[j, k3+1]/(xpts[k3+1]-xpts[k3]) + phi[j, k3-1]/(xpts[k3]-xpts[k3-1]))

            C[0] =-1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phi[-1, k3] # top boundary condition
            phi[:, k3] = solve_tridiagonal(len(B), A, B, C, D)

        # Step 2
        for j in range(1, nypts-1): # move through columns
            helpy = 1/(ypts[j+1]-ypts[j-1]) * (1/(ypts[j+1]-ypts[j]) + 1/(ypts[j]-ypts[j-1]))
            for i in range(1, nxpts-1): # move across columns
                helpx = bigA/(xpts[i+1]-xpts[i-1]) * (1/(xpts[i+1]-xpts[i]) + 1/(xpts[i]-xpts[i-1]))
                Ay[i-1] = -bigA/((xpts[i]-xpts[i-1]) * (xpts[i+1]-xpts[i-1]))
                By[i] = helpx + helpy
                Cy[i] = -bigA/((xpts[i+1]-xpts[i]) * (xpts[i+1]-xpts[i-1]))
                Dy[i] = 1/(ypts[j+1]-ypts[j-1]) * (phi[j+1, i]/(ypts[j+1]-ypts[j]) + phi[j-1, i]/(ypts[j]-ypts[j-1]))

            Cy[0] = 0 # left boundary condition
            By[0] = 1 # left boundary condition
            Dy[0] = phi[j, 0] # left boundary condition
            Ay[-2] = 0 # top boundary condition
            By[-1] = 1 # top boundary condition
            Dy[-1] = phi[j, -1] # top boundary condition
   
            phi[j, :] = solve_tridiagonal(len(By), Ay, By, Cy, Dy)
        
        # Bottom boundary conditions
        phi[0, 1:nxpts1-1] = phi[1, 1:nxpts1-1] # in front of the airfoil
        phi[0, nxpts1-1:nxpts1+nxpts2-1] = phi[1, nxpts1-1:nxpts1+nxpts2-1] + -Vinf*dymin*(chord/2-xpts2)/np.sqrt(fairf**2-(xpts2-chord/2)**2) # on the airfoil
        phi[0, nxpts1+nxpts2-1:nxpts1+nxpts2+nxpts3-2] = phi[1, nxpts1+nxpts2-1:nxpts1+nxpts2+nxpts3-2] # behind the airfoil

        # Calculate residual
        for i in range(1, nxpts-1): # move across columns
            for j in range(1, nypts-1): # move through columns
                helpx1 = 2*bigA*((phi[j, i+1]-phi[j, i])/(xpts[i+1] - xpts[i]) - (phi[j, i]-phi[j, i-1])/(xpts[i] - xpts[i-1]))/(xpts[i+1] - xpts[i-1])
                helpy1 = 2*((phi[j+1, i]-phi[j, i])/(ypts[j+1] - ypts[j]) - (phi[j, i]-phi[j-1, i])/(ypts[j] - ypts[j-1]))/(ypts[j+1] - ypts[j-1])
                res[j, i] = helpx1 + helpy1
        residualadi[t] = np.max(np.max(res))

    elapsed_time = time.time() - startadi
    if timing:
        print('ADI Elapsed Time: {:.3}'.format(elapsed_time))
    return residualadi

def get_streamfcn(phi, xpts, ypts):
    # TODO make this work
    psi = np.zeros_like(phi)
    psi2 = np.zeros_like(phi)
    v = np.zeros_like(phi)
    u = np.zeros_like(phi)
    nypts = len(ypts)
    nxpts = len(xpts)

    for i in range(1, nxpts-1): # move across columns
        for j in range(1, nypts-1): # move through columns
            v[j, i] = (phi[j+1, i] - phi[j-1, i])/(ypts[j+1] - ypts[j-1])
            u[j, i] = (phi[j, i+1] - phi[j, i-1])/(xpts[i+1] - xpts[i-1])

    for i in range(1, nxpts-1):
        # bottom boundary
        u[0, i] = (phi[0, i+1] - phi[0, i-1])/(xpts[i+1] - xpts[i-1])
        v[0, i] = (phi[0, i] - phi[1, i])/(ypts[0] - ypts[1])
        # top boundary
        u[-1, i] = (phi[-1, i+1] - phi[-1, i-1])/(xpts[i+1] - xpts[i-1])
        v[-1, i] = (phi[-1, i] - phi[-2, i])/(ypts[-1] - ypts[-2])

    for j in range(nxpts-1):
        # bottom boundary
        u[0, i] = (phi[0, i+1] - phi[0, i-1])/(xpts[i+1] - xpts[i-1])
        v[0, i] = (phi[0, i] - phi[1, i])/(ypts[0] - ypts[1])
        # top boundary
        u[-1, i] = (phi[-1, i+1] - phi[-1, i-1])/(xpts[i+1] - xpts[i-1])
        v[-1, i] = (phi[-1, i] - phi[-2, i])/(ypts[-1] - ypts[-2])
        
    for i in range(1, nxpts-1):
        for j in range(nypts-1):
            psi[j, i] = psi[j, i-1]-v[j+1, i]*(ypts[j+1] - ypts[j])
            psi2[j, i] = psi2[j, i-1]-v[j+1, i]*(xpts[i+1] - xpts[i])

    print(psi)
    plt.contour(xpts, ypts, u-u[0,0], cmap='coolwarm')
    # plt.contour(xpts, ypts, v)
    #plt.clabel(cont, inline=2, colors='k', manual=False)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.tick_params(axis='both', labelsize=12)
    plt.colorbar(orientation='vertical').set_label('Streamfunction Values (\u03A8)')
    plt.show()

def main():
    chord = 1
    thickness = .06 # times the chord th = .06c

    ptsx2 = 21 # points on the airfoil

    ptsx = 16 # points in front of and behind the airfoil
    dxmin = chord/20
    xmax = chord*50

    ptsy = 51
    dymin = thickness/10
    ymax = chord*50

    # Airfoil equations
    fairf = ((chord/2)**2 + (thickness/2)**2)/(2*(thickness/2)) # constant that makes circle equation work
    x = np.linspace(0, 1, 100)
    y = np.sqrt(fairf**2-(x-chord/2)**2) + (thickness/2) - fairf
    # dydx = (chord/2-x)/np.sqrt(fairf**2-(x-chord/2)**2)

    ypts = stretch_mesh(dymin, ymax, ptsy)
    xpts1 = np.flip(-1*stretch_mesh(dxmin, xmax, ptsx)) # in front of airfoil
    xpts2 = np.linspace(0, chord, ptsx2) # on airfoil
    xpts3 = stretch_mesh(dxmin, xmax, ptsx) + 1 # behind airfoil
    xpts = np.hstack((xpts1[0:-1], xpts2, xpts3[1:]))

    pinf = 1
    densityinf = 1
    gamma = 1.4
    Minf = 0.5
    ainf = np.sqrt(gamma*pinf/densityinf)
    Vinf = Minf*ainf
    bigA = 1 - Minf**2

    phi = np.vstack([xpts for t in ypts])*Vinf

    phipjc = np.copy(phi)
    phipjc2 = np.copy(phi)
    phipgs = np.copy(phi)
    philjc = np.copy(phi)
    philjc2 = np.copy(phi)
    philgs = np.copy(phi)
    phiadi = np.copy(phi)
    phipgc = np.copy(phi)

    xv, yv = np.meshgrid(xpts, ypts)
    sc = plt.scatter(xv, yv, c=phi[:, :len(xpts)])#, vmin=-30, vmax=30)
    plt.colorbar(sc)
    plt.title("initial condition")
    plt.savefig("initial condition")
    plt.close()
    # plt.show()

    time_steps = 200
    residualpjc, phif = point_jacobi(time_steps, Minf, xpts, ypts, phipjc, phipjc2, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=True)
    residualpgs = point_guass_seidel(time_steps, Minf, xpts, ypts, phipgs, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=True)
    residualljc, phif2 = line_jacobi(time_steps, Minf, xpts, ypts, philjc, philjc2, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=True)
    residuallgs = line_guass_seidel(time_steps, Minf, xpts, ypts, philgs, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=True)
    residualadi = ADI(time_steps, Minf, xpts, ypts, phiadi, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=True)

    # # For use to validate Prandtly-Glauert correction
    # # TODO Make this replicate book results
    # Minf2 = .01
    # ainf2 = np.sqrt(gamma*pinf/densityinf)
    # Vinf2 = Minf2*ainf2
    # residualpgc = line_guass_seidel(time_steps, Minf2, xpts, ypts, phipgc, xpts1, xpts2, xpts3, Vinf2, dymin, fairf, chord, timing=True)

    u_pjc = np.zeros_like(xpts)
    u_pgs = np.zeros_like(xpts)
    u_ljc = np.zeros_like(xpts)
    u_lgs = np.zeros_like(xpts)
    u_adi = np.zeros_like(xpts)
    v_pjc = np.zeros_like(xpts)
    v_pgs = np.zeros_like(xpts)
    v_ljc = np.zeros_like(xpts)
    v_lgs = np.zeros_like(xpts)
    v_adi = np.zeros_like(xpts)
    u_pgc = np.zeros_like(xpts)
    for i in range(1, len(xpts)-1):
        u_pjc[i] = (phif[0, i+1] - phif[0, i-1])/(xpts[i+1] - xpts[i-1])
        u_pgs[i] = (phipgs[0, i+1] - phipgs[0, i-1])/(xpts[i+1] - xpts[i-1])
        u_ljc[i] = (phif2[0, i+1] - phif2[0, i-1])/(xpts[i+1] - xpts[i-1])
        u_lgs[i] = (philgs[0, i+1] - philgs[0, i-1])/(xpts[i+1] - xpts[i-1])
        u_adi[i] = (phiadi[0, i+1] - phiadi[0, i-1])/(xpts[i+1] - xpts[i-1])
        v_pjc[i] = (phif[0, i] - phif[1, i])/(ypts[0] - ypts[1])
        v_pgs[i] = (phipgs[0, i] - phipgs[1, i])/(ypts[0] - ypts[1])
        v_ljc[i] = (phif2[0, i] - phif2[1, i])/(ypts[0] - ypts[1])
        v_lgs[i] = (philgs[0, i] - philgs[1, i])/(ypts[0] - ypts[1])
        v_adi[i] = (phiadi[0, i] - phiadi[1, i])/(ypts[0] - ypts[1])
        u_pgc[i] = (phipgc[0, i+1] - phipgc[0, i-1])/(xpts[i+1] - xpts[i-1])

    pressure_pjc = pinf*(1-(gamma-1)/2*Minf**2*((u_pjc**2+v_pjc**2)/Vinf**2-1))**(gamma/(gamma-1))
    pressure_pgs = pinf*(1-(gamma-1)/2*Minf**2*((u_pgs**2+v_pgs**2)/Vinf**2-1))**(gamma/(gamma-1))
    pressure_ljc = pinf*(1-(gamma-1)/2*Minf**2*((u_ljc**2+v_ljc**2)/Vinf**2-1))**(gamma/(gamma-1))
    pressure_lgs = pinf*(1-(gamma-1)/2*Minf**2*((u_lgs**2+v_lgs**2)/Vinf**2-1))**(gamma/(gamma-1))
    pressure_adi = pinf*(1-(gamma-1)/2*Minf**2*((u_adi**2+v_adi**2)/Vinf**2-1))**(gamma/(gamma-1))
    cp_pjc = 2*(pressure_pjc-pinf)/densityinf/Vinf**2
    cp_pgs = 2*(pressure_pgs-pinf)/densityinf/Vinf**2
    cp_ljc = 2*(pressure_ljc-pinf)/densityinf/Vinf**2
    cp_lgs = 2*(pressure_lgs-pinf)/densityinf/Vinf**2
    cp_adi = 2*(pressure_adi-pinf)/densityinf/Vinf**2
    # cp_pgc0 = -(u_pgc-Vinf2)/Vinf2
    # cp_pgc02 = cp_lgs*np.sqrt(1-Minf**2)
    # cp_pgcM = cp_pgc0/np.sqrt(1-Minf2**2)
    plt.plot(xpts, -cp_pjc, marker='.', label='pjc')
    plt.plot(xpts, -cp_pgs, marker='+', label='pgs')
    plt.plot(xpts, -cp_ljc, marker='o', label='ljc')
    plt.plot(xpts, -cp_lgs, marker='x', label='lgs')
    plt.plot(xpts, -cp_adi, marker='s', label='adi')
    # plt.plot(xpts, -cp_pgc0, label='M=0', linestyle='--')
    # plt.plot(xpts, -cp_pgc02, label='M=02', linestyle='--')
    # plt.plot(xpts, -cp_pgcM, label='M=0 P-G Correction', linestyle='--')
    plt.xlim([xpts1[-3], xpts3[3]])
    plt.ylim([-.35, .2])
    plt.xlabel('x')
    plt.ylabel('$-C_p$')
    plt.legend()
    plt.savefig('Exercise 5.5 Cps.png')
    plt.close()
    plt.show()

    # sc = plt.scatter(xv, yv, c=phiadi[:, :len(xpts)])#, vmin=-30, vmax=30)
    # plt.plot(x, y)
    # plt.colorbar(sc)
    # plt.title('t={} full adi'.format(time_steps))
    # plt.savefig('t={} full scatter adi'.format(time_steps))
    # # plt.close()
    # plt.show()

    plt.semilogy(np.linspace(0, time_steps-1, time_steps), residualpjc, label='pjc', linestyle='--')
    plt.semilogy(np.linspace(0, time_steps-1, time_steps), residualpgs, label='pgs', linestyle='-.')
    plt.semilogy(np.linspace(0, time_steps-1, time_steps), residualljc, label='ljc', linestyle=':')
    plt.semilogy(np.linspace(0, time_steps-1, time_steps), residuallgs, label='lgs', linestyle='-')
    plt.semilogy(np.linspace(0, time_steps-1, time_steps), residualadi, label='adi', linestyle='-')
    plt.legend()
    # plt.savefig('Exercise 5.5 Residuals.png')
    plt.close()
    plt.show()
if __name__ == '__main__':
    main()