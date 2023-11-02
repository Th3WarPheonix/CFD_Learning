
import numpy as np
import matplotlib.pyplot as plt
from Method_DB import solve_tridiagonal, stretch_mesh

def murman_cole(timesteps, Minf, xpts, ypts, phihat, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, gamma=1.4, timing=False):
    """Solves the transonic small disturbance equations with the Muirman-Cole algorithm"""

    nypts = len(ypts)
    nxpts = len(xpts)
    nxpts1 = len(xpts1)
    nxpts2 = len(xpts2)
    nxpts3 = len(xpts3)

    A = np.ones(nypts) # phi[j-1]
    B = np.ones(nypts) # phi[j]
    C = np.ones(nypts) # phi[j+1]
    D = np.ones(nypts) # known

    residual = np.ones(timesteps)
    res = np.zeros_like(phihat)
    for t in range(timesteps):
        # Zone 1 in front of the airfoil   
        for i in range(1, nxpts1-1):
            xpts10 = 1/(xpts[i+1]-xpts[i])
            xpts11 = 1/(xpts[i+1]-xpts[i-1])
            xpts01 = 1/(xpts[i]-xpts[i-1])
            xpts02 = 1/(xpts[i]-xpts[i-2])
            xpts12 = 1/(xpts[i-1]-xpts[i-2])

            for j in range(1, nypts-1):
                ypts10 = 1/(ypts[j+1]-ypts[j])
                ypts11 = 1/(ypts[j+1]-ypts[j-1])
                ypts01 = 1/(ypts[j]-ypts[j-1])

                # subsonic
                # Aij = A[i, j]
                # muij = mu[i, j]
                Aij = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phihat[i+1][j] - phihat[i-1][j])*xpts11
                if Aij > 0:
                    muij = 0
                elif Aij < 0:
                    muij = 1

                # supersonic (DONT USE at i=1 index [i-2] will be negative)\
                # Aij1 = A[i-1, j]
                # muij1 = mu[i-1, j]
                Aij1 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phihat[i][j] - phihat[i-2][j])*xpts02
                if Aij1 > 0: # subsonic
                    muij1 = 0
                elif Aij1 < 0: # supersonic
                    muij1 = 1
                
                muij = 0
                muij1 = 0

                A[j-1] = ypts10*ypts11
                B[j] = (muij-1)*Aij*(xpts10 + xpts01)*xpts11 + muij1*Aij1*xpts01*xpts02 - ypts10*ypts11 - ypts01*ypts11
                C[j] = ypts01*ypts11
                D[j] = (muij-1)*Aij*xpts11*(phihat[i+1][j]*xpts10 + phihat[i-1][j]*xpts01) - muij1*Aij1*xpts02*(-phihat[i-1][j]*xpts01 + (-phihat[i-1][j]+phihat[i-2][j])*xpts12)

            C[0] = -1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phihat[i, -1] # top boundary condition
            # print('A', A)
            # print('b', B)
            # print('c', C)
            # print('d', D)
            # print('phi1', phihat[i][:])
            phihat[i][:] = solve_tridiagonal(nypts, A, B, C, D)
            # print('phi2', phihat[i][:])

        # Zone 2 in front of the airfoil   
        for i in range(nxpts2):
            k2 = i + nxpts1 - 1
            xpts10 = 1/(xpts[k2+1]-xpts[k2])
            xpts11 = 1/(xpts[k2+1]-xpts[k2-1])
            xpts01 = 1/(xpts[k2]-xpts[k2-1])
            xpts02 = 1/(xpts[k2]-xpts[k2-2])
            xpts12 = 1/(xpts[k2-1]-xpts[k2-2])

            for j in range(1, nypts-1):
                ypts10 = 1/(ypts[j+1]-ypts[j])
                ypts11 = 1/(ypts[j+1]-ypts[j-1])
                ypts01 = 1/(ypts[j]-ypts[j-1])

                # Aij = A[i, j]
                # muij = mu[i, j]
                Aij = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phihat[k2+1][j] - phihat[k2-1][j])*xpts11
                if Aij > 0: # subsonic
                    muij = 0
                elif Aij < 0: # supersonic
                    muij = 1

                # supersonic (DONT USE at i=1 index [i-2] will be negative)
                # Aij1 = A[i-1, j]
                # muij1 = mu[i-1, j]
                Aij1 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phihat[k2][j] - phihat[k2-2][j])*xpts02
                if Aij1 > 0:  # subsonic
                    muij1 = 0
                elif Aij1 < 0: # supersonic
                    muij1 = 1
                
                muij = 0
                muij1 = 0

                A[j-1] = ypts10*ypts11
                B[j] = (muij-1)*Aij*(xpts10 + xpts01)*xpts11 + muij1*Aij1*xpts01*xpts02 - ypts10*ypts11 - ypts01*ypts11
                C[j] = ypts01*ypts11
                D[j] = (muij-1)*Aij*xpts11*(phihat[k2+1][j]*xpts10 + phihat[k2-1][j]*xpts01) - muij1*Aij1*xpts02*(-phihat[k2-1][j]*xpts01 + (-phihat[k2-1][j]+phihat[k2-2][j])*xpts12)

            C[0] =-1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] =-Vinf*dymin*(chord/2-xpts2[i])/np.sqrt(fairf**2-(xpts2[i]-chord/2)**2) # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phihat[k2, -1] # top boundary condition
            phihat[k2][:] = solve_tridiagonal(nypts, A, B, C, D)
            # print('zone2', k2, phihat[k2][:])

        # Zone 3 behind the airfoil   
        for i in range(nxpts3-2):
            k3 = i + nxpts1 + nxpts2 - 1
            xpts10 = 1/(xpts[k3+1]-xpts[k3])
            xpts11 = 1/(xpts[k3+1]-xpts[k3-1])
            xpts01 = 1/(xpts[k3]-xpts[k3-1])
            xpts02 = 1/(xpts[k3]-xpts[k3-2])
            xpts12 = 1/(xpts[k3-1]-xpts[k3-2])

            for j in range(1, nypts-1):
                ypts10 = 1/(ypts[j+1]-ypts[j])
                ypts11 = 1/(ypts[j+1]-ypts[j-1])
                ypts01 = 1/(ypts[j]-ypts[j-1])

                # subsonic
                # Aij = A[i, j]
                # muij = mu[i, j]
                Aij = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phihat[k3+1][j] - phihat[k3-1][j])*xpts11
                if Aij > 0: # subsonic
                    muij = 0
                elif Aij < 0: # supersonic
                    muij = 1
                
                # supersonic (DONT USE at i=1 index [i-2] will be negative)\
                # Aij1 = A[i-1, j]
                # muij1 = mu[i-1, j]
                Aij1 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phihat[k3][j] - phihat[k3-2][j])*xpts02
                if Aij1 > 0:
                    muij1 = 0
                elif Aij1 < 0: # supersonic
                    muij1 = 1

                muij = 0
                muij1 = 0

                A[j-1] = ypts10*ypts11
                B[j] = (muij-1)*Aij*(xpts10 + xpts01)*xpts11 + muij1*Aij1*xpts01*xpts02 - ypts10*ypts11 - ypts01*ypts11
                C[j] = ypts01*ypts11
                D[j] = (muij-1)*Aij*xpts11*(phihat[k3+1][j]*xpts10 + phihat[k3-1][j]*xpts01) - muij1*Aij1*xpts02*(-phihat[k3-1][j]*xpts01 + (-phihat[k3-1][j]+phihat[k3-2][j])*xpts12)

            C[0] = -1 # bottom boundary condition
            B[0] = 1 # bottom boundary condition
            D[0] = 0 # bottom boundary condition
            A[-2] = 0 # top boundary condition
            B[-1] = 1 # top boundary condition
            D[-1] = phihat[k3, -1] # top boundary condition
            phihat[k3][:] = solve_tridiagonal(nypts, A, B, C, D)
            # print(phihat[k3][:])

        # Calculate residual
        for i in range(1, nxpts-1): # move across columns
            xpts10 = 1/(xpts[i+1]-xpts[i])
            xpts11 = 1/(xpts[i+1]-xpts[i-1])
            xpts01 = 1/(xpts[i]-xpts[i-1])
            xpts02 = 1/(xpts[i]-xpts[i-2])
            xpts12 = 1/(xpts[i-1]-xpts[i-2])
            for j in range(1, nypts-1): # move through columns
                ypts10 = 1/(ypts[j+1]-ypts[j])
                ypts11 = 1/(ypts[j+1]-ypts[j-1])
                ypts01 = 1/(ypts[j]-ypts[j-1])
                # hardcoded for tests
                muij = 0
                muij1 = 0
                
                Dxxa = (phihat[i+1][j] - phihat[i][j])*xpts10 - (phihat[i][j] - phihat[i-1][j])*xpts01
                Dxx = Dxxa*xpts11

                Dyya = (phihat[i][j+1] - phihat[i][j])*ypts10 - (phihat[i][j] - phihat[i][j+1])*ypts01
                Dyy = Dyya*ypts11

                Dxx1a = (phihat[i][j] - phihat[i-1][j])*xpts01 - (phihat[i-1][j] - phihat[i-2][j])*xpts12
                Dxx1 = Dxx1a*xpts02
                res1 = (1-muij)*Aij*Dxx + muij1*Aij1*Dxx1 + Dyy
                res[i, j] = abs(res1)
        residual[t] = np.max(np.max(res))
    return residual

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

    Vinf = 1
    densityinf = 1
    gamma = 1.4
    Minf = [.735, .908]
    # Minf = [0.5]
    pinf = 1/gamma/Minf[0]**2
    
    bigA = 1 - Minf[0]**2

    phihat = np.ones((len(xpts), len(ypts))) # phihat is the disturbance in phi

    xv, yv = np.meshgrid(xpts, ypts)
    sc = plt.scatter(xv, yv, c=phihat[:, :len(xpts)])#, vmin=-30, vmax=30)
    plt.colorbar(sc)
    plt.title("initial condition")
    plt.savefig("initial condition")
    plt.close()
    # plt.show()

    time_steps = 400
    residuallgs = murman_cole(time_steps, Minf[0], xpts, ypts, phihat, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord)

    # u_adi = np.zeros_like(xpts)
    # v_adi = np.zeros_like(xpts)
    # for i in range(1, len(xpts)-1):
    #     u_adi[i] = Vinf + (phihat[0, i+1] - phihat[0, i-1])/(xpts[i+1] - xpts[i-1])
    #     v_adi[i] = (phihat[0, i] - phihat[1, i])/(ypts[0] - ypts[1])

    # pressure_adi = pinf*(1-(gamma-1)/2*Minf[0]**2*((u_adi**2+v_adi**2)/Vinf**2-1))**(gamma/(gamma-1))
    # cp_adi = 2*(pressure_adi-pinf)/densityinf/Vinf**2

    # plt.plot(xpts, -cp_adi, marker='s', label='adi')
    # plt.xlim([xpts1[-3], xpts3[3]])
    # plt.ylim([-.35, .2])
    # plt.xlabel('x')
    # plt.ylabel('$-C_p$')
    # plt.legend()
    # plt.savefig('Exercise 6.1 Cps.png')
    # # plt.close()
    # plt.show()   

    # sc = plt.scatter(xv, yv, c=phihat[:, :len(xpts)])#, vmin=-30, vmax=30)
    # plt.colorbar(sc)
    # plt.title("test")
    # plt.savefig("test")
    # # plt.close()
    # plt.show()

    plt.semilogy(np.linspace(0, time_steps-1, time_steps), residuallgs, label='lgs', linestyle='--')
    plt.legend()
    # plt.savefig('Exercise 5.6 Residuals.png')
    # plt.close()
    plt.show()

if __name__ == '__main__':
    main()
