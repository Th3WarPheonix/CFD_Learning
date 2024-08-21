import numpy as np
import matplotlib.pyplot as plt
import Method_DB as mdb
import time

def full_potential(time_steps, Minf, xpts, ypts, phi, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, gamma=1.4, timing=False):
    """Solves the full potential equations using the ADI method, and is
    applicable to arbitrary coordinates when certaain portions of the
    code are changed. This method uses a finte difference scheme where
    state variables are stored at the nodes and the metric terms are
    calculated at the half distances between nodes"""

    nypts = len(ypts)
    nxpts = len(xpts)
    nxpts1 = len(xpts1)
    nxpts2 = len(xpts2)

    # Solving one column of points at a time
    A = np.zeros(nypts) # phi[i-1]
    B = np.zeros(nypts) # phi[i]
    C = np.zeros(nypts) # phi[i+1]
    D = np.zeros(nypts) # known

    startlgs = time.time()
    residual = np.ones(time_steps)
    res = np.zeros_like(phi)

    # Constants
    gm1 = gamma - 1
    gm12 = (gamma - 1)/2
    # only ever take one step in the Xi or Eta direction
    de = 1
    dn = 1
    for t in range(time_steps):
        for i in range(1, nxpts-1): # move across columns
            for j in range(1, nypts-1): # move through columns
                
                # Step 1 Delaing with metric terms

                # pm notation is used to denote half steps between whole
                # number nodes a "p" in the i place mean i+1/2, an "m"
                # in the i place means i-1/2 likewise for j. If the
                # there is a halfstep in 1 direction but not the other
                # then the half step will contain the "p" or "m" and the
                # direction not incremented will contain either the i or
                # j. If there is a multiple step then a number will
                # precede the p or m
                # xpp = x[j+1/2, i+1/2]
                # xpm = x[j-1/2, i+1/2]
                # xmp = x[j+1/2, i-1/2]
                # multistep
                # xp2p = x[j+3/2, i+1/2]
                # x2pm = x[j-1/2, i+3/2]
                # x2m3p = x[j+5/2, i-3/2]

                # Getting the points of the "cell" around the ij point
                xpp = (xpts[j, i] + xpts[j, i+1] + xpts[j+1, i] + xpts[j+1, i+1])/4
                xpm = (xpts[j, i] + xpts[j, i+1] + xpts[j-1, i] + xpts[j-1, i+1])/4
                xmm = (xpts[j, i] + xpts[j, i-1] + xpts[j-1, i] + xpts[j-1, i-1])/4
                xmp = (xpts[j, i] + xpts[j, i-1] + xpts[j+1, i] + xpts[j+1, i-1])/4

                ypp = (ypts[j, i] + ypts[j, i+1] + ypts[j+1, i] + ypts[j+1, i+1])/4
                ypm = (ypts[j, i] + ypts[j, i+1] + ypts[j-1, i] + ypts[j-1, i+1])/4
                ymm = (ypts[j, i] + ypts[j, i-1] + ypts[j-1, i] + ypts[j-1, i-1])/4
                ymp = (ypts[j, i] + ypts[j, i-1] + ypts[j+1, i] + ypts[j+1, i-1])/4

                # F' (horiztonal) flux terms
                # right side of cell
                dxdnpj = (xpp - xpm)*dn
                dydnpj = (ypp - ypm)*dn
                dxdepj = (xpts[j, i+1] - xpts[j, i])*de
                dydepj = (ypts[j, i+1] - ypts[j, i])*de

                # left side of cell
                dxdnmj = (xmp - xmm)*dn
                dydnmj = (ymp - ymm)*dn
                dxdemj = (xpts[j, i] - xpts[j, i-1])*de
                dydemj = (ypts[j, i] - ypts[j, i-1])*de

                dxypj = dxdepj*dydnpj-dxdnpj*dydepj # F' Jacobian right
                dxymj = dxdemj*dydnmj-dxdnmj*dydemj # F' Jacobian left

                # G' (vertical) flux terms
                # top side of cell
                dxdeip = (xpp - xmp)*de
                dydeip = (ypp - ymp)*de
                dxdnip = (xpts[j+1, i] - xpts[j, i])*dn
                dydnip = (ypts[j+1, i] - ypts[j, i])*dn
                # bottom side of cell
                dxdeim = (xpm - xmm)*de
                dydeim = (ypm - ymm)*de
                dxdnim = (xpts[j, i] - xpts[j-1, i])*dn
                dydnim = (ypts[j, i] - ypts[j-1, i])*dn

                dxyip = dxdeip*dydnip-dxdnip*dydeip # G' Jacobian top
                dxyim = dxdeim*dydnim-dxdnim*dydeim # G' Jacobian bottom

                # Starting forming terms
                T11pj = dydnpj*dydnpj+dxdnpj*dxdnpj # F' right
                T12pj =-dydepj*dydnpj-dxdepj*dxdnpj # F' right
                T11mj = dydnmj*dydnmj+dxdnmj*dxdnmj # F' left
                T12mj =-dydemj*dydnmj-dxdemj*dxdnmj # F' left

                T22ip = dydeip*dydeip+dxdeip*dxdeip # G' top
                T21ip =-dydeip*dydnip-dxdeip*dxdnip # G' top
                T22im = dydeim*dydeim+dxdeim*dxdeim # G' bottom
                T21im =-dydeim*dydnim-dxdeim*dxdnim # G' bottom

                # Step 2 Dealing with density
                # right
                dpdepj = (phi[j, i+1] - phi[j, i])*de
                dpdnpj = (phi[j+1, i] + phi[j+1, i+1] - phi[j-1, i] - phi[j-1, i+1])/4*dn
                # left
                dpdemj = (phi[j, i] - phi[j, i-1])*de
                dpdnmj = (phi[j+1, i] + phi[j+1, i-1] - phi[j-1, i] - phi[j-1, i-1])/4*dn
                # top
                dpdeip = (phi[j, i+1] + phi[j+1, i+1] - phi[j, i-1] - phi[j+1, i-1])/4*de
                dpdnip = (phi[j+1, i] - phi[j, i])*dn
                # bottom
                dpdeim = (phi[j, i+1] + phi[j-1, i+1] - phi[j, i-1] - phi[j-1, i-1])/4*de
                dpdnim = (phi[j, i] - phi[j-1, i])*dn

                # right right
                x2pp = (xpts[j, i+1] + xpts[j, i+2] + xpts[j+1, i+1] + xpts[j+1, i+2])/4
                x2pm = (xpts[j, i+1] + xpts[j, i+2] + xpts[j-1, i+1] + xpts[j-1, i+2])/4
                y2pp = (ypts[j, i+1] + ypts[j, i+2] + ypts[j+1, i+1] + ypts[j+1, i+2])/4
                y2pm = (ypts[j, i+1] + ypts[j, i+2] + ypts[j-1, i+1] + ypts[j-1, i+2])/4
                dxdn2pj = (x2pp - x2pm)*dn
                dydn2pj = (y2pp - y2pm)*dn
                dxde2pj = (xpts[j, i+2] - xpts[j, i+1])*de
                dyde2pj = (ypts[j, i+2] - ypts[j, i+1])*de
                dxy2pj = dxde2pj*dydn2pj-dxdn2pj*dyde2pj
                dpde2pj = (phi[j, i+2] - phi[j, i+1])*de
                dpdn2pj = (phi[j+1, i+1] + phi[j+1, i+2] - phi[j-1, i+1] - phi[j-1, i+2])/4*dn

                upj = (dydnpj*dpdepj - dxdnpj*dpdnpj)/dxypj
                vpj = (-dydepj*dpdepj + dxdepj*dpdnpj)/dxypj
                basepj = (1-gm12*Minf**2*((upj**2+vpj**2)/Vinf**2-1))
                rhopj = pinf*basepj**(gamma/gm1)
                rhopj = rhoinf*basepj**(1/gm1)

                umj = (dydnmj*dpdemj - dxdnmj*dpdnmj)/dxymj
                vmj = (-dydemj*dpdemj + dxdemj*dpdnmj)/dxymj
                basemj = (1-gm12*Minf**2*((umj**2+vmj**2)/Vinf**2-1))
                rhomj = pinf*basemj**(gamma/gm1)
                rhomj = rhoinf*basemj**(1/gm1)

                u2pj = (dydn2pj*dpde2pj - dxdn2pj*dpdn2pj)/dxy2pj
                v2pj = (-dyde2pj*dpde2pj + dxde2pj*dpdn2pj)/dxy2pj
                base2pj = (1-gm12*Minf**2*((u2pj**2+v2pj**2)/Vinf**2-1))
                rho2pj = pinf*base2pj**(gamma/gm1)
                rho2pj = rhoinf*base2pj**(1/gm1)

                uprimeij = (phi[j, i+1] - phi[j, i-1])/2*de
                uprimepj = (phi[j, i+1] - phi[j, i])*de
                
                if uprimepj >= 0:
                    machtpj = machij**2
                else:
                    machtpj = mach1j**2

                vpj = np.max(0, machtpj**2 - 1)
                if uprimeij >= 0:
                    rhot = (1-vpj)*rhopj + vpj*rhomj
                else:
                    rhot = (1-vpj)*rhopj + vpj*rho2pj

                # Finish forming terms
                T11pj *= rhotpj/dxypj # right
                T12pj *= rhotpj/dxypj # right
                T11mj *= rhotmj/dxymj # left
                T12mj *= rhotmj/dxymj # left

                T22ip *= rhotip/dxyip # top
                T21ip *= rhotip/dxyip # top
                T22im *= rhotim/dxyim # bottom
                T21im *= rhotim/dxyim # bottom
    
                # Step 3 Matrix solution
                a0 = 1
                A = -T22ip - (T12pj - T12mj)/4
                B = T11pj + T11mj + T22ip + T22im + a0
                C = -T22im + (T12pj - T12mj)/4
    #         # Applying boundary conditions
    #         if i < nxpts1: # in front of airfoil
    #             C[0] = -1 # bottom boundary condition
    #             B[0] = 1 # bottom boundary condition
    #             D[0] = 0 # bottom boundary condition
    #             A[-2] = 0 # top boundary condition
    #             B[-1] = 1 # top boundary condition
    #             D[-1] = phi[-1, i] # top boundary condition            
    #         elif nxpts1 <= i <= nxpts1 + nxpts2 - 1: # on top of airfoil
    #             C[0] =-1 # bottom boundary condition
    #             B[0] = 1 # bottom boundary condition
    #             D[0] =-Vinf*dymin*(chord/2-xpts[i])/np.sqrt(fairf**2-(xpts[i]-chord/2)**2) # bottom boundary condition
    #             A[-2] = 0 # top boundary condition
    #             B[-1] = 1 # top boundary condition
    #             D[-1] = phi[-1, i] # top boundary condition
    #         elif nxpts1 + nxpts2 - 1 <= i <= nxpts: # behind of airfoil
    #             C[0] =-1 # bottom boundary condition
    #             B[0] = 1 # bottom boundary condition
    #             D[0] = 0 # bottom boundary condition
    #             A[-2] = 0 # top boundary condition
    #             B[-1] = 1 # top boundary condition
    #             D[-1] = phi[-1, i] # top boundary condition
    #         phi[:, i] = mdb.solve_tridiagonal(nypts, A, B, C, D)

    #     # Calculate residual
    #     for i in range(1, nxpts-1): # move across columns
    #         xpts10 = 1/(xpts[i+1]-xpts[i])
    #         xpts11 = 1/(xpts[i+1]-xpts[i-1])
    #         xpts01 = 1/(xpts[i]-xpts[i-1])
    #         xpts02 = 1/(xpts[i]-xpts[i-2])
    #         xpts12 = 1/(xpts[i-1]-xpts[i-2])

    #         for j in range(1, nypts-1): # move through columns
    #             ypts10 = 1/(ypts[j+1]-ypts[j])
    #             ypts11 = 1/(ypts[j+1]-ypts[j-1])
    #             ypts01 = 1/(ypts[j]-ypts[j-1])

    #             bigA0 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phi[j][i+1] - phi[j][i-1])*xpts11
    #             bigA1 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phi[j][i] - phi[j][i-2])*xpts02
    #             if bigA0 > 0:
    #                 mu0 = 0
    #             elif bigA0 <= 0:
    #                 mu0 = 1

    #             if bigA1 > 0:
    #                 mu1 = 0
    #             elif bigA1 <= 0:
    #                 mu1 = 1

    #             Dxxa = (phi[j, i+1]-phi[j, i])*xpts10 - (phi[j, i]-phi[j, i-1])*xpts01
    #             Dxx = Dxxa*xpts11

    #             Dxx1a = (phi[j][i] - phi[j][i-1])*xpts01 - (phi[j][i-1] - phi[j][i-2])*xpts12
    #             Dxx1 = Dxx1a*xpts02

    #             Dyya = (phi[j+1, i]-phi[j, i])*ypts10 - (phi[j, i]-phi[j-1, i])*ypts01
    #             Dyy = Dyya*ypts11

    #             res[j, i] = (1-mu0)*bigA0*Dxx + mu1*bigA1*Dxx1 + Dyy
    #     residual[t] = np.max(np.max(abs(res)))
    #     print(residual[t])
        
    elapsed_time = time.time() - startlgs
    if timing:
        print('Line Gauss-Seidel Elapsed Time: {:.3}'.format(elapsed_time))

    return residual

def airfoil(chord, thickness, x):
    # Airfoil equations
    fairf = ((chord/2)**2 + (thickness/2)**2)/(2*(thickness/2)) # constant that makes circle equation work
    y = np.sqrt(fairf**2-(x-chord/2)**2) + (thickness/2) - fairf
    dydx = (chord/2-x)/np.sqrt(fairf**2-(x-chord/2)**2)
    return y, dydx

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
    
    ypts = mdb.stretch_mesh(dymin, ymax, ptsy)
    xpts1 = -np.flip(mdb.stretch_mesh(dxmin, xmax, ptsx))[:-1] # in front of airfoil
    xpts2 = np.linspace(0, chord, ptsx2) # on airfoil
    xpts3 = mdb.stretch_mesh(dxmin, xmax, ptsx)[1:] + chord # behind airfoil
    xpts = np.hstack((xpts1, xpts2, xpts3))
    nypts = len(ypts)
    nxpts = len(xpts)
    nxpts1 = len(xpts1)
    nxpts2 = len(xpts2)

    phi = np.zeros((nypts, nxpts))
    Minf = 0.5
    Vinf = 1
    ainf = Vinf/Minf

    # Squish the mesh around the airfoil
    y, _ = airfoil(chord, thickness, xpts2)
    y2 = (y - ypts[-1])/(ypts[0] - ypts[-1])
    ypts2 = np.zeros((nypts, nxpts2))
    for i, _ in enumerate(ypts2[0]): # move across columns
        ypts2[:, i] = ypts*y2[i]+y[i]
    
    xv1, yv1 = np.meshgrid(xpts1, ypts)
    xv3, yv3 = np.meshgrid(xpts3, ypts)
    xv2, _ = np.meshgrid(xpts2, ypts2[:, 0])

    plt.scatter(xv1, yv1)
    plt.scatter(xv2, ypts2)
    plt.scatter(xv3, yv3)
    plt.title("initial condition")
    # plt.close()
    plt.show()

    time_steps = 1000
    # residualmclgs = murman_cole_lgs(time_steps, Minf, xpts, ypts, phi, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, timing=True)

    # u = np.zeros_like(phi)
    # v = np.zeros_like(phi)
    # vel = np.zeros_like(phi)
    # mach = np.zeros_like(phi)
    # for i in range(1, len(xpts)-1):
    #     for j in range(1, len(ypts)-1):
    #         u[j, i] = (phi[j, i+1] - phi[j, i-1])/(xpts[i+1] - xpts[i-1])+Vinf
    #         v[j, i] = (phi[j, i] - phi[j+1, i])/(ypts[j] - ypts[j+1])
    #         vel[j, i] = np.sqrt(u[j, i]**2 + v[j, i]**2)
    #         mach[j, i] = vel[j, i]/ainf
   
    # # Plot the residual
    # plt.semilogy(np.linspace(0, time_steps-1, time_steps), residualmclgs, label='lgs', linestyle='-')
    # plt.legend()
    # plt.show()

    # cp = calc_cp(phi, xpts, ypts[0]-ypts[1], Minf, type='small')
    # plt.plot(xpts, -cp, marker='.', label='pjc')
    # plt.xlim([-.25, 1.25])
    # plt.title(f'$C_p$ at Mach={Minf}')
    # plt.show()

    # if Minf < .8:
    #     mach_levels = [.7, .75, .8]
    # else:
    #     mach_levels = [1, 1.1, 1.2, 1.3]

    # fig, ax = plt.subplots()
    # ax.contour(xv, yv, mach, mach_levels)
    # ax.set_title("Mach")
    # ax.set_xlim([-.5, 1.5])
    # ax.set_ylim([0, 1.75])
    # # plt.close()
    # plt.show()

if __name__ == '__main__':
    main()