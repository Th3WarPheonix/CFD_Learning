
def solve_tridiagonal(n:int, A, B, C, D):
    """Solves tridiagonal matrix using the Thomas algorithm in O(n) time
    All vectors should be the same length. The last element of A and C
    are not used.
    
    Parameters
    ----------
    n : size of square array
    A : lower diagonal of coefficient matrix
    B : main  diagonal of coefficient matrix
    C : upper diagonal of coefficient matrix
    D : the known vector"""

    G = np.empty_like(C)
    R = np.empty_like(D)
    X = np.empty_like(D)

    G[0] = C[0]/B[0]
    R[0] = D[0]/B[0]

    for i in range(1, n-1):
        G[i] = C[i]/(B[i] - A[i-1]*G[i-1])
        R[i] = (D[i] - A[i-1]*R[i-1]) / (B[i] - A[i-1]*G[i-1])
    R[i+1] = (D[i+1] - A[i]*R[i]) / (B[i+1] - A[i]*G[i])
    X[i+1] = R[i+1]
    for i in range(n-2, -1, -1):
        X[i] = R[i] - G[i]*X[i+1]

    return X

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

def calc_cp(phi, xpts, dy, Minf, type='full', gamma=1.4):
    """Calculates the coefficient of pressure along 1 row of points
    based on the potential equations.

    Parameters
    ----------
    phi : the values of the potential for at least the row of points on 
        the surface and the row of points above the surface
    xpts : the array of x values
    dy : the distance between the first row and second row of points (y[0]-y[1])
    Minf : freestream Mach number
    type : "full" for full potential, "small" for small disturbance
    """
    densityinf = 1
    if type == 'full':
        pinf = 1
        ainf = np.sqrt(gamma*pinf/densityinf)
        Vinf = Minf*ainf
    else:
        pinf = 1/gamma/Minf**2
        Vinf = 1
    
    v = np.zeros_like(xpts)
    u = np.zeros_like(xpts)
    for i in range(1, len(xpts)-1):
        v[i] = (phi[0, i] - phi[1, i])/dy
        u[i] = (phi[0, i+1] - phi[0, i-1])/(xpts[i+1] - xpts[i-1])+Vinf

    pressure = pinf*(1-(gamma-1)/2*Minf**2*((u**2+v**2)/Vinf**2-1))**(gamma/(gamma-1))
    cp = 2*(pressure-pinf)/densityinf/Vinf**2
    return cp

def plot_residual(xpts, ypts, res, t=None):
    """Plot residual at each iteration or at the beginning or end
    
    Parameters
    ----------
    xpts : matrix of same size as residual of x values
    ypts : matrix of same size as residual of y values
    res : matrix of resdiual values
    t : optionally make the time step the title of the figure"""
    xv, yv = np.meshgrid(xpts, ypts)
    sc = plt.scatter(xv, yv, c=res)
    plt.colorbar(sc)
    if t:
        plt.title(f'{t} res')
    else:
        plt.title(f'Residual')
    plt.show() 

def murman_cole_lgs(time_steps, Minf, xpts, ypts, phi, xpts1, xpts2, xpts3, Vinf, dymin, fairf, chord, gamma=1.4, timing=False):
    """Solves the transonic small disturbance equations using the
    Murman-Cole method and line Guass-Seidel. Murman-Cole switches the
    differencing scheme based on if the flow is subsonic, sonic, or
    supersonic at the current point or the point behind the current"""

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
    for t in range(time_steps):
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

                # Dealing with the y-terms first since always present
                # and are not affected by the Murman-Cole equation
                A[j-1] = -ypts01 * ypts11
                B[j] = ypts11 * (ypts10 + ypts01)
                C[j] = -ypts10 * ypts11
                D[j] = 0

                # Employing the Murman-Cole switching
                bigA0 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phi[j][i+1] - phi[j][i-1])*xpts11
                bigA1 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phi[j][i] - phi[j][i-2])*xpts02
                if bigA0 > 0:
                    mu0 = 0
                elif bigA0 <= 0:
                    mu0 = 1

                if bigA1 > 0:
                    mu1 = 0
                elif bigA1 <= 0:
                    mu1 = 1

                # Following the book method of employing mu, could also
                # just move these into the if statements
                B[j] += (1-mu0)*bigA0*xpts11 * (xpts10 + xpts01)
                D[j] += (1-mu0)*bigA0*xpts11 * (phi[j, i+1]*xpts10 + phi[j, i-1]*xpts01)

                B[j] += -mu1*bigA1*xpts01*xpts02
                D[j] += -mu1*bigA1*xpts02*((phi[j][i-1]-phi[j][i-2])*xpts12 + phi[j][i-1]*xpts01)

            # Applying boundary conditions
            if i < nxpts1: # in front of airfoil
                C[0] = -1 # bottom boundary condition
                B[0] = 1 # bottom boundary condition
                D[0] = 0 # bottom boundary condition
                A[-2] = 0 # top boundary condition
                B[-1] = 1 # top boundary condition
                D[-1] = phi[-1, i] # top boundary condition            
            elif nxpts1 <= i <= nxpts1 + nxpts2 - 1: # on top of airfoil
                C[0] =-1 # bottom boundary condition
                B[0] = 1 # bottom boundary condition
                D[0] =-Vinf*dymin*(chord/2-xpts[i])/np.sqrt(fairf**2-(xpts[i]-chord/2)**2) # bottom boundary condition
                A[-2] = 0 # top boundary condition
                B[-1] = 1 # top boundary condition
                D[-1] = phi[-1, i] # top boundary condition
            elif nxpts1 + nxpts2 - 1 <= i <= nxpts: # behind of airfoil
                C[0] =-1 # bottom boundary condition
                B[0] = 1 # bottom boundary condition
                D[0] = 0 # bottom boundary condition
                A[-2] = 0 # top boundary condition
                B[-1] = 1 # top boundary condition
                D[-1] = phi[-1, i] # top boundary condition
            phi[:, i] = solve_tridiagonal(nypts, A, B, C, D)

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

                bigA0 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phi[j][i+1] - phi[j][i-1])*xpts11
                bigA1 = 1 - Minf**2 - (gamma+1)*Minf**2/Vinf*(phi[j][i] - phi[j][i-2])*xpts02
                if bigA0 > 0:
                    mu0 = 0
                elif bigA0 <= 0:
                    mu0 = 1

                if bigA1 > 0:
                    mu1 = 0
                elif bigA1 <= 0:
                    mu1 = 1

                Dxxa = (phi[j, i+1]-phi[j, i])*xpts10 - (phi[j, i]-phi[j, i-1])*xpts01
                Dxx = Dxxa*xpts11

                Dxx1a = (phi[j][i] - phi[j][i-1])*xpts01 - (phi[j][i-1] - phi[j][i-2])*xpts12
                Dxx1 = Dxx1a*xpts02

                Dyya = (phi[j+1, i]-phi[j, i])*ypts10 - (phi[j, i]-phi[j-1, i])*ypts01
                Dyy = Dyya*ypts11

                res[j, i] = (1-mu0)*bigA0*Dxx + mu1*bigA1*Dxx1 + Dyy
        residual[t] = np.max(np.max(abs(res)))
        print(residual[t])
        
    elapsed_time = time.time() - startlgs
    if timing:
        print('Line Gauss-Seidel Elapsed Time: {:.3}'.format(elapsed_time))

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
    xpts1 = -np.flip(stretch_mesh(dxmin, xmax, ptsx))[:-1] # in front of airfoil
    xpts2 = np.linspace(0, chord, ptsx2) # on airfoil
    xpts3 = stretch_mesh(dxmin, xmax, ptsx)[1:] + chord # behind airfoil
    xpts = np.hstack((xpts1, xpts2, xpts3))
    phi = np.zeros((len(ypts), len(xpts)))
    Minf = 0.5
    Vinf = 1
    ainf = Vinf/Minf

    # xv, yv = np.meshgrid(xpts, ypts)
    # sc = plt.scatter(xv, yv, c=phi)#, vmin=-30, vmax=30)
    # plt.colorbar(sc)
    # plt.title("initial condition")
    # plt.savefig("initial condition")
    # # plt.close()
    # plt.show()

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

int main(){
}