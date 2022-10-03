# Austin Ryan

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.sparse import linalg

def gs_black(f, P, h, dt, w):
    # BLACK
    for i in range(len(P)):
        for j in range(len(P[i])):
            if (i+j) % 2 == 0: # black
                #print('b', i, j)
                # CORNER
                if i == 0 and j == 0: # Bottom left
                    a = P[i][j+1] + P[i+1][j]
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)

                elif i == 0 and j == len(P[i])-1: # top left
                    a = P[i][j-1] + P[i+1][j]
                    P[i][j] = w*.5*(-f[i][j]*h/dt+a)

                elif i == len(P)-1 and j == 0: # top right
                    a = P[i][j+1] + P[i-1][j]
                    P[i][j] = -.5*(f[i][j]*h/dt-a)

                elif i == len(P)-1 and j == len(P[i])-1: # bottom right
                    a = P[i][j-1] + P[i-1][j]
                    P[i][j] = -.5*(f[i][j]*h/dt-a)
                # BORDER
                elif i == 0: # LEFT
                    a = P[i][j+1] + P[i][j-1] + P[i+1][j]
                    P[i][j] = -1/3*(f[i][j]*h/dt-a)

                elif i == len(P) - 1: # RIGHT
                    a = P[i][j+1] + P[i][j-1] + P[i-1][j]
                    P[i][j] = 1/3*(-f[i][j]*h/dt+a)

                elif j == 0: # BOTTOM
                    a = P[i][j+1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = -1/3*(f[i][j]*h/dt-a)

                elif j == len(P[i]) - 1: # TOP
                    a = P[i][j-1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = -1/3*(f[i][j]*h/dt-a)

                else:
                    a = P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]
                    P[i][j] = 1/4*(-f[i][j]*h/dt+a)


def gs_red(f, P, h, dt, w):
    # RED
    for i in range(len(P)):
        for j in range(len(P[i])):
            if (i+j) % 2 == 1: # red
                #print('r', i, j)
                # CORNER
                if i == 0 and j == 0: # Bottom left
                    a = P[i][j+1] + P[i+1][j]
                    P[i][j] = -.5*(f[i][j]*h/dt-a)

                elif i == 0 and j == len(P[i])-1: # top left
                    a = P[i][j-1] + P[i+1][j]
                    P[i][j] = -.5*(f[i][j]*h/dt-a)

                elif i == len(P)-1 and j == 0: # top right
                    a = P[i][j+1] + P[i-1][j]
                    P[i][j] = -.5*(f[i][j]*h/dt-a)

                elif i == len(P)-1 and j == len(P[i])-1: # bottom right
                    a = P[i][j-1] + P[i-1][j]
                    P[i][j] = -.5*(f[i][j]*h/dt-a)
                # BORDER
                elif i == 0: # LEFT
                    
                    a = P[i][j+1] + P[i][j-1] + P[i+1][j]
                    P[i][j] = -1/3*(f[i][j]*h/dt-a)

                elif i == len(P) - 1: # RIGHT
                    a = P[i][j+1] + P[i][j-1] + P[i-1][j]
                    P[i][j] = -1/3*(f[i][j]*h/dt-a)

                elif j == 0: # BOTTOM
                    a = P[i][j+1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = -1/3*(f[i][j]*h/dt-a)

                elif j == len(P[i]) - 1: # TOP
                    a = P[i][j-1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = -1/3*(f[i][j]*h/dt-a)

                else:
                    a = P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]
                    P[i][j] = -1/4*(f[i][j]*h/dt-a)

def gauss_seidel(U, V, P, h, dt):
    w = 1.6
    f = np.zeros((N, N))
    for i in range(len(f)):
        for j in range(len(f[i])):
            b = U[i+2][j+1]-U[i+1][j+1]
            c = V[i+1][j+2]-V[i+1][j+1]
            f[i][j] = b+c

    gs_residual = 0
    for i in range(100):

        gs_black(f, P, h, dt, w)
        gs_red(f, P, h, dt, w)
        
        gs_residual = np.sum(abs(P-f))
        print(gs_residual)


def smart(U, V, i, j, flux):
    """Returns convective flux"""
    # origin is top right, positive is down and right
    if flux == 'f': # u trans vel, u trans quant
        q = (U[i+1][j+1] + U[i+2][j+1])/2
        if q > 0:
            state1 = U[i+2][j+1]
            state0 = U[i+1][j+1]
            statem1 = U[i+0][j+1]
        if q <= 0:
            state1 = U[i+1][j+1]
            state0 = U[i+2][j+1]
            statem1 = U[i+3][j+1]
    elif flux == 'g': # v trans vel, v trans quant
        q = (V[i+1][j+1] + V[i+1][j+2])/2
        if q > 0:
            state1 = V[i+1][j+2]
            state0 = V[i+1][j+1]
            statem1 = V[i+1][j+0]
        if q <= 0:
            state1 = V[i+1][j+1]
            state0 = V[i+1][j+2]
            statem1 = V[i+1][j+3]

    elif flux == 'Hx': # v trans vel, u trans quant
        q = (V[i+1][j+1] + V[i+0][j+1])/2
        if q > 0:
            state1 = U[i+1][j+1]
            state0 = U[i+1][j+0]
            statem1 = U[i+1][j-1]
        if q <= 0:
            state1 = U[i+1][j+0]
            state0 = U[i+1][j+1]
            statem1 = U[i+1][j+2]

    elif flux == 'Hy': # u trans vel, v trans quant
        q = (U[i+1][j+1] + U[i+1][j+0])/2
        if q > 0:
            state1 = U[i+1][j+1]
            state0 = U[i+0][j+1]
            statem1 = U[i-1][j+1]
        if q <= 0:
            state1 = U[i+0][j+1]
            state0 = U[i+1][j+1]
            statem1 = U[i+2][j+1]

    if state1-statem1 == 0 and state1-statem1 == 0:
        state0hat = state1
    elif state1-statem1 == 0:
        state0hat = 0 # to tigger the first condition in the next if statement
    else:
        state0hat = (state0-statem1)/(state1-statem1)

    if state0hat >= 1 or state0hat <= 0:
        state12hat = state0hat
    elif 0 < state0hat <= 1/6:
        state12hat = 3*state0hat
    elif 1/6 < state0hat <= 5/6:
        state12hat = 3/8*(2*state0hat+1)
    elif 5/6 < state0hat < 1:
        state12hat = 1

    state12 = state12hat*(state1-statem1)+statem1

    flux = state12

    return flux

def quick(U, V, i, j, flux):
    """Returns convective flux"""
    # origin is top right, positive is down and right
    if flux == 'f': # u trans vel, u trans quant
        q = (U[i+1][j+1] + U[i+2][j+1])/2
        if q > 0:
            phi = (3*U[i+2][j+1]+6*U[i+1][j+1]-U[i+0][j+1])/8
        elif q <= 0:
            phi = (3*U[i+1][j+1]+6*U[i+2][j+1]-U[i+3][j+1])/8
        
    elif flux == 'g': # v trans vel, v trans quant
        q = (V[i+1][j+1] + V[i+1][j+2])/2
        if q > 0:
            phi = (3*V[i+1][j+2]+6*V[i+1][j+1]-V[i+1][j+0])/8
        elif q <= 0:
            phi = (3*V[i+1][j+1]+6*V[i+1][j+2]-V[i+1][j+3])/8

    elif flux == 'Hx': # v trans vel, u trans quant
        q = (V[i+1][j+1] + V[i+0][j+1])/2
        if q > 0:
            phi = (3*U[i+1][j+1]+6*U[i+1][j+0]-U[i+1][j-1])/8
        elif q <= 0:
            phi = (3*U[i+1][j+0]+6*U[i+1][j+1]-U[i+1][j+2])/8

    elif flux == 'Hy': # u trans vel, v trans quant
        q = (U[i+1][j+1] + U[i+1][j+0])/2
        if q > 0:
            phi = (3*V[i+1][j+1]+6*V[i+0][j+1]-V[i-1][j+1])/8
        elif q <= 0:
            phi = (3*V[i+0][j+1]+6*V[i+1][j+1]-V[i+2][j+1])/8
    
    flux = q*phi
    return flux

def flux(U, V, F, G, Hx, Hy, nu, h):
    for i in range(len(F)):
        for j in range(len(F[i])):
            F[i][j] = quick(U, V, i, j, 'f')-nu/h*(U[i+2][j+1]-U[i+1][j+1])
            G[i][j] = quick(U, V, i, j, 'g')-nu/h*(V[i+1][j+2]-V[i+1][j+1])
    
    for i in range(len(Hx)):
        for j in range(len(Hx[i])):
            if j == 0 or j == len(Hx[i])-1:
                Hx[i][j] = 0-nu/h*(U[i+1][j+1]-U[i+1][j+0]) # No transport vel on horz walls
            else:
                Hx[i][j] = quick(U, V, i, j, 'Hx')-nu/h*(U[i+1][j+1]-U[i+1][j+0])

    for i in range(len(Hy)):
        for j in range(len(Hy[i])):
            if i == 0 or i == len(Hy)-1:
                Hy[i][j] = 0-nu/h*(U[i+1][j+1]-U[i+0][j+1]) # No transport vel on vert walls
            else:
                Hy[i][j] = quick(U, V, i, j, 'Hy')-nu/h*(V[i+1][j+1]-V[i+0][j+1])

def update_vel1(U, V, F, G, Hx, Hy, dt, h, N): 
    """Half time step update. Loops over only the interior vel nodes"""
    # U = 0 on vertical walls
    # V = 0 on horizontal walls
    # Velocity i,j

    for i in range(2, N+1):
        for j in range(1, N+1):
            F1 = (F[i-1][j-1] - F[i-2][j-1])/h
            Hx1 = (Hx[i-1][j] - Hx[i-1][j-1])/h
            U[i][j] = U[i][j] - dt*(F1 + Hx1) # Only has to be solved at interior U

    for i in range(1, N+1):
        for j in range(2, N+1):
            G1 = (G[i-1][j-1]-G[i-1][j-2])/h
            Hy1 = (Hy[i][j-1]-Hy[i-1][j-1])/h
            V[i][j] = V[i][j] - dt*(G1 + Hy1) # Only has to be solved at interior V

def update_vel2(U, V, P, dt, h, N):
    """Full time step update. Loops over only the interior vel nodes"""
    # U Velocity i,j
    for i in range(2, N+1):
        for j in range(1, N+1):
            U[i][j] = U[i][j] - dt/h*(P[i-1][j-1]-P[i-2][j-1])
            
    # V Velocity i,j
    for i in range(1, N+1):
        for j in range(2, N+1):
            V[i][j] = V[i][j] - dt/h*(P[i-1][j-1]-P[i-1][j-2])

def Poissona(N,h):
    """N is number of cells"""
    data = np.ones((5,N*N)) # diagonals for making a sparse matrix, number of diags and size of matrix
    data[0,:] *= 1
    data[1,:] *= 1
    data[2,:] *= -4
    data[3,:] *= 1
    data[4,:] *= 1
    diags = np.array([-N, -1, 0, 1, N]) # which diags are nonzero
    A = sparse.spdiags(data, diags, N*N, N*N,'csr')

    # Corner cells corrections
    A[0, 0] = -2 # Top left
    A[N-1, N-1] = -2 # bottom left
    A[N-1, N+0] = 0
    A[N*N-N, N*N-N] = -2 # top right
    A[N*N-N, N*N-N-1] = 0
    A[N*N-1, N*N-1] = -2 # bottom right

    for i in range(1, N-1): # Vertical wall cell corrections
        A[i, i] = -3
        A[N*N-i-1, N*N-i-1] = -3
    
    tops = np.linspace(N, N*N-N, N-1)
    tops = np.delete(tops, -1)
    for i in tops: # Horizontal wall cell corrections
        A[i, i] = -3 # Top
        A[i, i-1] = 0 # No reference above top
        A[i+N-1, i+N-1] = -3 # Bottom
        A[i+N-1, i+N-1+1] = 0 # No reference below bottom

    print(A.toarray())
    print(linalg.inv(A))
    A[0, 1] = 0
    A[0, N] = 0
    A *= 1/h**2
    A[0, 0] = 1
    return A

def Poissonb(U, V, N, dt, h):
    NN = N*N
    B = np.zeros(NN)
    
    # F,G flux i,j
    for k in range(len(B)):
        if k == 0:
            j = 0
            i = 0
        elif k % N == 0:
            i += 1
            j = 0
        
        au = U[i+2][j+1]-U[i+1][j+1]
        av = V[i+1][j+2]-V[i+1][j+1]

        B[k] = (au+av)/(dt*h)
        j += 1
    B[0] = 0
    return B

def ghost_cells(U, V, uwall):
    # Horizontal wall ghost cells
    for i in range(len(U)):
        U[i][0] = 2*uwall - U[i][1] # Top wall
        U[i][-1] = -U[i][-2] # Bottom wall
    for i in range(len(V)):
        V[i][0] = V[i][2] # Top wall
        V[i][-1] = V[i][-3] # bottom wall
    U[1][0] = 0
    U[4][0] = 0

    # Vertical wall ghost cells
    for j in range(len(V[0])):
        V[0][j] = -V[1][j]
        V[-1][j] = -V[-2][j]
    for j in range(len(U[0])):
        U[0][j] = U[2][j]
        U[-1][j] = U[-3][j]

def residual(F, G, Hx, Hy, P, h):
    Ri2 = np.zeros((N-1,N)) # R_(i+1/2,j) one for each interior vertical edge
    Rj2 = np.zeros((N,N-1)) # R_(i,j+1/2) one for each interior horizontal edge

    for i in range(len(Ri2)):
        for j in range(len(Ri2[i])):
            ar = F[i+1][j]+P[i+1][j]-F[i][j]-P[i][j]
            br = Hx[i+1][j+1]-Hx[i+1][j]
            Ri2[i][j] = h*ar + h*br

    for i in range(len(Rj2)):
        for j in range(len(Rj2[i])):
            ar = G[i][j+1]+P[i][j+1]-G[i][j]-P[i][j]
            br = Hy[i+1][j+1]-Hy[i][j+1]
            Rj2[i][j] = h*ar + h*br
    
    RL1 = np.sum(abs(Ri2)) + np.sum(abs(Rj2))
    
    return RL1

def main(Re, N):
    rho = 1
    L = 1
    uwall = 1
    Nt = 1

    nu = uwall*L/Re
    h = L/N
    a = h**2/(4*nu)
    b = 4*nu/uwall**2
    beta = .5
    dt = beta*min(a, b)
    print('nu', 'h', 'dt')
    print(nu, h, dt)

    # Origin is the top right of the mesh
    P = np.zeros((N, N)) # Pressure, one value per cell
    U = np.zeros((N+3, N+2)) # Horizontal velocity, one value per vertical edge and one layer of ghost values all around
    V = np.zeros((N+2, N+3)) # vertical velocity, one value per horizontal edge and one layer of ghost values all around

    F = np.zeros((N,N)) # horizontal x-momentum flux, one value per cell
    G = np.zeros((N,N)) # vertical y-momentum flux, one value per cell
    Hx = np.zeros((N+1,N+1)) # vertical x-momentum flux, one value per grid node
    Hy = np.zeros((N+1,N+1)) # horizontal y-momentum flux, one value per grid node
    
    A = Poissona(N, h)

    for i in range(15):
        
        ghost_cells(U, V, uwall)
        print('ghost u\n', np.around(U, 4))
        print('v', np.around(V, 4))

        flux(U, V, F, G, Hx, Hy, nu, h)
        print('f', np.around(F, 4))
        print('g', np.around(G, 4))
        print('hx', np.around(Hx, 4))
        print('hy', np.around(Hy, 4))

        update_vel1(U, V, F, G, Hx, Hy, dt, h, N)
        print()
        print('1 u\n', np.around(U, 4))
        print('1 v\n', np.around(V, 4))

        B = Poissonb(U, V, N, dt, h)
        #print('a', A.toarray())
        #print('b', B)
        #print()
        
        P = linalg.spsolve(A, np.transpose(B))
        P = np.reshape(P, (N,N), order='F')
        print('p', P)
        update_vel2(U, V, P, dt, h, N)
        print('2 u\n', np.around(U, 4))
        print('2 v\n', np.around(V, 4))

        RL1 = residual(F, G, Hx, Hy, P, h)
        print(i, RL1)
    print()
    print(np.around(U, 4))

if __name__ == '__main__':
    Re = 100
    N = 3
    #main(Re, N)
    Poissona(3,5)
