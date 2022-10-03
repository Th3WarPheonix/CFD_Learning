# Austin Ryan
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
import time

def gs_black(f, P, h, dt, w):
    # BLACK
    for i in range(len(P)):
        for j in range(len(P[i])):
            if (i+j) % 2 == 0: # black
                #print('b', i, j)
                # CORNER
                if i == 0 and j == 0: # Bottom left
                    a = P[i][j+1] + P[i+1][j]
                    P[i][j] = w*.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif i == 0 and j == len(P[i])-1: # top left
                    a = P[i][j-1] + P[i+1][j]
                    P[i][j] = w*.5*(-f[i][j]*h/dt+a)+(1-w)*P[i][j]

                elif i == len(P)-1 and j == 0: # top right
                    a = P[i][j+1] + P[i-1][j]
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif i == len(P)-1 and j == len(P[i])-1: # bottom right
                    a = P[i][j-1] + P[i-1][j]
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j]
                # BORDER
                elif i == 0: # LEFT
                    a = P[i][j+1] + P[i][j-1] + P[i+1][j]
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif i == len(P) - 1: # RIGHT
                    a = P[i][j+1] + P[i][j-1] + P[i-1][j]
                    P[i][j] = w*1/3*(-f[i][j]*h/dt+a)+(1-w)*P[i][j]

                elif j == 0: # BOTTOM
                    a = P[i][j+1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif j == len(P[i]) - 1: # TOP
                    a = P[i][j-1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                else:
                    a = P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]
                    P[i][j] = w*1/4*(-f[i][j]*h/dt+a)+(1-w)*P[i][j]

def gs_red(f, P, h, dt, w):
    # RED
    for i in range(len(P)):
        for j in range(len(P[i])):
            if (i+j) % 2 == 1: # red
                #print('r', i, j)
                # CORNER
                if i == 0 and j == 0: # Bottom left
                    a = P[i][j+1] + P[i+1][j]
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif i == 0 and j == len(P[i])-1: # top left
                    a = P[i][j-1] + P[i+1][j]
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif i == len(P)-1 and j == 0: # top right
                    a = P[i][j+1] + P[i-1][j]
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif i == len(P)-1 and j == len(P[i])-1: # bottom right
                    a = P[i][j-1] + P[i-1][j]
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j]
                # BORDER
                elif i == 0: # LEFT
                    
                    a = P[i][j+1] + P[i][j-1] + P[i+1][j]
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif i == len(P) - 1: # RIGHT
                    a = P[i][j+1] + P[i][j-1] + P[i-1][j]
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif j == 0: # BOTTOM
                    a = P[i][j+1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                elif j == len(P[i]) - 1: # TOP
                    a = P[i][j-1] + P[i+1][j] + P[i-1][j]
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

                else:
                    a = P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]
                    P[i][j] = w*-1/4*(f[i][j]*h/dt-a)+(1-w)*P[i][j]

def gsres(P, f):
    residual = 0
    for i in range(len(P)):
        for j in range(len(P[i])):
            # CORNER
            if i == 0 and j == 0: # Bottom left
                residual += abs(f[i][j] - (-2*P[i][j] + P[i][j+1] + P[i+1][j]))

            elif i == 0 and j == len(P[i])-1: # top left
                residual += abs(f[i][j] - (-2*P[i][j] + P[i][j-1] + P[i+1][j]))

            elif i == len(P)-1 and j == 0: # top right
                residual += abs(f[i][j] - (-2*P[i][j] + P[i][j+1] + P[i-1][j]))

            elif i == len(P)-1 and j == len(P[i])-1: # bottom right
                residual += abs(f[i][j] - (-2*P[i][j] + P[i][j-1] + P[i-1][j]))

            # BORDER
            elif i == 0: # LEFT
                residual += abs(f[i][j] - (-3*P[i][j] + P[i][j+1] + P[i][j-1] + P[i+1][j]))
                
            elif i == len(P) - 1: # RIGHT
                residual += abs(f[i][j] - (-3*P[i][j] + P[i][j+1] + P[i][j-1] + P[i-1][j]))

            elif j == 0: # BOTTOM
                residual += abs(f[i][j] - (-3*P[i][j] + P[i][j+1] + P[i+1][j] + P[i-1][j]))

            elif j == len(P[i]) - 1: # TOP
                residual += abs(f[i][j] - (-3*P[i][j] + P[i][j-1] + P[i+1][j] + P[i-1][j]))

            else:
                residual += abs(f[i][j] - (-4*P[i][j] + P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]))

    return residual

def gauss_seidel(U, V, P, h, dt, N):
    w = 1.4
    f = np.zeros((N, N))
    for i in range(len(f)):
        for j in range(len(f[i])):
            b = U[i+2][j+1]-U[i+1][j+1]
            c = V[i+1][j+2]-V[i+1][j+1]
            f[i][j] = b+c

    residual = 1
    kiter = 0
    while residual > 1e-5:

        gs_black(f, P, h, dt, w)
        gs_red(f, P, h, dt, w)
        residual = gsres(P, f)
        print(residual)
        kiter += 1
    print(f'\tk {kiter}')


def Poissona(N, h):
    """N is number of cells"""
    data = np.ones((5,N*N)) # diagonals for making a sparse matrix, number of diags and size of matrix
    data[0,:] *= 1
    data[1,:] *= 1
    data[2,:] *= -4
    data[3,:] *= 1
    data[4,:] *= 1
    # left, bottom, center, top, right
    diags = np.array([-N, -1, 0, 1, N]) # which diags are nonzero
    A = sparse.spdiags(data, diags, N*N, N*N,'csr')
    #print(A.toarray())
    # Corner cells corrections
    A[0, 0] = -2 # bottom left
    A[N-1, N-1] = -2 # top left
    A[N-1, N+0] = 0
    A[N*N-N, N*N-N] = -2 # bottom right
    A[N*N-N, N*N-N-1] = 0
    A[N*N-1, N*N-1] = -2 # top right

    for i in range(1, N-1): # Vertical wall cell corrections
        A[i, i] = -3
        A[N*N-i-1, N*N-i-1] = -3
    
    tops = np.linspace(N, N*N-N, N-1)
    tops = np.delete(tops, -1) # removes unnecessary index
    print(tops)
    for i in tops: # Horizontal wall cell corrections
        i = int(i)
        A[i, i] = -3 # Top
        A[i, i-1] = 0 # No reference above top
        A[i+N-1, i+N-1] = -3 # Bottom
        A[i+N-1, i+N-1+1] = 0 # No reference below bottom

    A[0, 1] = 0
    A[0, N] = 0
    #A *= 1/h**2 # Change B too
    A[0, 0] = 1
    return A

def poissonb(U, V, h, dt, N):
    #print(U)
    #print(V)
    f = np.zeros((N, N))
    for i in range(len(f)):
        for j in range(len(f[i])):
            b = U[i+2][j+1]-U[i+1][j+1]
            c = V[i+1][j+2]-V[i+1][j+1]
            f[i][j] = b+c
            #print(U[i+2][j+1], U[i+1][j+1])
            #print(V[i+1][j+2], V[i+1][j+1])
            #print(i, j, f[i][j])
    #print('f\n', f)
    B = np.reshape(f, N**2, order='c')
    #print('b\n', B)
    #B *= 1/(dt*h)
    B *= h/dt
    return B

def solve_poisson(A, U, V, h, dt, N):
    B = poissonb(U, V, h, dt, N)
    #print(B.shape, A.shape)
    P = linalg.spsolve(A, B)
    #print(h, dt)
    #print('b\n', B)
    #print(P)
    P = np.reshape(P, (N, N), order='c')
    #print(P)
    return P

def ghost_vel(U, V, uwall):
    for i in range(len(U)): # Horizontal wall
        U[i][-1] = 2*uwall-U[i][-2] # top wall
        U[i][0]  = -U[i][1] # bottom wall
    for i in range(len(V)):
        V[i][-1] = V[i][-3] # top wall
        V[i][0]  = V[i][2] # bottom wall

    for i in range(len(U[0])): # Vertical wall
        U[0][i]  = U[2][i] # left wall
        U[-1][i] = U[-3][i] # right wall
    for i in range(len(V[0])):
        V[0][i]  = -V[1][i] # left wall
        V[-1][i] = -V[-2][i] # right wall
    #U[1][-1] = 0
    #U[4][-1] = 0

def smart(q, ull, ul, ur, urr):
    if q > 0:
        phim1 = ull
        phi0 = ul
        phi1 = ur
    elif q <= 0:
        phim1 = urr
        phi0 = ur
        phi1 = ul

    if (phi1-phim1) == 0 and (phi0-phim1) == 0:
        phi12 = phi0
    elif (phi1-phim1) == 0:
        phi12 = phi0
    else:
        phihat = (phi0-phim1)/(phi1-phim1)
        if phihat <= 0 or phihat >= 1:
            phi12hat = phihat
        elif 0 < phihat <= 1/6:
            phi12hat = 3*phihat
        elif 1/6 < phihat <= 5/6:
            phi12hat = 3/8*(2*phihat+1)
        elif 5/6 < phihat <= 1:
            phi12hat = 1

        phi12 = phi12hat*(phi1-phim1)+phim1
    return phi12

def quickF(q, ull, ul, ur, urr):
    if q > 0: # Positive is to right
        phi = (3*ur+6*ul-ull)/8
    elif q <= 0: # negative is to left
        phi = (3*ul+6*ur-urr)/8
    return phi

def quickG(q, vtt, vt, vb, vbb):
    if q > 0: # Positive is up
        phi = (3*vt+6*vb-vbb)/8
    elif q <= 0: # negative is down
        phi = (3*vb+6*vt-vtt)/8
    return phi

def quickHx(q, utt, ut, ub, ubb):
    if q > 0: # Positive is up
        phi = (3*ut+6*ub-ubb)/8
    elif q <= 0: # negative is down
        phi = (3*ub+6*ut-utt)/8
    return phi

def quickHy(q, vll, vl, vr, vrr):
    if q > 0: # Positive is to right
        phi = (3*vr+6*vl-vll)/8
    elif q <= 0: # negative is to left
        phi = (3*vl+6*vr-vrr)/8
    return phi
    
def flux(U, V, F, G, Hx, Hy, h, nu, N):
    for i in range(len(F)):
        for j in range(len(F[i])):
            urr = U[i+3][j+1]
            ur  = U[i+2][j+1]
            ul  = U[i+1][j+1]
            ull = U[i+0][j+1]

            vtt = V[i+1][j+3]
            vt  = V[i+1][j+2]
            vb  = V[i+1][j+1]
            vbb = V[i+1][j+0]
            qF = (ur+ul)/2
            qG = (vt+vb)/2
            phiF = quickF(qF, ull, ul, ur, urr)
            #phiG = quickG(qG, vtt, vt, vb, vbb)
            #phiF = smart(qF, ull, ul, ur, urr)
            phiG = smart(qG, vbb, vb, vt, vtt)
            F[i][j] = qF*phiF - nu/h*(ur-ul)
            G[i][j] = qG*phiG - nu/h*(vt-vb)

    for i in range(len(Hx)):
        for j in range(len(Hx[i])):
            ut  = U[i+1][j+1]
            ub  = U[i+1][j+0]

            vr  = V[i+1][j+1]
            vl  = V[i+0][j+1]
            
            qHx = (vr+vl)/2
            qHy = (ut+ub)/2

            if i == 0: # Left wall
                qHy = 0
                phiHy = 0
            elif i == N: # Right wall
                qHy = 0
                phiHy = 0
            else:
                vrr = V[i+2][j+1]
                vll = V[i-1][j+1]
                #phiHy = quickHy(qHy, vll, vl, vr, vrr)
                phiHy = smart(qHy, vll, vl, vr, vrr)
            
            if j == 0: # Bottomm wall
                qHx = 0
                phiHx = 0
            elif j == N: # Top wall
                qHx == 0
                phiHx = 0
            else:
                utt = U[i+1][j+2]
                ubb = U[i+1][j-1]
                #phiHx = quickHx(qHx, utt, ut, ub, ubb)
                phiHx = smart(qHx, ubb, ub, ut, utt)

            Hx[i][j] = qHx*phiHx - nu/h*(ut-ub)
            Hy[i][j] = qHy*phiHy - nu/h*(vr-vl)

def update1(U, V, F, G, Hx, Hy, h, dt, N):
    #print(h, dt, )
    #print(h, dt, N)
    # Update U
    for i in range(2, N+1):
        for j in range(1, N+1):
            F1 = F[i-1][j-1] - F[i-2][j-1] # Fr - Fl
            Hx1 = Hx[i-1][j] - Hx[i-1][j-1] # Hxu - Hxd
            U[i][j] = U[i][j] - dt/h*(F1+Hx1)
    # Update V
    for i in range(1, N+1):
        for j in range(2, N+1):
            G1 = G[i-1][j-1] - G[i-1][j-2] # Gt - Gb
            Hy1 = Hy[i][j-1] - Hy[i-1][j-1] # Hyr - Hyl
            V[i][j] = V[i][j] - dt/h*(G1+Hy1)

def update2(U, V, P, h, dt, N):
    for i in range(2, N+1):
        for j in range(1, N+1):
            U[i][j] = U[i][j] - dt/h*(P[i-1][j-1]-P[i-2][j-1]) # u - dt*h*(Pr-Pl)

    for i in range(1, N+1):
        for j in range(2, N+1):
            V[i][j] = V[i][j] - dt/h*(P[i-1][j-1]-P[i-1][j-2]) # u - dt*h*(Pt-Pb)

def residual(F, G, Hx, Hy, P, h, N):
    Ri2 = np.zeros((N-1, N))
    Rj2 = np.zeros((N, N-1))
    
    for i in range(len(Ri2)):
        for j in range(len(Ri2[i])):
            F1 = F[i+1][j] - F[i][j]
            P1i = P[i+1][j] - P[i][j]
            Hx1 = Hx[i+1][j+1] - Hx[i+1][j-0]
            Ri2[i][j] = h*(F1+P1i+Hx1)

    for i in range(len(Rj2)):
        for j in range(len(Rj2[i])):
            G1 = G[i][j+1] - G[i][j]
            P1j = P[i][j+1] - P[i][j]
            Hy1 = Hy[i+1][j+1] - Hy[i-0][j+1]
            Rj2[i][j] = h*(G1+P1j+Hy1)

    '''res = np.sum(abs(Ri2)) + np.sum(abs(Rj2))
    return res'''
    return np.sum(abs(Ri2)), np.sum(abs(Rj2))

def plot_vel(U, V, h):
    if Re == 100:
        u100 = [1, .84123, .78871, .73722, .68717, .23151, .00332, -.13641, -.20581, -.21090, -.15662, -.10150, -.06434, -.04775, -.04192, -.03717, 0]
        ys   = [1, .9766,  .96880, .96090, .95310, .85160, .73440, 0.67210, 0.50000, 0.45310, 0.28130, 0.17190, 0.10160, 0.07030, .006250, .0547, 0]

        plt.plot(np.linspace(-h/2, 1+h/2, len(U[0])), U[int(N/2), :], label='Projection')
        plt.plot(ys, u100, label='Ghia et al', marker='o', linewidth=0)
        plt.legend()
        plt.xlabel('Y Coordinate')
        plt.ylabel('U Velocity')
        plt.title('U Velocity Along Vertical Centerline\nRe={}, N={}'.format(Re, N))
        plt.grid()
        #plt.savefig('Re100u.pdf')
        plt.close()
        plt.show()
        v100 = [0, -.05906, -.07391, -.08864, -.10313, -.16914, -.22445, -.24533, .05454, .17527, .17507, .16077, .12317, .10890, .10091, .09233, 0]
        ys   = [1,  .96880, .96090, .95310, .9453, .9063, .8594, .8047, 0.50000, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0]
        
        plt.plot(np.linspace(-h/2, 1+h/2, len(V)), V[:, int(N/2)+1], label='Projection')
        plt.plot(ys, v100, label='Ghia et al', marker='o', linewidth=0)
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('V Velocity')
        plt.title('V Velocity Along Horizontal Centerline\nRe={}, N={}'.format(Re, N))
        plt.grid()
        #plt.savefig('Re100v.pdf')
        plt.close()
        plt.show()
    elif Re == 400:
        u400 = [1, .75837, .68439, .1756, .55892, .29093, .16256, .02135, -.11477, -.17119, -.32726, -.24229, -.14612, -.10338, -.09266, -.08186, 0]
        ys   = [1, .9766,  .96880, .96090, .95310, .85160, .73440, 0.67210, 0.50000, 0.45310, 0.28130, 0.17190, 0.10160, 0.07030, .006250, .0547, 0]

        plt.plot(np.linspace(-h/2, 1+h/2, len(U[0])), U[int(N/2), :], label='Projection')
        plt.plot(ys, u400, label='Ghia et al', marker='o', linewidth=0)
        plt.legend()
        plt.xlabel('Y Coordinate')
        plt.ylabel('U Velocity')
        plt.title('U Velocity Along Vertical Centerline\nRe={}, N={}'.format(Re, N))
        plt.grid()
        #plt.savefig('Re400u.pdf')
        plt.close()
        plt.show()
        v400 = [0, -.12146, -.15663, -.19254, -.22847, -.23827, -.44993, -.38598, .05186, .30174, .30203, .28124, .22965, .20920, .19713, .18360,  0]
        ys   = [1,  .96880, .96090, .95310, .9453, .9063, .8594, .8047, 0.50000, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0]
        
        plt.plot(np.linspace(-h/2, 1+h/2, len(V)), V[:, int(N/2)+1], label='Projection')
        plt.plot(ys, v400, label='Ghia et al', marker='o', linewidth=0)
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('V Velocity')
        plt.title('V Velocity Along Horizontal Centerline\nRe={}, N={}'.format(Re, N))
        plt.grid()
        #plt.savefig('Re400v.pdf')
        plt.close()
        plt.show()
    elif Re == 1000:
        u1000 = [1, .65928, .57492, .51117, .46604, .33304, .18719, .05702, -.06080, -.10648, -.27805, -.38289, -.29730, -.22220, -.20196, -.18109, 0]
        ys   = [1, .9766,  .96880, .96090, .95310, .85160, .73440, 0.67210, 0.50000, 0.45310, 0.28130, 0.17190, 0.10160, 0.07030, .006250, .0547, 0]

        plt.plot(np.linspace(-h/2, 1+h/2, len(U[0])), U[int(N/2), :], label='Projection')
        plt.plot(ys, u1000, label='Ghia et al', marker='o', linewidth=0)
        plt.legend()
        plt.xlabel('Y Coordinate')
        plt.ylabel('U Velocity')
        plt.title('U Velocity Along Vertical Centerline\nRe={}, N={}'.format(Re, N))
        plt.grid()
        #plt.savefig('Re1000u.pdf')
        plt.close()
        plt.show()
        v1000 = [0, -.21388, -.27669, -.33714, -.39188, -.51550, -.42665, -.31966, .02526, .32235, .33075, .37065, .32627, .30353, .29012, .27485,  0]
        ys   = [1,  .96880, .96090, .95310, .9453, .9063, .8594, .8047, 0.50000, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0]
        
        plt.plot(np.linspace(-h/2, 1+h/2, len(V)), V[:, int(N/2)+1], label='Projection')
        plt.plot(ys, v1000, label='Ghia et al', marker='o', linewidth=0)
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('V Velocity')
        plt.title('V Velocity Along Horizontal Centerline\nRe={}, N={}'.format(Re, N))
        plt.grid()
        #plt.savefig('Re1000v.pdf')
        plt.close()
        plt.show()

def plot_psi(U, V, h, N):
    psi = np.zeros((N+1, N+1))
    X = np.zeros((N+1, N+1))
    Y = np.zeros((N+1, N+1))
    #print(psi)
    for i in range(1, len(psi)):
        for j in range(1, len(psi[i])):
            psi[i][j] = psi[i-1][j]-V[i][j+1]*h

    for i in range(1, len(psi)):
        X[i,:] = X[i-1][0] + h
    for i in range(1, len(psi[0])):
        Y[:,i] = Y[0][i-1] + h
    #print('x', X)
    #print('y', Y)
    TITLE = 'Psi Values\nRe={}, N={}'.format(Re, N)
    FNAME = 'psi Re{}N{}.pdf'.format(Re,N)
    FONTSIZE = 11
    FIGSIZE = (8.5, 5.5)
    LEVELS = [-0.1175,-0.115,-0.11,-0.1,-0.09,-0.07,-0.05,-0.03,-0.01,-0.0001,-1e-5,-1e-7,-1e-10,1e-8,1e-7,1e-6,1e-5,5e-5,1e-4,2.5e-4,1e-3,1.5e-3,3e-3]

    f = plt.figure(figsize = FIGSIZE)
    cont = plt.contour(X, Y, psi, levels=LEVELS, cmap='coolwarm')
    #plt.clabel(cont, inline=2, colors='k', manual=False)
    plt.xlabel('X', fontsize=FONTSIZE)
    plt.ylabel('Y', fontsize=FONTSIZE)
    plt.figure(f.number)
    plt.grid()
    plt.tick_params(axis='both', labelsize=12)
    cbar = plt.colorbar(orientation='vertical').set_label('Streamfunction Values (\u03A8)')
    #f.tight_layout()
    plt.title(TITLE)
    #plt.savefig(FNAME)
    #plt.show(block=True)
    plt.close()
def main(Re, N):
    rho = 1
    L = 1
    uwall = 1
    Nt = 1

    nu = uwall*L/Re
    h = L/N
    a = h**2/(4*nu)
    b = 4*nu/uwall**2
    beta = .8
    dt = beta*min(a, b)
    #print('nu', 'h', 'dt')
    #print(nu, h, dt)

    # Origin is the bottom right of the mesh
    P = np.zeros((N, N)) # Pressure, one value per cell
    U = np.zeros((N+3, N+2)) # Horizontal velocity, one value per vertical edge and one layer of ghost values all around
    V = np.zeros((N+2, N+3)) # vertical velocity, one value per horizontal edge and one layer of ghost values all around

    F = np.zeros((N,N)) # horizontal x-momentum flux, one value per cell
    G = np.zeros((N,N)) # vertical y-momentum flux, one value per cell
    Hx = np.zeros((N+1,N+1)) # vertical x-momentum flux, one value per grid node
    Hy = np.zeros((N+1,N+1)) # horizontal y-momentum flux, one value per grid node
    f = np.zeros((N,N))
    A = Poissona(N, h)  
        
    res = np.array([])
    resi = 1
    i = 0
    while resi > 1e-5:
    #while i < 5:
        ghost_vel(U, V, uwall)
        #print('g u\n', np.around(U, 4))
        #print('g v\n', np.around(V, 4))
        flux(U, V, F, G, Hx, Hy, h, nu, N)
        #print('hx\n', Hx)
        #print('hy\n', Hy)
        #print(F)
        #print(G)
        update1(U, V, F, G, Hx, Hy, h, dt, N)
        #print('1 u\n', np.around(U, 4))
        #print('1 v\n', np.around(V, 4))
        P = solve_poisson(A, U, V, h, dt, N)
        #gauss_seidel(U, V, P, h, dt, N)
        #print('P')
        #print(P)
        update2(U, V, P, h, dt, N)
        for ii in range(N):
            for jj in range(N):
                b = U[ii+2][jj+1]-U[ii+1][jj+1]
                c = V[ii+1][jj+2]-V[ii+1][jj+1]
                f[ii][jj] = b+c
        #print(f)
        #print('2 u\n', np.around(U, 4))
        #print('2 v\n', np.around(V, 4))
        resi, resj = residual(F, G, Hx, Hy, P, h, N)
        res = np.append(res, resi+resj)
        res2 = resi+resj
        print(i, res2)
        i += 1
        
    
    plt.semilogy(np.linspace(0, i, i), res, label='Residual')
    plt.ylabel('Residual')
    plt.xlabel('Iteration')
    plt.title('Residuals vs. Iteration\nRe={}, N={}'.format(Re, N))
    plt.legend()
    plt.grid()
    #plt.savefig('Debug Residual Final Re{}N{}.pdf'.format(Re, N))
    plt.close()
    #plt.show()
    plot_vel(U, V, h)
    plot_psi(U, V, h, N)

if __name__ == '__main__':
    Re = 1000
    N = 64
    main(Re, N)

    '''Re = 400
    N = 32
    main(Re, N)'''

    '''Re = 1000
    N = 64
    main(Re, N)'''
