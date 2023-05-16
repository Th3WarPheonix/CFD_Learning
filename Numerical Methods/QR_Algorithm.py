
import numpy as np
from numpy import linalg
from scipy.linalg import hessenberg


def qr_algo(n, A):
    """Find eigenvalues using QR algorithm by using the most direct approach, no shifts, no inverse iteration"""
    eigval = np.empty(n)

    hess = hessenberg(A)
    for j in range(n): # Iterate on the deflated matrix to get all eigenvalues
        for i in range(10): # Iterate on a constant size matrix by changing the shift variable to converge to an eigenvalue
            Q, R = linalg.qr(hess - hess[-1,-1]*np.eye(n-j))
            hess = R@Q + hess[-1,-1]*np.eye(n-j)
        eigval[j] = hess[-1,-1]
        hess = hess[:-1, :-1]

    return eigval

def qr_sii(n, A):
    """Find eigenvalues using QR algorithm by using simultaneous inverse iterations"""
    eigval = np.empty(n)

    hess = hessenberg(A)
    for j in range(n): # Iterate on the deflated matrix to get all eigenvalues
        Qn = np.eye(n-j)
        mu = hess[-1,-1]
        for i in range(10): # Iterate on a constant size matrix by changing the shift variable to converge to an eigenvalue
            Qn1, Rn1 = linalg.qr((hess - mu*np.eye(n-j))@Qn) # Using the simultaneous inverse interation with shifts
            Hn1 = Rn1@np.transpose(np.conjugate(Qn))@Qn1 + mu*np.eye(n-j) # Using the simultaneous inverse interation with shifts
            mu = Hn1[-1,-1]
            Qn = Qn1
        eigval[j] = hess[-1,-1]
        hess = Hn1[:-1, :-1]

    return eigval

def main():
    n = 4
    target_eigenvalues = np.linspace(1, n, n)
    D = np.diag(target_eigenvalues)
    print('Target Eigenvalues = {}'.format(target_eigenvalues))
    S = np.random.rand(n,n)
    S = 2*(S-0.5)
    A = S@D@linalg.inv(S)

if __name__ == '__main__':
    main()  