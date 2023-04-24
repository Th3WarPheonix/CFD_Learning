
import numpy as np
from numpy import linalg
from scipy.linalg import hessenberg

n = 4
target_eigenvalues = np.linspace(1, n, n)
D = np.diag(target_eigenvalues)
print('Target Eigenvalues = {}'.format(target_eigenvalues))
S = np.random.rand(n,n)
S = 2*(S-0.5)
A = S@D@linalg.inv(S)
H = hessenberg(A)

print('Straight QR')

for j in range(n): # Iterate on the deflated matrix to get all eigenvalues
    for i in range(5): # Iterate on a constant size matrix by changing the shift variable to converge to an eigenvalue
        Q, R = linalg.qr(H - H[-1,-1]*np.eye(n-j))
        H = R@Q + H[-1,-1]*np.eye(n-j)
    print('Eigenvalue = {}'.format(H[-1,-1]))
    H = H[:-1, :-1]

print('Simultaneous Inverse Iteration QR')

H2 = hessenberg(A)

for j in range(n): # Iterate on the deflated matrix to get all eigenvalues
    Qn = np.eye(n-j)
    mu = H2[-1,-1]
    for i in range(5): # Iterate on a constant size matrix by changing the shift variable to converge to an eigenvalue
        Qn1, Rn1 = linalg.qr((H2 - mu*np.eye(n-j))@Qn)
        Hn1 = Rn1@np.transpose(np.conjugate(Qn))@Qn1 + mu*np.eye(n-j)
        mu = Hn1[-1,-1]
        Qn = Qn1
    print('Eigenvalue = {}'.format(Hn1[-1,-1]))
    H2 = Hn1[:-1, :-1]    