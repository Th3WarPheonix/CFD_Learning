#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <algorithm>

int printmat(int lenrow, int lencol, double mat[lenrow][lencol]){
    int i, j;
    for (i = 0; i<lenrow; i++){
        for (j = 0; j<lencol; j++){
            printf("%f ", mat[i][j]);
        }
        printf("%d", i);
        printf("\n");
    }
    printf("\n");
}

int zero_out(int lrow, int lcol, double mat[lrow][lcol]){
    int i, j;
    for (i = 0; i<lrow; i++){
        for (j = 0; j<lcol; j++){
            mat[i][j] = 0;
        }
    }
}

int gs_black(int N, double f[N][N], double P[N][N], float h, float dt, float w){
    int i, j;
    float a;
    // BLACK
    for (i = 0; i<N; i++){
        for (j = 0; j<N; j++){
            if ((i+j) % 2 == 0){ // black
                //print('b', i, j)
                // CORNER
                if (i == 0 && j == 0){ // Bottom left
                    a = P[i][j+1] + P[i+1][j];
                    P[i][j] = w*.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (i == 0 && j == N-1){ // top left
                    a = P[i][j-1] + P[i+1][j];
                    P[i][j] = w*.5*(-f[i][j]*h/dt+a)+(1-w)*P[i][j];
                }

                else if (i == N-1 && j == 0){ // top right
                    a = P[i][j+1] + P[i-1][j];
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (i == N-1 && j == N-1){ // bottom right
                    a = P[i][j-1] + P[i-1][j];
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }
                // BORDER
                else if (i == 0){ // LEFT
                    a = P[i][j+1] + P[i][j-1] + P[i+1][j];
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (i == N - 1){ // RIGHT
                    a = P[i][j+1] + P[i][j-1] + P[i-1][j];
                    P[i][j] = w*1/3*(-f[i][j]*h/dt+a)+(1-w)*P[i][j];
                }

                else if (j == 0){ // BOTTOM
                    a = P[i][j+1] + P[i+1][j] + P[i-1][j];
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (j == N - 1){ // TOP
                    a = P[i][j-1] + P[i+1][j] + P[i-1][j];
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else{
                    a = P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j];
                    P[i][j] = w*1/4*(-f[i][j]*h/dt+a)+(1-w)*P[i][j];
                }
            }
        }
    }
}

int gs_red(int N, double f[N][N], double P[N][N], float h, float dt, float w){
    int i, j;
    float a;
    // RED
    for (i = 0; i<N; i++){
        for (j = 0; j<N; j++){
            if (i+j % 2 == 1){ // red
                //print('r', i, j)
                // CORNER
                if (i == 0 && j == 0){ // Bottom left
                    a = P[i][j+1] + P[i+1][j];
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (i == 0 && j == N-1){ // top left
                    a = P[i][j-1] + P[i+1][j];
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (i == N-1 && j == 0){ // top right
                    a = P[i][j+1] + P[i-1][j];
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (i == N-1 && j == N-1){ // bottom right
                    a = P[i][j-1] + P[i-1][j];
                    P[i][j] = w*-.5*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }
                // BORDER
                else if (i == 0){ // LEFT
                    a = P[i][j+1] + P[i][j-1] + P[i+1][j];
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (i == N - 1){ // RIGHT
                    a = P[i][j+1] + P[i][j-1] + P[i-1][j];
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (j == 0){ // BOTTOM
                    a = P[i][j+1] + P[i+1][j] + P[i-1][j];
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else if (j == N - 1){ // TOP
                    a = P[i][j-1] + P[i+1][j] + P[i-1][j];
                    P[i][j] = w*-1/3*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }

                else{
                    a = P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j];
                    P[i][j] = w*-1/4*(f[i][j]*h/dt-a)+(1-w)*P[i][j];
                }
            }
        }
    }
}

int gauss_seidel(int N, double U[N+3][N+2], double V[N+2][N+3], double P[N][N], float h, float dt){
    int i, j, k;
    
    float w = 1.4;
    double f[N][N];
    double P0 = 1000.0;
    double P01 = 0.0;    
    
    for (i = 0; i<N; i++){
        for (j = 0; j<N; j++){
            f[i][j] = U[i+2][j+1]-U[i+1][j+1]+V[i+1][j+2]-V[i+1][j+1];
        }
    }
    k = 0;
    while (fabs(P0-P01) > .00001){
        P01 = P0;
        gs_black(N, f, P, h, dt, w);
        gs_red(N, f, P, h, dt, w);

        P0 = P[0][0];
        if (k % 1000 == 0 && k != 0){
            printf("\tk %d\n", k);
        }
        k++;
    }
    printf("\tk %d\n", k);
}

int update1(int N, double U[N+3][N+2], double V[N+2][N+3], double F[N][N], double G[N][N], double Hx[N+1][N+1], double Hy[N+1][N+1], float h, float dt){
    // Update U
    int i, j;
    double F1, Hx1, G1, Hy1;

    for (i = 2; i < N+1; i++){
        for (j = 1; j < N+1; j++){
            F1 = F[i-1][j-1] - F[i-2][j-1]; // Fr - Fl
            Hx1 = Hx[i-1][j] - Hx[i-1][j-1]; // Hxu - Hxd
            U[i][j] = U[i][j] - dt/h*(F1+Hx1);
        }
    }
    // Update V
    for (i = 1; i < N+1; i++){
        for (j = 2; j < N+1; j++){
            G1 = G[i-1][j-1] - G[i-1][j-2]; // Gt - Gb
            Hy1 = Hy[i][j-1] - Hy[i-1][j-1]; // Hyr - Hyl
            V[i][j] = V[i][j] - dt/h*(G1+Hy1);
        }
    }
}

int update2(int N, double U[N+3][N+2], double V[N+2][N+3], double P[N][N], float h, float dt){
    int i, j;

    for (i = 2; i<N+1; i++){
        for (j = 1; j<N+1; j++){
            U[i][j] = U[i][j] - dt/h*(P[i-1][j-1]-P[i-2][j-1]); // u - dt*h*(Pr-Pl);
        }
    }

    for (i = 1; i<N+1; i++){
        for (j = 2; j<N+1; j++){
            V[i][j] = V[i][j] - dt/h*(P[i-1][j-1]-P[i-1][j-2]); // u - dt*h*(Pt-Pb);
        }
    }
}


double smart(double q, double ull, double ul, double ur, double urr){
    double phim1, phi0, phi1, phi12, phihat, phi12hat;

    if (q > 0){
        phim1 = ull;
        phi0 = ul;
        phi1 = ur;
    }
    else if (q <= 0){
        phim1 = urr;
        phi0 = ur;
        phi1 = ul;
    }

    if ((phi1-phim1) == 0 && (phi0-phim1) == 0){
        phi12 = phi0;
    }
    else if ((phi1-phim1) == 0){
        phi12 = phi0;
    }
    else{
        phihat = (phi0-phim1)/(phi1-phim1);
        if (phihat <= 0 || phihat >= 1){
            phi12hat = phihat;
        }
        else if (0 < phihat <= 1/6){
            phi12hat = 3*phihat;
        }
        else if (1/6 < phihat <= 5/6){
            phi12hat = 3/8*(2*phihat+1);
        }
        else if (5/6 < phihat <= 1){
            phi12hat = 1.0;
        }

        phi12 = phi12hat*(phi1-phim1)+phim1;
    }
    return phi12;
}

int flux(int N, double U[N+3][N+2], double V[N+2][N+3], double F[N][N], double G[N][N], double Hx[N+1][N+1], double Hy[N+1][N+1], float h, double nu){
    int i, j;
    double urr, ur, ul, ull, vtt, vt, vb, vbb;
    double utt, ut, ub, ubb, vrr, vr, vl;
    double qF, qG, phiF, phiG, vll;
    double qHx, qHy, phiHy, phiHx;

    for (i = 0; i<N; i++){
        for (j = 0; j<N; j++){
            urr = U[i+3][j+1];
            ur  = U[i+2][j+1];
            ul  = U[i+1][j+1];
            ull = U[i+0][j+1];
 
            vtt = V[i+1][j+3];
            vt  = V[i+1][j+2];
            vb  = V[i+1][j+1];
            vbb = V[i+1][j+0];

            qF = (ur+ul)/2;
            qG = (vt+vb)/2;
            
            phiF = smart(qF, ull, ul, ur, urr);
            phiG = smart(qG, vbb, vb, vt, vtt);

            F[i][j] = qF*phiF - nu/h*(ur-ul);
            G[i][j] = qG*phiG - nu/h*(vt-vb);
        }
    }
    
    for (i = 0; i<N+1; i++){
        for (j = 0; j<N+1; j++){
            ut = U[i+1][j+1];
            ub = U[i+1][j+0];

            vr = V[i+1][j+1];
            vl = V[i+0][j+1];
            vll;
            
            qHx = (vr+vl)/2;
            qHy = (ut+ub)/2;

            if (i == 0){ // Left wall
                qHy = 0;
                phiHy = 0;
            }
            else if (i == N){ // Right wall
                qHy = 0;
                phiHy = 0;
            }
            else{
                vrr = V[i+2][j+1];
                vll = V[i-1][j+1];
                phiHy = smart(qHy, vll, vl, vr, vrr);
            }
            
            if (j == 0){ // Bottom wall
                qHx = 0;
                phiHx = 0;
            }
            else if (j == N){ // Top wall
                qHx == 0;
                phiHx = 0;
            }
            else{
                utt = U[i+1][j+2];
                ubb = U[i+1][j-1];
                phiHx = smart(qHx, ubb, ub, ut, utt);
            }

            Hx[i][j] = qHx*phiHx - nu/h*(ut-ub);
            Hy[i][j] = qHy*phiHy - nu/h*(vr-vl);
        }
    }
}

int ghost_vel(int N, double U[N+3][N+2], double V[N+2][N+3], float uwall){
    int i = 0;
    for (i = 0; i < N+3; i++){ // Horizontal wall
        U[i][N+1] = 2*uwall-U[i][N]; // top wall
        U[i][0]  = -U[i][1]; // bottom wall
    }

    for (i = 0; i < N+2; i++){
        V[i][N+2] = V[i][N]; // top wall
        V[i][0]  = V[i][2]; // bottom wall
    }

    for (i = 0; i < N+2; i++){ // Vertical wall
        U[0][i]  = U[2][i]; // left wall
        U[N+2][i] = U[N][i]; // right wall
    }
    
    for (i = 0; i < N+3; i++){
        V[0][i]  = -V[1][i]; // left wall
        V[N+1][i] = -V[N][i]; // right wall
    }
}

double sum_elem(int lrow, int lcol, double mat[lrow][lcol]){
    int i, j;
    double sum = 0.0;
    for (i = 0; i<lrow; i++){
        for (j = 0; j<lcol; j++){
            sum = sum + fabs(mat[i][j]);
        }
    }
    return sum;
}

double residual(int N, double F[N][N], double G[N][N], double Hx[N+1][N+1], double Hy[N+1][N+1],double P[N][N], float h, double Ri2[N-1][N], double Rj2[N][N-1]){
    int i, j;
    double F1, P1i, Hx1, G1, P1j, Hy1;

    //zero_out(N-1, N, Ri2);
    //zero_out(N, N-1, Rj2);

    for (i = 0; i < N-1; i++){
        for (j = 0; j < N; j++){
            F1 = F[i+1][j] - F[i][j];
            P1i = P[i+1][j] - P[i][j];
            Hx1 = Hx[i+1][j+1] - Hx[i+1][j-0];
            Ri2[i][j] = h*(F1+P1i+Hx1);            
        }
    }

    for (i = 0; i < N; i++){
        for (j = 0; j < N-1; j++){
            G1 = G[i][j+1] - G[i][j];
            P1j = P[i][j+1] - P[i][j];
            Hy1 = Hy[i+1][j+1] - Hy[i-0][j+1];
            Rj2[i][j] = h*(G1+P1j+Hy1);
        }
    }

    return sum_elem(N-1, N, Ri2)+sum_elem(N, N-1, Rj2);
}

double mainloop(float rho, float L, float uwall, int Re, int N){

    const double nu = uwall*L/Re;
    const float h = L/N;
    double a = pow(h, 2)/(4*nu);
    double b = 4*nu/pow(uwall, 2);
    const float beta = .8;
    float dt;

    if (a < b){
        dt = beta*a;
    }
    else {
        dt = beta*b;
    }
    // printf('nu', 'h', 'dt')
    // printf(nu, h, dt)

    //  Origin is the bottom right of the mesh
    double P[N][N]; //  Pressure, one value per cell
    double U[N+3][N+2]; //  Horizontal velocity, one value per vertical edge and one layer of ghost values all around
    double V[N+2][N+3]; //  vertical velocity, one value per horizontal edge and one layer of ghost values all around
    double F[N][N]; //  horizontal x-momentum flux, one value per cell
    double G[N][N]; //  vertical y-momentum flux, one value per cell
    double Hx[N+1][N+1]; //  vertical x-momentum flux, one value per grid node
    double Hy[N+1][N+1]; //  horizontal y-momentum flux, one value per grid node
    double Ri2[N-1][N];
    double Rj2[N][N-1];

    zero_out(N, N, P);
    zero_out(N+3, N+2, U); 
    zero_out(N+2, N+3, V);
    zero_out(N, N, F);
    zero_out(N, N, G);
    zero_out(N+1, N+1, Hx);
    zero_out(N+1, N+1, Hy);
    zero_out(N-1, N, Ri2);
    zero_out(N, N-1, Rj2);
    
    double resi = 1.0;
    double resj = 1.0;
    double restot = 5.0;

    int i = 0;
    while (i < 5){
        //printmat(N+3, N+2, U);
        //printmat(N+2, N+3, V);
        ghost_vel(N, U, V, uwall);
        //printmat(N+3, N+2, U);
        //printmat(N+2, N+3, V);
        flux(N, U, V, F, G, Hx, Hy, h, nu);
        //printf("hx\n");
        //printmat(N+1, N+1, Hx);
        //printmat(N+1, N+1, Hy);
        //printf("F\n");
        //printmat(N, N, F);
        //printf("G\n");
        //printmat(N, N, G);
        update1(N, U, V, F, G, Hx, Hy, h, dt);
        //printf("U\n");
        //printmat(N+3, N+2, U);
        //printmat(N+2, N+3, V);
        gauss_seidel(N, U, V, P, h, dt);
        printf("P\n");
        printmat(N, N, P);
        update2(N, U, V, P, h, dt);
        // printf('2 u\n', around(U, 4))
        // printf('2 v\n', around(V, 4))
        restot = residual(N, F, G, Hx, Hy, P, h, Ri2, Rj2);
        printf("%d %.3e\n", i, restot);
        i++;
    
    }
    return 0.0;
}

int main(){
    
    float rho = 1.0;
    float L = 1.0;
    float uwall = 1.0;

    int Re = 100;
    int N = 4;
    mainloop(rho, L, uwall, Re, N);

    //Re = 400;
    //N = 32;
    //mainloop(rho, L, uwall, Re, N);

    //Re = 1000;
    //N = 64;
    //mainloop(rho, L, uwall, Re, N);
    
    /*
    *plt.semilogy(linspace(0, i, i), res, label='Residual')
    *plt.ylabel('Residual')
    *plt.xlabel('Iteration')
    *plt.title('Residuals vs. Iteration\nRe={}, N={}'.format(Re, N))
    *plt.legend()
    *plt.grid()
    *plt.savefig('Debug Residual Final Re{}N{}.pdf'.format(Re, N))
    *plt.close()
    *plt.show()
    *plot_vel(U, V, h)
    *plot_psi(U, V, h, N)
    */
    return 0;
}
