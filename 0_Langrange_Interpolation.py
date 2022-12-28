
import numpy as np
import matplotlib.pyplot as plt


def lagrange_basis_function(nodes, point): # Computes Lagrange basis function values
    """
    This function only needs to be used if the value at the node is specified, if the gradient is specified use lagrange_basis_interpolation
    nodes : at which the value is known (the value of the node is not needed for this function just the x coordinate)
    point : the x location of the point at which the value needs to be interpolated"""
    lagrange_values = np.zeros(len(nodes)) # Storage for output of lagrange polynomial at all nodes at the current interpolation point
    for j, nodej in enumerate(nodes): # Loop through all of the nodes
        node_lagrange_value = 1 # lagrange function value resets for each node
        for i, nodei in enumerate(nodes):
            if i != j: # If the current j node is not the current i node
                node_lagrange_value *= (point-nodei)/(nodej-nodei)
        lagrange_values[j] = node_lagrange_value

    return lagrange_values

def lagrange_basis_gradients(xn, x): # Computes Lagrange basis gradients
    N = len(xn)
    gphi = np.zeros(N)
    for j in range(N):
        gphi[j] = 0
        for k in range(N):
            if k!=j:
                gj = 1/(xn[j]-xn[k])
            for i in range(N):
                if i == j or i == k:
                    continue
                gj *= (x-xn[i])/(xn[j]-xn[i])
                gphi[ j ] += gj
    return gphi

def lagrange_polynomial(num_points, node_value, node_locs, poi):
    """
    nunm_points : Number of points at which to interpolate from the nodes given\n
    node_values : Values of the nodes\n
    node_locs : X-axis value at of the node\n
    poi : explicit points at which the interpolated value would like to be known

    Returns The final Lagrange polynomial, an array of the points of the constituent Lagrange polynomials (Phi), and the interpolation points\n
    Tips: Phi[:,i] to plot the polynomials that constitute the final Lagrange polynomial against the interpolation points
    """
    
    node_locs = np.sort(node_locs)
    num_nodes = len(node_locs)
    node_locations = node_locs  # node locations
    interp_points = np.linspace(0,node_locs[-1]-node_locs[0],num_points) # interpolation points

    Phi = np.zeros((num_points,num_nodes)) # First column is the first lagrange polynomial values, second column is second lagrange polynomial values, etc
    for n, pointi in enumerate(interp_points): # Loop through all of the points at which you want the polynomial to be interpolated
        Phi[n,:] = lagrange_basis_function(node_locations, pointi) # basis values
    
    lagrange_poly = np.empty(num_points) # end result of the lagrange polynomial
    for i in range(num_nodes):
        lagrange_poly += Phi[:,i]*node_value[i]

    # Repeating the exact same process as above but only for the points of interest
    Phi_poi = np.zeros((len(poi),num_nodes))
    for n, pointi in enumerate(poi):
        Phi_poi[n,:] = lagrange_basis_function(node_locations, pointi) # basis values
    poi_values = np.empty(len(poi))
    for i in range(num_nodes):
        poi_values += Phi_poi[:,i]*node_value[i]
    print()
    return lagrange_poly, poi_values, Phi, interp_points
    

def plot_lapoly(lagrange_poly, Phi, interp_points):
    plt.figure(figsize=(8,5))
    colors = ['black', 'red', 'blue', 'green', 'magenta', 'cyan']
    for i in range(Phi.shape[1]):
        plt.plot(interp_points, Phi[:,i], linewidth=2, color=colors[i%len(colors)], label='P{}'.format(i+1))

    plt.plot(interp_points, lagrange_poly, label='Lagrange Polynomial', linewidth=2, color='chartreuse')
    plt.scatter(node_locs, node_value, marker='o', color='chartreuse')
    plt.xlabel('Domain', fontsize=16)
    plt.ylabel('Lagrange Interpolation Results', fontsize=16)
    plt.legend()
    plt.title('Lagrange Interpolation')
    plt.show()

if __name__ == '__main__':
    num_points = 200
    node_locs = [0, .75, 1, 5]
    node_value = [1, 3, 7, -5]
    points_interest = [6, 3]
    result = lagrange_polynomial(num_points, node_value, node_locs, points_interest)
    plot_lapoly(result[0], result[2], result[3])
