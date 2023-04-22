# Page 64
import numpy as np
import matplotlib.pyplot as plt

""" CODE NOTE:
un1 is the value of the node at the next time step
un  is the value of the node at the current time step
CFL = cdt/dx"""

def explicit_backward(value_array, CFL_num, time_steps):
    for t in range(time_steps):
        for i, un in enumerate(value_array):
            un1 = un - CFL_num

def explicit_forward(CFL_num):
    pass

def explicit_central(CFL_num):
    pass

def implicit_central(CFL_num):
    pass

def crank_nicolson(CFL_num):
    pass

def lax_method(CFL_num):
    pass

def lax_wendroff(CFL_num):
    pass

def maccormack_method(CFL_num):
    pass

def jameson_method(CFL_num):
    pass

def warming_beam(CFL_num):
    pass

def upwind_method(CFL_num):
    pass

def initial_condition(value1, value2, x_sep, x_start, x_end, num_nodes):
    """Can only produce a two level step"""
    domain = np.linspace(x_start, x_end, num_nodes)
    values1 = np.linspace(value1, value1, int(np.ceil(x_sep/(x_end-x_start)*num_nodes)))
    values2 = np.linspace(value2, value2, int(np.ceil((1-x_sep/(x_end-x_start))*num_nodes))-1)
    return domain, np.concatenate([values1, values2])


def main():
   domain, values = initial_condition(1, .5, .5, 0, 2, 41)

   plt.plot(domain, values, label='Initial Condition', linestyle='--')
   plt.xlabel('x')
   plt.ylabel('u')
   plt.legend()
   plt.show()
   

if __name__ == "__main__":
    main()
