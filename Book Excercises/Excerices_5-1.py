# Page 64
import numpy as np
import matplotlib.pyplot as plt

""" CODE NOTE:
un1 is the value of the node at the next time step
un  is the value of the node at the current time step
CFL = cdt/dx"""

def explicit_backward(domain_array, value_array, advection_coeff, deltat, deltax, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            un1s[i] = uns[i] - advection_coeff*deltat*(CURRENT_VALUE-LAST_VALUE)/deltax

        uns = np.copy(un1s)
    return un1s

def explicit_forward(domain_array, value_array, advection_coeff, deltat, deltax, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - advection_coeff*deltat*(NEXT_VALUE-CURRENT_VALUE)/deltax

        uns = np.copy(un1s)

    return un1s

def explicit_central(domain_array, value_array, advection_coeff, deltat, deltax, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - advection_coeff*deltat*(NEXT_VALUE-LAST_VALUE)/deltax

        uns = np.copy(un1s)

    return un1s

def implicit_central():
    pass

def crank_nicolson():
    pass

def lax_method(domain_array, value_array, advection_coeff, deltat, deltax, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = (LAST_VALUE + NEXT_VALUE)/2 - advection_coeff*deltat*(NEXT_VALUE-LAST_VALUE)/deltax

        uns = np.copy(un1s)

    return un1s

def lax_wendroff(domain_array, value_array, advection_coeff, deltat, deltax, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - 0.5*advection_coeff*deltat*(NEXT_VALUE-LAST_VALUE)/deltax + 0.5*advection_coeff**2*deltat**2*(NEXT_VALUE-2*CURRENT_VALUE+LAST_VALUE)/deltax**2

        uns = np.copy(un1s)

    return un1s

def maccormack_method(domain_array, value_array, advection_coeff, deltat, deltax, time_steps):
    uns    = np.copy(value_array)
    un1s_2 = np.copy(value_array)
    un1s   = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s_2[i] = CURRENT_VALUE - advection_coeff*deltat*(NEXT_VALUE-CURRENT_VALUE)/deltax
            un1s[i] = 0.5*(CURRENT_VALUE + un1s_2[i] - advection_coeff*deltat*(un1s_2[i]-un1s_2[i-1])/deltax)

        uns = np.copy(un1s)

    return un1s

def jameson_method(domain_array, value_array, advection_coeff, deltat, deltax, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)
    alpha = [1/4, 1/3, 1/2, 1]

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            
            unk0 = uns[i]
            unk1 = CURRENT_VALUE - 1/2*advection_coeff*deltat*unk0/deltax
            unk2 = CURRENT_VALUE - 1/2*advection_coeff*deltat*unk1/deltax
            unk3 = CURRENT_VALUE - advection_coeff*deltat*unk2/deltax
            unk4 = CURRENT_VALUE - 1/6*advection_coeff*deltat*(unk0 + 2*unk1 + 2*unk2 + unk3)/deltax
        
        un1s[i] = unk4
        uns = np.copy(un1s)

    return un1s

def warming_beam():
    pass

def upwind_method():
    pass

def initial_condition(value1, value2, x_sep, x_start, x_end, num_nodes):
    """Can only produce a two level step"""
    domain = np.linspace(x_start, x_end, num_nodes)
    values1 = np.linspace(value1, value1, int(np.ceil(x_sep/(x_end-x_start)*num_nodes)))
    values2 = np.linspace(value2, value2, int(np.ceil((1-x_sep/(x_end-x_start))*num_nodes))-1)
    return domain, np.concatenate([values1, values2])


def main():
    CFL_num = 0.9
    advection_coeff = 1
    num_points = 40
    x_start = 0
    x_end = 2
    deltax = (x_end-x_start)/num_points
    deltat = CFL_num*deltax*advection_coeff # The expression (domain[0] - domain[1])/advection_coeff should really be a minimum over the spacing of all points
    time_steps = 10
    domain, values_ic = initial_condition(1, .5, .5, 0, 2, num_points+1) # set up inital conditions

    # Running the finite difference methods
    values_eb = explicit_backward(domain, values_ic, advection_coeff, deltat, deltax, time_steps)
    values_ef = explicit_forward(domain, values_ic, advection_coeff, deltat, deltax, time_steps)
    values_ec = explicit_central(domain, values_ic, advection_coeff, deltat, deltax, time_steps)
    values_lx = lax_method(domain, values_ic, advection_coeff, deltat, deltax, time_steps)
    values_lw = lax_wendroff(domain, values_ic, advection_coeff, deltat, deltax, time_steps)
    values_mc = maccormack_method(domain, values_ic, advection_coeff, deltat, deltax, time_steps)
    values_jm = jameson_method(domain, values_ic, advection_coeff, deltat, deltax, time_steps)

    # Plotting the finite difference methods
    fig, axis = plt.subplots(5, 2, figsize=(16, 13.25))
    plt.subplots_adjust(hspace=.302, bottom=.05, top=.955)
    fig.suptitle('Excerise 5.1, 5.2', fontsize=16)
    axis[0,0].plot(domain, values_eb, label='Explicit Backward', marker='.')
    axis[0,1].plot(domain, values_ef, label='Explicit Forward', marker='.')
    axis[1,0].plot(domain, values_ec, label='Explicit Central', marker='.')
    axis[1,1].plot(domain, values_lx, label='Lax', marker='.')
    axis[2,0].plot(domain, values_lw, label='Lax Wendroff', marker='.')
    axis[2,1].plot(domain, values_mc, label='MacCormack', marker='.')
    axis[3,0].plot(domain, values_jm, label='Jameson', marker='.')

    for plot_row in axis:
        for plot in plot_row:
            plot.plot(domain, values_ic, label='Initial Condition', linestyle='--', zorder=0)

            plot.set_xlabel('x')
            plot.set_ylabel('u')
            plot.legend()

    plt.show()
    

if __name__ == "__main__":
    main()
