# Page 64
import numpy as np
import matplotlib.pyplot as plt

""" CODE NOTE:
un1 is the value of the node at the next time step
un  is the value of the node at the current time step
CFL = cdt/dx"""

def explicit_backward(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            un1s[i] = uns[i] - CFL_num*(CURRENT_VALUE-LAST_VALUE)

        uns = np.copy(un1s)
    return un1s

def explicit_forward(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - CFL_num*(NEXT_VALUE-CURRENT_VALUE)

        uns = np.copy(un1s)

    return un1s

def explicit_central(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE)

        uns = np.copy(un1s)

    return un1s

def implicit_central():
    pass

def crank_nicolson():
    pass

def lax_method(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = (LAST_VALUE + NEXT_VALUE)/2 - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE)

        uns = np.copy(un1s)

    return un1s

def lax_wendroff(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE) + 0.5*CFL_num**2*(NEXT_VALUE-2*CURRENT_VALUE+LAST_VALUE)

        uns = np.copy(un1s)

    return un1s

def maccormack_method(domain_array, value_array, CFL_num, time_steps):
    uns    = np.copy(value_array)
    un1s_2 = np.copy(value_array)
    un1s   = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s_2[i] = CURRENT_VALUE - CFL_num*(NEXT_VALUE-CURRENT_VALUE)
            un1s[i] = 0.5*(CURRENT_VALUE + un1s_2[i] - CFL_num*(un1s_2[i]-un1s_2[i-1]))

        uns = np.copy(un1s)

    return un1s

def jameson_method(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)
    alpha = [1/4, 1/3, 1/2, 1]

    for t in range(1):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            
            unk0 = uns[i]
            unk1 = CURRENT_VALUE - 1/4*CFL_num*unk0
            unk2 = CURRENT_VALUE - 1/3*CFL_num*(unk1-unk0)
            unk3 = CURRENT_VALUE - 1/2*CFL_num*(unk2-unk1)
            unk4 = CURRENT_VALUE - 1/1*CFL_num*(unk3-unk2)
        
            un1s[i] = unk4
        uns = np.copy(un1s)

    return un1s

def warming_beam(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(2, len(domain_array)-1):
            LAST_VALUE2   = uns[i-2]
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]

            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(3*CURRENT_VALUE-4*LAST_VALUE+LAST_VALUE2) + 0.5*(CFL_num)**2*(LAST_VALUE2-2*LAST_VALUE+CURRENT_VALUE)
        uns = np.copy(un1s)

    return un1s

def upwind_method(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(2, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]

            # The expression abs(CFL_num) should really be abs(advection_coeff)*deltat/deltax
            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE) + 0.5*abs(CFL_num)*(NEXT_VALUE-2*CURRENT_VALUE+LAST_VALUE)
        uns = np.copy(un1s)

    return un1s

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
    domain, exact_sol = initial_condition(1, .5, 1-deltax, 0, 2, num_points+1) # set up inital conditions


    # Running the finite difference methods CFL = 0.9
    values_eb = explicit_backward(domain, values_ic, CFL_num, time_steps)
    values_ef = explicit_forward(domain, values_ic, CFL_num, time_steps)
    values_ec = explicit_central(domain, values_ic, CFL_num, time_steps)
    values_lx = lax_method(domain, values_ic, CFL_num, time_steps)
    values_lw = lax_wendroff(domain, values_ic, CFL_num, time_steps)
    values_mc = maccormack_method(domain, values_ic, CFL_num, time_steps)
    values_jm = jameson_method(domain, values_ic, CFL_num, time_steps)
    values_wb = warming_beam(domain, values_ic, CFL_num, time_steps)
    values_up = upwind_method(domain, values_ic, CFL_num, time_steps)

    CFL_num = 2
    values_jm2 = jameson_method(domain, values_ic, CFL_num, time_steps)
    values_wb2 = warming_beam(domain, values_ic, CFL_num, time_steps)


    # Plotting the finite difference methods
    fig, axis = plt.subplots(6, 2, figsize=(16, 13.25))
    plt.subplots_adjust(hspace=.302, bottom=.05, top=.955)
    fig.suptitle('Excerise 5.1, 5.2', fontsize=16)
    axis[0,0].plot(domain, values_eb, label='Explicit Backward', marker='.')
    axis[0,1].plot(domain, values_ef, label='Explicit Forward', marker='.')
    axis[1,0].plot(domain, values_ec, label='Explicit Central', marker='.')
    axis[1,1].plot(domain, values_lx, label='Lax', marker='.')
    axis[2,0].plot(domain, values_lw, label='Lax Wendroff', marker='.')
    axis[2,1].plot(domain, values_mc, label='MacCormack', marker='.')
    axis[3,0].plot(domain, values_jm, label='Jameson', marker='.')
    axis[4,0].plot(domain, values_wb, label='Warming Beam', marker='.')


    axis[3,1].plot(domain, values_jm2, label='Jameson CFL=2', marker='.')
    axis[4,1].plot(domain, values_wb2, label='Warming Beam CFL=2', marker='.')

    axis[5,0].plot(domain, values_up, label='Upwind', marker='.')



    for plot_row in axis:
        for plot in plot_row:
            plot.plot(domain, values_ic, label='Initial Condition', linestyle='--', zorder=0)
            plot.plot(domain, exact_sol, label='Exact Solution', linestyle='-.', zorder=0)

            plot.set_xlabel('x')
            plot.set_ylabel('u')
            plot.legend()

    plt.show()
    

if __name__ == "__main__":
    main()
