
import numpy as np

def runge_kutta4(fun, dt:float, time, y0:float):
    """ODE integration using 4th-order Runge-Kutta

    Parameters
    ----------
    fun  : ODE to be integrated, if function is vector function y0 and function output must be a single row i.e. [1, 2, 3, ...], not [[1], [2], [3], ...]
    dt   : time step
    time : array of evenly spaced time intervals
    y0   : initial condition
    """

    yfinal = np.empty((len(y0), len(time)))
    yfinal[:, 0] = y0
    for i, t in enumerate(time):
        f1 = fun(t, y0)
        f2 = fun(t + dt / 2, y0 + (dt / 2) * f1)
        f3 = fun(t + dt / 2, y0 + (dt / 2) * f2)
        f4 = fun(t + dt, y0 + dt * f3)
        yfinal[:, i] = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        y0 = yfinal[:, i]
    return yfinal
