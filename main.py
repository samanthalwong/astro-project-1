import numpy as np

#QUESTION 1
def rk2(n, x, h, derivsRK):
    """Runge-Kutta integrator (2nd order)
       Input arguments -
        n = polytropic index
        x = current value of dependent variable
        h = step size (delta x)
        derivsRK = right hand side of the ODE; derivsRK is the
                  name of the function which returns dx/dt
                  Calling format derivsRK (x,t,param).
       Output arguments -
        xout = new value of x after a step of size tau
    """
    half_h = 0.5 * h
    F1 = derivsRK(n, x)
    r_half = n + half_h
    xtemp = x + half_h * F1
    F2 = derivsRK(r_half, xtemp)
    xout = x + h * F2
    return xout

def odes(n,s):
    """
    Returns RHS of coupled ODE drho_bar/dr_bar
    inputs:
    n: polytropic index
    s: state vector [theta, zeta, v]
    output:
    derivatives [du/dzeta, dv/dzeta]
    """
    #unpack state vector
    theta = s[0]
    zeta = s[1]
    v = s[2]

    u = theta

    #solve derivatives
    du_dzeta = v
    dv_dzeta = -u**n-2*v/zeta

    return np.array([du_dzeta, dv_dzeta])
