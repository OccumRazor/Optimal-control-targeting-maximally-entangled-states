import numpy as np,scipy
from scipy.special import jv,j0,j1

'''.vscode\
  !! * :cite:`Tal-EzerJCP84`
  !!   Tal-Ezer and Kosloff, J. Chem. Phys. 81, 3967 (1984)
  !! * :cite:`NdongJCP2009`
  !!   Ndong et al.,  J. Chem. Phys. 130, 124108 (2009)
  Am-Shallem, Morag, et al. "Three approaches for representing Lindblad dynamics by a matrix-vector notation." arXiv preprint arXiv:1510.08634 (2015).
'''

def log_factorial(num):
    result = 0
    for i in range(1,num+1):
        result += np.log10(i)
    return result


def BF(k,l,x):
    return ((-1)**l)*10**((2*l+k)*np.log10(x/2) - log_factorial(l) - log_factorial(k+l))
def Bessel_function(num,k,lower_bound=1e-17):
    l = 0
    coe_amp = BF(k,l,num)
    coefficient = coe_amp
    while np.abs(coe_amp) > lower_bound:
        l += 1
        coe_amp = BF(k,l,num)
        coefficient += coe_amp
    return coefficient

def expm(herm_mat,initial_state,dt,backwards = False):
    if isinstance(herm_mat,list):herm_mat=np.array(herm_mat)
    if backwards: Ut = scipy.linalg.expm(1j * herm_mat * dt)
    else: Ut = scipy.linalg.expm(-1j * herm_mat * dt)
    return Ut.dot(initial_state)

def Chebyshev(herm_mat,initial_state,E_max,E_min,dt,backwards = False):
    if isinstance(herm_mat,list):herm_mat=np.array(herm_mat)
    Delta = E_max - E_min
    R = Delta  * dt / 2
    max_eval = int(4 * np.abs(R))
    if max_eval < 40: max_eval = 40
    if max_eval > 500: raise ValueError(f"variable max_eval {max_eval} is larger than 500, decrease dt to decrease max_eval.")
    if backwards == False: # forward propagation 
        p0 = -1j
    else:
        p0 = 1j
    H = (p0 * dt / R) * herm_mat
    norm_factor = p0 * (E_max + E_min) / Delta
    phase_factor = np.exp(p0 * (E_max + E_min) * dt / 2)
    phi0 = initial_state
    Bessel_index = [j0(R),j1(R)] + [jv(i,R) for i in range(2,max_eval)]
    phi1 = H.dot(phi0) - norm_factor * phi0
    final_state = 2 * Bessel_index[1] * phi1 + Bessel_index[0] * phi0
    norm_factor *= 2
    H *= 2
    for k in range(2,max_eval):
        phi2 = H.dot(phi1)  - norm_factor * phi1 + phi0
        final_state += 2 * Bessel_index[k] * phi2
        phi0 = phi1
        phi1 = phi2
    final_state *= phase_factor
    return final_state

def Newton(initial_state,backwards = False):
    return 0
