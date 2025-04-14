import propagation_method,qutip
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix

def KROTOV_CHEBY(H,state,dt,c_ops=None,backwards=False,initialize=False):
    if c_ops is None:
        c_ops = []
    if len(c_ops) > 0:
        raise NotImplementedError("Liouville exponentiation not implemented")
    assert isinstance(H, list) and len(H) > 0
    if isinstance(H[0], list):
        Ht = H[0][1] * H[0][0]
    else:
        Ht = H[0]
    for part in H[1:]:
        if isinstance(part, list):
            Ht += part[1] * part[0]
        else:
            Ht += part
    ok_types = (state.type == 'oper' and Ht.type == 'super') or (
        state.type in ['ket', 'bra'] and Ht.type == 'oper'
    )
    Ht = Ht.full()
    eig_vals = eigsh(Ht,return_eigenvectors=False)
    E_max = max(eig_vals)
    E_min = min(eig_vals)
    return qutip.Qobj(propagation_method.Chebyshev(Ht,state.full(),E_max,E_min,dt,backwards = backwards))

'''
Somehow this implementation is way slower.
def KROTOV_CHEBY(H,state,dt,c_ops=None,backwards=False,initialize=False):
    if c_ops is None:
        c_ops = []
    if len(c_ops) > 0:
        raise NotImplementedError("Liouville exponentiation not implemented")
    assert isinstance(H, list) and len(H) > 0
    eqm_factor = -1j  # factor in front of H on rhs of the equation of motion
    eqm_factor = 1
    if isinstance(H[0], list):
        if H[0][1].type == 'super':
            eqm_factor = 1
        if backwards:
            eqm_factor = eqm_factor.conjugate()
        A = (eqm_factor * H[0][1]) * H[0][0]
    else:
        if H[0].type == 'super':
            eqm_factor = 1
        if backwards:
            eqm_factor = eqm_factor.conjugate()
        A = eqm_factor * H[0]
    for part in H[1:]:
        if isinstance(part, list):
            A += (eqm_factor * part[1]) * part[0]
        else:
            A += eqm_factor * part
    ok_types = (state.type == 'oper' and A.type == 'super') or (
        state.type in ['ket', 'bra'] and A.type == 'oper'
    )
    E_max = 100
    E_min = -100
    return qutip.Qobj(propagation_method.Chebyshev(A.full(),state.full(),E_max,E_min,dt,backwards = backwards))
'''