import numpy as np,math,qutip,random,copy,os,localTools,scipy,read_write

def tau(state,ref):
    res=0j
    for i in range(len(state)):
        res+=state[i]*np.conjugate(ref[i])
    try:
        return res[0]
    except TypeError:
        return res

def chis_tau(state,ref):
    return ref

def J_T_taus(state,ref):
    tau_val = tau(state,ref)
    return np.real(tau_val*np.conjugate(tau_val))

def chis_taus(state,ref):
    #return -tau(state,ref) * ref
    return ref

def J_T_re(state,ref):
    return 1-np.real(tau(state,ref))

def J_T_ss(state,ref):
    tau_val = tau(state,ref)
    return 1-np.real(tau_val*np.conjugate(tau_val))

def chis_ss(state,ref):
    return tau(state,ref) * ref

def J_T_abs(state,ref):
    return 1-np.abs(tau(state,ref))


def localFidelity(state,ref='None'):
    if isinstance(state,qutip.Qobj):state=state.full()
    if isinstance(ref,qutip.Qobj):ref=ref.full()
    if localTools.isinstanceVector(state):dm=localTools.densityMatrix(state)
    else:dm=state
    if localTools.isinstanceVector(ref):ref=localTools.densityMatrix(ref)
    ref=scipy.linalg.sqrtm(ref)
    mat=np.matmul(ref,np.matmul(dm,ref))
    return np.real(np.trace(scipy.linalg.sqrtm(mat)))**2
def chis_inFidelity(fw_states_T, objectives, tau_vals):
    ref=objectives[0].target.full()     
    return [qutip.Qobj(np.matmul(localTools.densityMatrix(ref),fw_states_T[0].full()))]
def J_T_inFidelity(fw_states_T,objectives,tau_vals=None,**kwargs):
    state=fw_states_T[0].full()
    ref=objectives[0].target.full()
    return inFidelity(state,localTools.densityMatrix(ref))
def inFidelity(state,ref=False):
    return 1-localFidelity(state,ref)

def functional_master(functional_name):
    if functional_name == 'inFidelity' or 'JT_ss':
        return [chis_inFidelity,J_T_inFidelity,inFidelity]
    if functional_name == 'XN':
        return [chis_XN,J_T_XN,J_T_XN]
    #surnames = ["chis_", "J_T_", ""]
    #functional_names = [surname + functional_name for surname in surnames]
    #return [eval(functional) for functional in functional_names]

def J_T_opVarN(fw_states_T,objectives=None,tau_vals=None,**kwargs):
    return opVarN(fw_states_T[0].full())

def opVar(mat,state):
    mat_sqr = np.matmul(mat,mat)
    state_dm = localTools.densityMatrix(state)
    return np.real(np.trace(np.matmul(state_dm,mat_sqr))-np.trace(np.matmul(state_dm,mat))**2)

def mat_X(num_qubit,X_factor):
    X = np.zeros([2**num_qubit, 2**num_qubit])
    for i in range(2**num_qubit):
        X[i][2**num_qubit - i - 1] = X_factor
    return X

def mat_N(num_qubit,N_factor,shuffle_key=False):
    N = np.zeros([2**num_qubit, 2**num_qubit])
    N_val = np.sqrt(2 ** (num_qubit - 2 + N_factor))
    valsDeposit0 = list(
        np.linspace(1, N_val, 2 ** (num_qubit - 1)))
    if shuffle_key:random.shuffle(valsDeposit0)
    valsDeposit1 = copy.deepcopy(valsDeposit0)
    valsDeposit1.reverse()
    valsDeposit = valsDeposit0 + valsDeposit1
    for i in range(2**num_qubit):
        N[i][i] += valsDeposit[i]
    return N


def opVarX(state,num_qubit,X_factor = 1):
    return opVar(mat_X(num_qubit,X_factor),state)
def opVarN(state,num_qubit,N_factor = 1,shuffle_key=False):
    return opVar(mat_N(num_qubit,N_factor,shuffle_key),state)

def chis_XN(fw_states_T, objectives=None, tau_vals=None):
    state=fw_states_T[0].full()
    num_qubit=int(math.log(len(state),2))
    rho=localTools.densityMatrix(state)
    X_factor = 1
    N_factor = 1.5
    N_Operator=mat_N(num_qubit,N_factor)
    N2=np.matmul(N_Operator,N_Operator)
    N_exp=np.trace(np.matmul(N_Operator,rho))
    X_Operator=mat_N(num_qubit,X_factor)
    X2 = np.eye(2**num_qubit)
    X_exp=np.trace(np.matmul(X_Operator,rho))
    chisKet=0.5*np.matmul(2*N_exp*N_Operator-N2,state)
    chisKet+=0.5*np.matmul(2*X_exp*X_Operator-X2,state)
    return [qutip.Qobj(chisKet)]

def J_T_XN(fw_states_T,objectives,tau_vals=None,**kwargs):
    state=fw_states_T[0].full()
    num_qubit = int(np.log2(len(state)))
    X_factor = 1
    N_factor = 1.5
    return 0.5 * opVar(mat_X(num_qubit,X_factor),state) + 0.5 *opVar(mat_N(num_qubit,N_factor),state)

def bestCat(state,min_key=True):
    num_qubit=int(math.log(len(state),2))
    allCats=[]
    for i in range(2**(num_qubit-1)):
        allCats.append(np.complex128(np.zeros([2**num_qubit,1])))
        allCats[i][i]=np.exp(1j*np.angle(state[i][0]))/np.sqrt(2)
        allCats[i][2**(num_qubit)-i-1]=np.exp(1j*np.angle(state[2**(num_qubit)-i-1][0]))/np.sqrt(2)
    fidelityList=[J_T_ss(state,thisCat) for thisCat in allCats]
    if min_key:return f'{fidelityList.index(min(fidelityList))}+', min(fidelityList)
    else:return fidelityList

def bestCat_phase(state,min_key=True,val_opt = False):
    num_qubit=int(math.log(len(state),2))
    allCats=[]
    r_angle = []
    for i in range(2**(num_qubit-1)):
        allCats.append(np.complex128(np.zeros([2**num_qubit,1])))
        allCats[i][i]=np.exp(1j*np.angle(state[i][0]))/np.sqrt(2)
        allCats[i][2**(num_qubit)-i-1]=np.exp(1j*np.angle(state[2**(num_qubit)-i-1][0]))/np.sqrt(2)
        r_angle.append(np.angle(state[i][0])-np.angle(state[2**(num_qubit)-i-1][0]))
    fidelityList=[J_T_ss(state,thisCat) for thisCat in allCats]
    min_loc = fidelityList.index(min(fidelityList))
    if val_opt:return min(fidelityList),r_angle[min_loc]/np.pi
    if min_key:return f'{min_loc}+', min(fidelityList),f'{r_angle[min_loc]/np.pi}pi'
    else:return fidelityList,r_angle

def cat_res(folder,T,canoLabel):
    if os.path.exists(f'{folder}/psi_final_after_oct.dat'):
        state = read_write.stateReader(f'{folder}/psi_final_after_oct.dat',16)
        state = localTools.rotate_state(state,4,0,float(T))
        JT_i = bestCat(state,False)
        best_JT_ss_m = min(JT_i)
        best_cat = f'{JT_i.index(best_JT_ss_m)}+'
        best_JT_ss_b = JT_i[int(canoLabel[0])]
        varX_i = opVarX(state,4)
        varN_i = opVarN(state,4)
        return best_cat,best_JT_ss_b,best_JT_ss_m,varX_i,varN_i
    else:return None,1,None,None,None


def cat_res_P(folder,T):
    if os.path.exists(f'{folder}/psi_final_after_oct.dat'):
        state = read_write.stateReader(f'{folder}/psi_final_after_oct.dat',16)
        state = localTools.rotate_state(state,4,0,float(T))
        print(bestCat_phase(state,True))

