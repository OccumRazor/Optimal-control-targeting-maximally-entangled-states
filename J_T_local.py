import numpy as np,math,qutip,random,copy,os,localTools,scipy,read_write

def tau(state,ref):
    if isinstance(state,qutip.Qobj):state = state.full()
    if isinstance(ref,qutip.Qobj):ref=ref.full()
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

def JT_tau(states,refs):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            val += J_T_taus(states[i],refs[i])
        return val / len(states)
    else:
        return J_T_taus(states,refs)

def J_T_re(state,ref):
    return 1-np.real(tau(state,ref))

def chis_ss(states,refs):
    if isinstance(states,list):
        for i in range(len(states)):
            if isinstance(states[i],qutip.Qobj):states[i] = states[i].full()
            if isinstance(refs[i],qutip.Qobj):refs[i] = refs[i].full()
        chis = []
        for i in range(len(states)):
            chis.append(tau(states[i],refs[i]) * refs[i])
        return chis
    else:
        return tau(states,refs) * refs

def JT_ss(states,refs):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            tau_val = tau(states[i],refs[i])
            val += 1-np.real(tau_val*np.conjugate(tau_val))
        return val / len(states)
    else:
        tau_val = tau(states,refs)
        return 1-np.real(tau_val*np.conjugate(tau_val))


def J_T_abs(state,ref):
    return 1-np.abs(tau(state,ref))

def mat_X(num_qubit,X_factor):
    X = np.zeros([2**num_qubit, 2**num_qubit],dtype=np.complex128)
    for i in range(2**num_qubit):
        X[i][2**num_qubit - i - 1] = X_factor
    return X

def mat_N(num_qubit,N_factor,shuffle_key=False):
    N = np.zeros([2**num_qubit, 2**num_qubit])
    N_val = 2 ** (num_qubit - 2 + N_factor)
    valsDeposit0 = list(
        np.linspace(1, N_val, 2 ** (num_qubit - 1)))
    if shuffle_key:random.shuffle(valsDeposit0)
    valsDeposit1 = copy.deepcopy(valsDeposit0)
    valsDeposit1.reverse()
    valsDeposit = valsDeposit0 + valsDeposit1
    for i in range(2**num_qubit):
        N[i][i] += valsDeposit[i]
    return N

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
    if functional_name in ['inFidelity','JT_ss']:
        return [chis_inFidelity,J_T_inFidelity,inFidelity]
    if functional_name == 'XN':
        return [chis_XN,J_T_XN,J_T_XN]
    if functional_name == 'X':
        return [chis_X,J_T_X,J_T_X]
    #surnames = ["chis_", "J_T_", ""]
    #functional_names = [surname + functional_name for surname in surnames]
    #return [eval(functional) for functional in functional_names]

def chis_var(psi_Ts,operators,op_sqs):
    chis_Kets = []
    for psi_T in psi_Ts:
        chis_Ket = np.zeros([len(operators[0]),1],dtype=np.complex128)
        if isinstance(psi_T,qutip.Qobj):psi_T = psi_T.full()
        state_dm = localTools.densityMatrix(psi_T)
        for op,op_sq in zip(operators,op_sqs):
            op_exp = np.trace(np.matmul(state_dm,op))
            #print(f'chis exp part {op_exp}')
            chis_Ket += (1/len(operators)) * np.matmul(2*op_exp*op - op_sq,psi_T)
        chis_Kets.append(qutip.Qobj(chis_Ket))
    return chis_Kets

def JT_var(psi_Ts,operators,op_sqs):
    JT_val = 0.
    for psi_T in psi_Ts:
        if isinstance(psi_T,qutip.Qobj):psi_T = psi_T.full()
        state_dm = localTools.densityMatrix(psi_T)
        for op,op_sq in zip(operators,op_sqs):
            JT_val += np.real(np.trace(np.matmul(state_dm,op_sq))-np.trace(np.matmul(state_dm,op))**2)
    return JT_val / len(operators) / len(psi_Ts)

def opVar(mat,state,f_name=None):
    mat_sqr = np.matmul(mat,mat)
    state_dm = localTools.densityMatrix(state)
    if f_name:
        state_text = read_write.matrix2text(state_dm)
        with open(f_name,'w') as state_f:
            state_f.write(state_text)
    return np.real(np.trace(np.matmul(state_dm,mat_sqr))-np.trace(np.matmul(state_dm,mat))**2)

def opVarX(state,num_qubit,X_factor = 1):
    return opVar(mat_X(num_qubit,X_factor),state)
def opVarN(state,num_qubit,N_factor = 1,shuffle_key=False):
    return opVar(mat_N(num_qubit,N_factor,shuffle_key),state)

def chis_X(fw_states_T, objectives=None, tau_vals=None):
    state=fw_states_T[0].full()
    num_qubit=int(math.log(len(state),2))
    rho=localTools.densityMatrix(state)
    X_factor = 1
    X_Operator=mat_X(num_qubit,X_factor)
    X2 = np.eye(2**num_qubit)
    X_exp=np.trace(np.matmul(X_Operator,rho))
    chisKet=np.matmul(2*X_exp*X_Operator-X2,state)
    chisKet = localTools.rotate_state(chisKet,4,1,20.0)
    return [qutip.Qobj(chisKet)]

def J_T_X(fw_states_T,objectives,tau_vals=None,**kwargs):
    state=fw_states_T[0].full()
    num_qubit = int(np.log2(len(state)))
    X_factor = 1
    return opVar(mat_X(num_qubit,X_factor),state)

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
    fidelityList=[JT_ss(state,thisCat) for thisCat in allCats]
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
    fidelityList=[JT_ss(state,thisCat) for thisCat in allCats]
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

def cat_res_2(state,T,canoLabel):
    state = localTools.rotate_state(state,4,0,float(T))
    JT_i = bestCat(state,False)
    best_JT_ss_m = min(JT_i)
    min_cat = f'{JT_i.index(best_JT_ss_m)}+'
    best_JT_ss_b = JT_i[int(canoLabel[0])]
    varX_i = opVarX(state,4)
    varN_i = opVarN(state,4)
    return min_cat,best_JT_ss_b,best_JT_ss_m,varX_i,varN_i


def cat_res_P(folder,T):
    if os.path.exists(f'{folder}/psi_final_after_oct.dat'):
        state = read_write.stateReader(f'{folder}/psi_final_after_oct.dat',16)
        state = localTools.rotate_state(state,4,0,float(T))
        print(bestCat_phase(state,True))

