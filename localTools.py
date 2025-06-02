import math,  random, os, shutil, numpy as np,qutip,time,scipy,krotov
from scipy.interpolate import interp1d
from scipy.sparse import coo_matrix

freqZ=[5.158,5.034,4.863,5.125,5.159,4.979,5.157]
freqX=freqZ
couplings=[0.0722,0.11089,0.097732,0.080947,0.121882,0.1003596]


def binary_search(data_array,target,source = None):
    # this is not the precise binary search
    # it's return value 'L' suits data_array[L-1] <= target <= data_array[L+1]
    if data_array[0] <= target:return 0
    if data_array[-1] >= target:return -1
    if (data_array[0]-target)*(data_array[1]-target)<=0:return 0
    if (data_array[-1]-target)*(data_array[-2]-target)<=0:return -1
    if len(data_array)<16:
        L = 0
        while (data_array[L]-target)*(data_array[L+1]-target)>0 and L < len(data_array) - 2:
            L += 1
        return L
    L = 0
    R = len(data_array)-1
    while L != R:
        m=math.ceil((L+R)/2)
        if data_array[m] < target:R = m -1
        else:L=m
    if (data_array[L-1]-target)*(data_array[L+1]-target)<=0:return L
    else:
        raise ValueError(f'target {target} does not sit within L({L} pm 1)({data_array[L-1]}, {data_array[L]},{data_array[L+1]})\nCheck whether oct_iter in {source} is monotonic.')

def del_redundent(folder,black_list):
    files = os.listdir(folder)
    for file in files:
        if any([key_word in file for key_word in black_list]):
            #print(f'existence of {folder+file}:{os.path.exists(folder+file)}')
            time.sleep(0.2)
            os.remove(folder+file)
    return 0

def read_oct_iters(working_folder,delta_JT = False):
    if not os.path.exists(working_folder+'/oct_iters.dat'):
        if delta_JT: return [1,0]
        return 1
    this_JTS = np.loadtxt(working_folder+'/oct_iters.dat', usecols=(1))
    if delta_JT:return [this_JTS[-1],(this_JTS[-2]-this_JTS[-1])/this_JTS[-1]]
    return this_JTS[-1]

def S(t,t_start,t_stop,t_rise,t_fall):
    """Scales the Krotov methods update of the pulse value at the time t"""
    return krotov.shapes.flattop(t, t_start=t_start, t_stop=t_stop, t_rise=t_rise, t_fall=t_fall, func='blackman')

def half_step_tlist(qdyn_tlist):
    if len(qdyn_tlist) == 3:qdyn_tlist=qdyn_tlist[1:]
    dt = (qdyn_tlist[0]) / (qdyn_tlist[1] - 1)
    tgrid = np.linspace(
        float(dt / 2),
        float(qdyn_tlist[0] - dt / 2),
        qdyn_tlist[1] - 1,
        dtype=float,
    )  #! here the default is np.float 64, it has been changed manually, also in QDYN python package
    return tgrid

def control_generator_S2L(num_qubit, control_source, endTime=None, header=None):
    control_args = []
    for i in range(num_qubit):
        if os.path.isfile(control_source + f"{header}_{i}.dat"):
            detupleTlist, detupleGuess = controlReader(
                control_source + f"{header}_{i}.dat"
            )
            oriTime = float(detupleTlist[-1] + (detupleTlist[-1] - detupleTlist[-2])/2)
        else:
            raise KeyError(f"Unable to find control in {control_source}{header}, {header}_{i}.dat does not exist.")
        former_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate"
        )
        new_tlist = half_step_tlist([endTime, len(detupleGuess) + 1])
        delta_T = endTime - oriTime
        new_Amp = []
        for t in new_tlist:
            if t<= delta_T: new_Amp.append(0)
            else: new_Amp.append(former_fit(float(t-delta_T)))
        cubicSpline_fit = interp1d(
            new_tlist, new_Amp, kind="cubic", fill_value="extrapolate")
        control_args.append({"fit_func": cubicSpline_fit})
    return control_args

def control_generator(
    num_qubit, control_source, endTime=None, header=None
):
    if isinstance(control_source, str):
        control_args = control_generator_read(
            num_qubit, control_source, header, endTime)
    else:
        control_args = control_generator_random(
            num_qubit, control_source, endTime)
    return control_args

def control_generator_read(num_qubit, control_source, header, endTime):
    control_args = []
    for i in range(num_qubit):
        if os.path.isfile(control_source + f"{header}_{i}.dat"):
            detupleTlist, detupleGuess = controlReader(
                control_source + f"{header}_{i}.dat"
            )
            #detupleTlist = half_step_tlist([endTime, len(detupleGuess) + 1])
        else:
            raise KeyError(f"Unable to find control in {control_source}{header}, {header}_{i}.dat does not exist.")
        cubicSpline_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate"
        )
        control_args.append({"fit_func": cubicSpline_fit})
    return control_args


def control_generator_random(num_qubit, guess_amp, endTime):
    num_points = 25
    control_args = []
    for i in range(num_qubit):
        detupleGuess = (
            [0,0,0] + [guess_amp * random.random() - guess_amp / 2 for _ in range(num_points)] + [0,0,0])
        detupleTlist = np.linspace(0, endTime, len(detupleGuess))
        cubicSpline_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate"
        )
        control_args.append({"fit_func": cubicSpline_fit})
    return control_args

def store_control_source(control_source,dst_folder):
    src_files = os.listdir(control_source)
    opt_label = 1
    while f'former_opt_{opt_label}' in src_files:
        opt_label += 1
    if control_source == dst_folder:
        if not os.path.exists(dst_folder+f'former_opt_{opt_label}/'):
            os.mkdir(dst_folder+f'former_opt_{opt_label}/')
        for file in src_files:
            if 'former_opt' not in file:
                shutil.copy(control_source+file,dst_folder+f'former_opt_{opt_label}/')
    else:
        shutil.copytree(control_source,dst_folder+f'former_opt_{opt_label}/')
        for i in range(1,opt_label):
            shutil.copytree(dst_folder+f'former_opt_{opt_label}/former_opt_{i}/',dst_folder+f'former_opt_{i}/')
            shutil.rmtree(dst_folder+f'former_opt_{opt_label}/former_opt_{i}/')

sq_dict = {
    "I": [[1, 0], [0, 1]],
    "X": [[0, 1], [1, 0]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
    "0": [[1, 0], [0, 0]],
    "1": [[0, 0], [0, 1]],
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
}


def fastKron(iptString):
    # Wenn es in iptString '+' gibt, zuerst iptString.split('+'), dann alle sammeln.
    if not isinstance(iptString, str):
        raise TypeError(
            "Input must be a string (consisted of +, 0, 1, I, H, X, Y and Z)"
        )
    if len(iptString) == 0:
        return np.array([[1]])
    if "+" in iptString:
        addMats = iptString.split("+")
        addMats = [fastKron(addMat) for addMat in addMats]
        return sum(addMats)
    kronMat = sq_dict[iptString[0]]
    for thisKey in iptString[1:]:
        kronMat = np.kron(kronMat, sq_dict[thisKey])
    return np.array(kronMat)

def controlReader(file_name, dummy=False):
    thisTlist, thisControl = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    detupleControl = []
    detupleTlist = []
    if dummy:
        thisControl, thisControlImg = np.split(
            np.loadtxt(file_name, usecols=(1, 2)), 2, axis=1
        )
        # in this case, thisControl is the real part, while thisControlImg is the imaginary part.
        detupleControlImg = []
        for j in range(len(thisTlist)):
            detupleControlImg.append(thisControlImg[j][0])
        detupleControlImg = list(detupleControlImg)
    for j in range(len(thisTlist)):
        detupleControl.append(thisControl[j][0])
        detupleTlist.append(thisTlist[j][0])
    detupleTlist = list(detupleTlist)
    detupleControl = list(detupleControl)
    if dummy:
        return detupleTlist, detupleControl, detupleControlImg
    else:
        return detupleTlist, detupleControl

def stateReader(file_name, num_qubit,n_levels = False):
    ids, res = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    try:
        res, ims = np.split(np.loadtxt(file_name, usecols=(1, 2)), 2, axis=1)
        imag_flag = True
    except Exception:
        imag_flag = False
        #ims = np.array([[0] * len(res)])
    if n_levels:
        state = np.zeros([2*n_levels**num_qubit, 1], dtype=np.complex64)    
    else:state = np.zeros([2**num_qubit, 1], dtype=np.complex64)
    ids = ids.T[0]
    res = res.T[0]
    ids = [int(ids[i] - 1) for i in range(len(ids))]
    #ids = [int(ids[i]) for i in range(len(ids))]
    if imag_flag:
        ims = ims.T[0]
        for i in range(len(ids)):
            state[ids[i]][0] += res[i] + 1j * ims[i]
    else:
        for i in range(len(ids)):
            state[ids[i]][0] += res[i]
    dm = densityMatrix(state)
    traceValue = np.trace(dm)
    event = False
    if np.abs(np.imag(traceValue)) > 1e-6 or np.abs(np.imag(traceValue)) < -1e-6:
        event = True
        print("stateReader: trace imagnary part non-zero")
        print(state)
        print(np.imag(traceValue))
    if np.abs(np.real(traceValue)) > 1 + 1e-6 or np.abs(np.real(traceValue)) < 1 - 1e-6:
        event = True
        print("stateReader: trace real part non-one")
        print(state)
        print(np.real(traceValue))
    if event:
        with open("non_unity_report.txt", "a") as file:
            file.write(f"Occured at \n{file_name}\n")
    return state

def write_pulse(tlist,pulse,file_title):
    with open(file_title,'a') as log_f:
        for t,amp in zip(tlist,pulse):
            log_f.write(f'{t:.16e}    {amp:.16e}\n')


def densityMatrix(state):
    if not isinstanceVector(state):
        return state
    if isinstance(state[0], list) or isinstance(state[0], np.ndarray):
        state = list(np.array(state).T[0])
    #dm = []
    dm = np.zeros([len(state),len(state)],dtype=np.complex128)
    for i in range(len(state)):
        #dm.append([])
        for j in range(len(state)):
            #dm[i].append(complex(state[i] * np.conjugate(state[j])))
            dm[i][j] = complex(state[i] * np.conjugate(state[j]))
    return np.array(dm)

def isinstanceVector(state):
    # this function check whether the state is a ket/bra or not.
    if not isinstance(state, list) and not isinstance(state, np.ndarray):
        return 1
    if len(state[0]) == 1:
        return 1
    return 0

def cano_freq():
    n_list = range(8)
    binary_list = [bin(n)[2:] for n in n_list]
    for i in range(len(binary_list)):
        if len(binary_list[i]) < 4:
            binary_list[i] = '0' * (4 - len(binary_list[i])) +binary_list[i]
    freqs = []
    for i in range(len(binary_list)):
        freq0 = 0
        for j in range(4):
            freq0 += (-1)**(int(binary_list[i][j])) * freqZ[j]
        freqs.append(np.abs(freq0))
    center_locs = [20.783,20.722,20.886,38.5,20.66,34.1,28.5,20.543]
    return freqs,center_locs

def f_phi(dist):
    return dist - 1

def cat_phi(T,canoLabel):
    if not isinstance(T,np.ndarray):T = np.array(T)
    #phi = (T - 29.0) * np.pi/2/(32.7 - 29.0)
    freqs,center_locs = cano_freq()
    cat_num = int(canoLabel[0])
    freq = freqs[cat_num]
    center_loc = center_locs[cat_num]
    period = np.pi / freq
    distance = ((T - center_loc) % period) / period
    phi = f_phi(distance)# * np.pi
    return phi

def canoGHZGen(num_qubit, canoLabel, phi = 0):
    # Variable canoLabel can be either a label like 3+ or a pure state 010
    if phi != 0:
        state = np.zeros([2**num_qubit, 1],dtype=np.complex128)
    else:
        state = np.zeros([2**num_qubit, 1])
        phi = 1
    if canoLabel[-1] != "+" and canoLabel[-1] != "-":
        state[int(canoLabel, 2)][0] = 1
        return qutip.Qobj(state)
    stateLabel = int(canoLabel[: len(canoLabel) - 1])
    state[stateLabel][0] = 1 / np.sqrt(2)
    if canoLabel[-1] == "+":
        state[2**num_qubit - stateLabel - 1][0] = phi / np.sqrt(2)
    else:
        state[2**num_qubit - stateLabel - 1][0] = -phi / np.sqrt(2)
    return qutip.Qobj(state)

def rotate_state(state, num_qubit, direction=0, endTime=0):
    H0 = np.zeros([2**num_qubit, 2**num_qubit], dtype=np.complex128)
    for i in range(num_qubit):
        H0 += (
            0.5
            * freqX[i]
            * fastKron("I" * i + "Z" + "I" * (num_qubit - i - 1))
            * endTime)
    if isinstance(state, qutip.Qobj):
        state = state.full()
    if direction:
        return np.matmul(
            scipy.linalg.expm(1j * H0), state
        )  # \psi_RWA=U\dagger\psi_nR, return RWA frame state.
    else:
        return np.matmul(
            scipy.linalg.expm(-1j * H0), state
        )  # \psi_nR=U\psi_RWA, return lab frame state.

def rotate_matrix(mat, num_qubit, direction=0, endTime=0):
    H0 = np.zeros([2**num_qubit, 2**num_qubit], dtype=np.complex128)
    for i in range(num_qubit):
        H0 += (
            0.5
            * freqX[i]
            * fastKron("I" * i + "Z" + "I" * (num_qubit - i - 1))
            * endTime)
    if isinstance(mat, qutip.Qobj):
        mat = mat.full()
    if direction:return np.matmul(np.matmul(scipy.linalg.expm(1j*H0),mat),scipy.linalg.expm(-1j*H0))
    else:return np.matmul(np.matmul(scipy.linalg.expm(-1j*H0),mat),scipy.linalg.expm(1j*H0))

def Hamiltonian_Spin_Chain_NR(num_qubit, **kwargs):
    # OC here refers to operator controllable, https://arxiv.org/abs/2212.04828
    # intrinsic frequencies comes from IBM_PERTH, https://quantum-computing.ibm.com/services/resources?tab=systems&system=ibm_perth
    if num_qubit > 7:
        raise ValueError("number of qubit should be no larger than 7.")
    # 2pi/J in (51.6,87.0)
    # 2pi/f in (1.218,1.292)
    # Suggested starting T: 20 ns
    H0 = np.zeros([2**num_qubit, 2**num_qubit], dtype=np.complex64)
    HC = []
    for i in range(num_qubit - 1):
        H0 += couplings[i] * fastKron("I" * i + "ZZ" + "I" * (num_qubit - i - 2))
    for i in range(num_qubit):
        H0 += 0.5 * freqZ[i] * fastKron("I" * i + "Z" + "I" * (num_qubit - i - 1))
    for i in range(num_qubit):
        HC.append(fastKron("I" * i + "X" + "I" * (num_qubit - i - 1)))
    # each control is of form 2*cos(freqZ[i]t)c_i(t)\sigma_X^i, c_i(t) is the pulse to be optimized, use random_guess_cos(t,control_args)
    return H0, HC

def Hamiltonian_Spin_Chain(num_qubit, **kwargs):
    if num_qubit > 7:
        raise ValueError("number of qubit should be no larger than 7.")
    # 2pi/J in (51.6,87.0)
    # 2pi/f in (1.218,1.292)
    # Suggested starting T: 20 ns
    H0 = np.zeros([2**num_qubit, 2**num_qubit], dtype=np.complex64)
    HC = []
    for i in range(num_qubit - 1):
        H0 += couplings[i] * fastKron("I" * i + "ZZ" + "I" * (num_qubit - i - 2))
    for i in range(num_qubit):
        H0 += 0.5 * (freqZ[i] - freqX[i]) * fastKron("I" * i + "Z" + "I" * (num_qubit - i - 1))
    for i in range(num_qubit):
        HC.append(fastKron("I" * i + "X" + "I" * (num_qubit - i - 1)))
    # control in RWA is of form c_i(t), use random_guess(t,control_args)
    return H0, HC

def random_guess(t, control_args):
    fit_func = control_args.get("fit_func")
    return fit_func(t)

def random_guess_cos(t, control_args):
    fit_func = control_args.get("fit_func")
    frequency = control_args.get("freqX")
    return fit_func(t) * np.cos(frequency * t) * 2

def Hamiltonian_NR(num_qubit=3, **kwargs):
    H0, Hc = Hamiltonian_Spin_Chain_NR(num_qubit, **kwargs)
    return [qutip.Qobj(H0)] + [[qutip.Qobj(Hc[i]),lambda t,args:random_guess(t,args)] for i in range(len(Hc))]

def Hamiltonian(num_qubit=3, **kwargs):
    H0, Hc = Hamiltonian_Spin_Chain(num_qubit, **kwargs)
    return [qutip.Qobj(H0)] + [[qutip.Qobj(Hc[i]),lambda t,args:random_guess(t,args)] for i in range(len(Hc))]


