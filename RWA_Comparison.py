import numpy as np,csv,config_task,J_T_local,localTools,time,matplotlib.pyplot as plt,task_obj

def plot_controls(tlist_NR,tlist_RWA,controls_NR,controls_RWA):
    fig,ax = plt.subplots(2,2,figsize=(12,12))
    for i in range(4):
        ax[int(i/2)][i%2].plot(tlist_NR,[2*controls_NR[i][j] for j in range(len(controls_NR[i]))],label=f'{i}')
        ax[int(i/2)][i%2].plot(tlist_RWA,controls_RWA[i])
    plt.legend()
    plt.show()

def plot_controls_sg(tlist,controls):
    for i in range(4):
        if isinstance(controls[i],dict):plt.plot(tlist,controls[i]["fit_func"](tlist),label=i)
        else:plt.plot(tlist,controls[i],label=i)
    plt.legend()
    plt.show()

def RWA_control(qubit_id,control):
    ori_fit = control["fit_func"]
    return {'fit_func':lambda t:2*ori_fit(t)*np.cos(t*localTools.freqX[qubit_id])}

def Krotov_config_task(tlist,control_source=None):
    num_qubit = 4
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    pulse_options={}
    control_args = localTools.control_generator(num_qubit,control_source,tlist[1],'pulse_oct')
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(args=control_args[i-1])
    prop = task_obj.Propagation(H,tlist,'cheby',initial_states,'pulse_initial',pulse_options)
    return prop

def compare(T,control_source):
    #prop,config = config_task.config_prop(path)
    prop = Krotov_config_task([0,T,801],control_source)
    tlist_NR = prop.tlist_long
    tic = time.time()
    psi_T_0 = prop.propagate()
    tac = time.time()
    psi_T_0 = [localTools.rotate_state(psi_T_0[0].full(),4,0,T)]
    #c_nr_plot = prop.obtain_pulse()
    dt_RWA = tac - tic
    ham_nr = localTools.Hamiltonian_NR(4)
    prop.Hamiltonian = ham_nr
    controls = []
    for k,v in prop.pulse_options.items():
        controls.append(prop.pulse_options[k]['args'])
    #plot_controls_sg(prop.tlist_long,controls)
    period_0 = np.pi/sum(localTools.freqX)
    t_step = int(T / period_0 * 20) + 1
    tlist = localTools.half_step_tlist([0,T,t_step])
    rwa_controls = [RWA_control(i,controls[i]) for i in range(len(controls))]
    pulse_options = {}
    for i in range(4):
        pulse_options[ham_nr[i+1][1]] = {'args':rwa_controls[i]}
    prop.pulse_options = pulse_options
    prop.tlist_long = tlist
    tic = time.time()
    psi_T_1 = prop.propagate()
    tac = time.time()
    #c_rwa_plot = prop.obtain_pulse()
    #plot_controls_sg(prop.tlist_long,rwa_controls)
    #print(psi_T_1)
    dt_NR = tac - tic
    J_ss = J_T_local.JT_ss(psi_T_0,psi_T_1)
    print(f"J_ss: {J_T_local.JT_ss(psi_T_0,psi_T_1)}, dt RWA: {dt_RWA}, dt_NR: {dt_NR}")
    return J_ss,dt_RWA,dt_NR
    #plot_controls(tlist_NR,tlist,c_nr_plot,c_rwa_plot)

params = [0,0,0]
Tlist = [30.0,35.0,40.03,45.0,50.03]
folders_main = [f'opt_examples/{T}/' for T in Tlist]
folders_main[2] = 'opt_examples/40.03_local_minima/'

for T,folder in zip(Tlist,folders_main):
    for subfolder in ['stage_X/','stage_XN/']:
        J_ss,dt_RWA,dt_NR = compare(T,folder+subfolder)
        params[0] += J_ss
        params[1] += dt_RWA
        params[2] += dt_NR

n_tests = len(Tlist) * 2

# Test with control sequence obtained by optimization with J_XN.
print(f"Number of tests: {n_tests}\nAverage infidelity: {params[0]/n_tests} \nAverage time without RWA: {params[1]/n_tests} \nAverage time with RWA: {params[2]/n_tests}")
        
n_test = 10
T = 50.0
params = [0,0,0]
for _ in range(n_tests):
    J_ss,dt_RWA,dt_NR = compare(T,1)
    params[0] += J_ss
    params[1] += dt_RWA
    params[2] += dt_NR

# Test with randomly generated control sequence
print(f"Number of tests: {n_tests}\nAverage infidelity: {params[0]/n_tests} \nAverage time without RWA: {params[1]/n_tests} \nAverage time with RWA: {params[2]/n_tests}")