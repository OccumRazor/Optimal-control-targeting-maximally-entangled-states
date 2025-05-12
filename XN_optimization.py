import numpy as np,config_task,krotov,matplotlib.pyplot as plt,J_T_local,localTools,qutip,task_obj
from functools import partial

def Krotov_forward():
    psi_f = 0
    return psi_f

def Krotov_backward():
    chis_t = 0
    return chis_t

def JT_tau(states,refs):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            val += J_T_local.J_T_taus(states[i],refs[i])
        return val / len(states)
    else:
        return J_T_local.J_T_taus(states,refs)

def chis_ss(states,refs):
    if isinstance(states,list):
        for i in range(len(states)):
            if isinstance(states[i],qutip.Qobj):states[i] = states[i].full()
            if isinstance(refs[i],qutip.Qobj):refs[i] = refs[i].full()
        chis = []
        for i in range(len(states)):
            chis.append(J_T_local.chis_ss(states[i],refs[i]))
        return chis
    else:
        return J_T_local.chis_ss(states,refs)

def JT_ss(states,refs):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            val += J_T_local.J_T_ss(states[i],refs[i])
        return val / len(states)
    else:
        return J_T_local.J_T_ss(states,refs)

def Krotov_config_runfolder(runfolder,tlist):
    num_qubit = 4
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    pulse_options={}
    control_source = 1
    control_args = localTools.control_generator(num_qubit,control_source,tlist[0])
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(oct_lambda_a = 50, t_rise = 1, t_fall = 1, args=control_args[i-1])
    prop = task_obj.Propagation(H,tlist,'cheby',initial_states,'pulse_initial',pulse_options)
    oct = task_obj.Optimization(prop,'krotov',JT_conv=1e-4,delta_JT_conv=1e-6,iter_dat='oct_iter.dat',iter_stop=1)
    target_states = [qutip.Qobj(localTools.rotate_state(localTools.canoGHZGen(num_qubit,'0+'), num_qubit, 1, tlist[1]))]
    oct.set_target_states(target_states)
    return oct
    #oct.config(runfolder)
    #oct.Krotov_run('inFidelity')

def obtain_pulse(Hamiltonian,tlist,pulse_options):
    pulses = []
    for Hi in Hamiltonian:
        if isinstance(Hi,list):
            pulses.append(Hi[1](tlist,pulse_options[Hi[1]]['args']))
    return pulses

def monotonic(JT_iter):
    return all([JT_iter[i+1]<JT_iter[i] for i in range(len(JT_iter)-1)])

#opt_obj = Krotov_config_runfolder('',[20.0,801])
#opt_obj.GRAPE(opt_obj.target_states,1,10,JT_tau)
opt_obj = Krotov_config_runfolder('',[46.0,801])
target_state = localTools.canoGHZGen(4,'0+')
target_state = localTools.rotate_state(target_state,4,1,46.0)
opt_obj.set_target_states([qutip.Qobj(target_state)])
O1 = J_T_local.mat_X(4,1)
for i in range(1000):
    last_success_controls = obtain_pulse(opt_obj.prop.Hamiltonian,opt_obj.prop.tlist_long,opt_obj.prop.pulse_options)
    O2 = J_T_local.mat_N(4,1,True)
    opt_obj.set_observables([O1,O2])
    JT_iter = opt_obj.Krotov_optimization(chis_ss,JT_ss,'XN')
    if not monotonic(JT_iter):
        opt_obj.update_control(last_success_controls)