import numpy as np,config_job,krotov,matplotlib.pyplot as plt,J_T_local,localTools,qutip
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

def Krotov_optimization(opt_obj:config_job.Optimization):
    JT_iter = []
    psi_T = opt_obj.prop.propagate()
    print(psi_T)
    JT_iter.append(JT_ss(psi_T,opt_obj.target_states))
    print(f'iter    JT,{' ' * 16}    ga_int')
    print(f'0      {JT_iter[-1]}, 0')
    pulse_options = {}
    for k,v in opt_obj.prop.pulse_options.items():
        pulse_options[k] = {'oct_lambda_a':v['oct_lambda_a'],'shape_function':partial(krotov.shapes.flattop,t_start=opt_obj.prop.tlist[0], t_stop=opt_obj.prop.tlist[1], t_rise=float(v['t_rise']), t_fall=float(v['t_fall']), func='blackman')}
    for iters in range(100):
        chis_T = chis_ss(psi_T,opt_obj.target_states)
        chis_t = opt_obj.prop.propagate(True,True,False,prop_options={'initial_states':chis_T})
        chis_t.reverse()
        psi_T,new_controls,ga_int = opt_obj.prop.propagate(False,False,True,chis_t)
        JT_iter.append(JT_ss(psi_T,opt_obj.target_states))
        print(f'{iters}      {JT_iter[-1]}    {ga_int}')
        opt_obj.prop.update_control(new_controls)
        if JT_iter[-1] < 1e-3 or (JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3:
            print(f'stop condition met (JT_iter[-1] < 1e-3: {JT_iter[-1] < 1e-3}, (JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3): {(JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3}, break')
            break
    print(psi_T)
    plt.plot(JT_iter)
    plt.show()
    return 0

def GRAPE_update_pulse(psi_t,lambda_t,Hamiltonian,epsilon,tlist,n_states,pulse_options):
    lambda_t.reverse()
    for i in range(len(lambda_t)):
        for k in range(n_states):
            psi_t[i][k] = localTools.densityMatrix(psi_t[i][k])
            lambda_t[i][k] = localTools.densityMatrix(lambda_t[i][k])
    new_pulses = []
    dt = tlist[1] - tlist[0]
    ga = []
    for i in range(len(Hamiltonian)):
        if isinstance(Hamiltonian[i],list):
            id_pulse = []
            id_ga = 0
            if isinstance(Hamiltonian[i][0],qutip.Qobj):H_i = Hamiltonian[i][0].full()
            else:H_i = Hamiltonian[i][0]
            for j in range(len(tlist)):
                Delta_it = 0
                for k in range(n_states):
                    Delta_it += np.real(-1j * dt * np.trace(np.matmul(lambda_t[j][k],
                                np.matmul(H_i,psi_t[j][k])-np.matmul(psi_t[j][k],H_i))))
                id_pulse.append(Hamiltonian[i][1](tlist[j],pulse_options[Hamiltonian[i][1]]['args'])-pulse_options[Hamiltonian[i][1]]['update_shape'](tlist[j])*epsilon*Delta_it)
                id_ga += np.abs(epsilon*Delta_it)
            new_pulses.append(id_pulse)
            ga.append(id_ga)
    return new_pulses,id_ga

def GRAPE(prop_obj:config_job.Propagation,target_states,epsilon,iter_stop):
    JT_iter = []
    n_states = len(target_states)
    print(target_states)
    for i in range(iter_stop):
        psi_t = prop_obj.propagate(False,True)
        JT_iter.append(JT_tau(psi_t[-1],target_states))
        if i:print(f'iter {i}: {JT_iter[-1]}, ga_int: {ga}')
        else:print(f'iter {i}: {JT_iter[-1]}')
        lambda_t = prop_obj.propagate(True,True,prop_options={'initial_states':target_states})
        new_pulses,ga = GRAPE_update_pulse(psi_t,lambda_t,prop_obj.Hamiltonian,epsilon,prop_obj.tlist_long,n_states,prop_obj.pulse_options)
        prop_obj.update_control(new_pulses)
        if JT_iter[-1] > 1 - 1e-3 or (JT_iter[-1] - JT_iter[-2])/JT_iter[-1] < 1e-3:
            print(f'stop condition met (JT_iter[-1] > 1 - 1e-3: {JT_iter[-1] > 1 - 1e-3}, (JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3): {(JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3}, break')
            break
    prop_obj.plot_pulses()
    psi_T = prop_obj.propagate(False,False)
    JT_iter.append(JT_tau(psi_T,target_states))
    print(f'iter {i}: {JT_iter[-1]}, ga_int: {ga}')
    print(psi_T)
    plt.plot(JT_iter)
    plt.show()

def Krotov_config_runfolder(runfolder,tlist):
    num_qubit = 2
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    pulse_options={}
    control_source = 1
    control_args = localTools.control_generator(num_qubit,control_source,tlist[0])
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(oct_lambda_a = 1, t_rise = 1, t_fall = 1, args=control_args[i-1])
    prop = config_job.Propagation(H,tlist,'cheby',initial_states,'pulse_initial',pulse_options)
    oct = config_job.Optimization(prop,'krotov',JT_conv=0.1,delta_JT_conv=1e-4,iter_dat='oct_iter.dat',iter_stop=2)
    target_states = [qutip.Qobj(localTools.rotate_state(localTools.canoGHZGen(num_qubit,'0+'), num_qubit, 1, tlist[1]))]
    oct.set_target_states(target_states)
    return oct
    #oct.config(runfolder)
    #oct.Krotov_run('inFidelity')

#opt_obj,config = config_job.config_opt('control_source/rf1/')
opt_obj = Krotov_config_runfolder('',[20.0,801])
#GRAPE(opt_obj.prop,opt_obj.target_states,1,100)
Krotov_optimization(opt_obj)