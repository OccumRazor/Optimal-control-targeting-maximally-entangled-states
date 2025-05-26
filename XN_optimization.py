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

def Krotov_config_runfolder(canoLabel,tlist,lambda_a,JT_conv,dJT_conv,iter_stop):
    num_qubit = 4
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    pulse_options={}
    control_source = 1
    control_args = localTools.control_generator(num_qubit,control_source,tlist[1])
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(oct_lambda_a = lambda_a, t_rise = 1, t_fall = 1, args=control_args[i-1])
    prop = task_obj.Propagation(H,tlist,'cheby',initial_states,'pulse_initial',pulse_options)
    oct = task_obj.Optimization(prop,'krotov',JT_conv=JT_conv,delta_JT_conv=dJT_conv,iter_dat='oct_iter.dat',iter_stop=iter_stop)
    if canoLabel:
        target_states = [localTools.rotate_state(localTools.canoGHZGen(num_qubit,canoLabel), num_qubit, 1, tlist[1])]
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
T = 45.0
target_functional = 1
JX = 1
while JX > 0.01:
    if target_functional:opt_obj = Krotov_config_runfolder('6+',[0,T,801],25,1e-2,1e-7,300)
    else:opt_obj = Krotov_config_runfolder('6+',[0,T,801],0.1,1e-2,1e-5,50)
    #opt_obj.Krotov_run('states/','X')

    num_qubit = 4
    O1 = J_T_local.mat_X(num_qubit,1)

    O1 = localTools.rotate_matrix(O1,num_qubit,1,T)
    opt_obj.set_observables([O1])
    for i in range(1):
        #last_success_controls = obtain_pulse(opt_obj.prop.Hamiltonian,opt_obj.prop.tlist_long,opt_obj.prop.pulse_options)
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional,monotonic=True)
    JX = JT_iter[-1]
#print(psi_T)
#plt.matshow(np.abs(localTools.densityMatrix(psi_T[0])))
#plt.show()
#plt.plot(JT_iter)
#plt.show()
#opt_obj.prop.plot_pulses()

opt_obj.oct_info['iter_stop'] = 1
opt_obj.prop.change_lambda_a(100,0)
iter_stop = 200
for i in range(iter_stop):
    last_success_controls = obtain_pulse(opt_obj.prop.Hamiltonian,opt_obj.prop.tlist_long,opt_obj.prop.pulse_options)
    O2 = J_T_local.mat_N(4,1,True)
    opt_obj.set_observables([O1,O2])
    JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional)
    #opt_obj.prop.plot_pulses()
    #print(psi_T)
    if i%5 == 0 or i == iter_stop - 1:
        plt.imshow(np.abs(localTools.densityMatrix(psi_T[0])))
        plt.savefig(f'states/{i}.pdf')
    if not monotonic(JT_iter):
        print('JT increased, resume to last step.')
        opt_obj.update_control(last_success_controls)


