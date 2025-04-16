import krotov,numpy as np,J_T_local,localTools,qutip,config_job,Krotov_API
from functools import partial

def random_guess(t,control_args):
    fit_func=control_args.get('fit_func')
    return fit_func(t)
def random_guess_cos(t,control_args):
    fit_func=control_args.get('fit_func')
    freqX=control_args.get('freqX')
    return fit_func(t)*np.cos(freqX*t)*2
def Krotov_config_runfolder(runfolder,tlist):
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    pulse_options={}
    control_source = 5
    control_args = localTools.control_generator(num_qubit,control_source,tlist[1])
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(oct_lambda_a = 0.1, t_rise = 1, t_fall = 1, args=control_args[i-1])
    prop = config_job.Propagation(H,tlist,'cheby',initial_states,'pulse_initial',pulse_options)
    oct = config_job.Optimization(prop,'krotov',JT_conv=0.1,delta_JT_conv=1e-4,iter_dat='oct_iter.dat',iter_stop=2)
    target_states = [qutip.Qobj(localTools.rotate_state(localTools.canoGHZGen(num_qubit,'6+'), num_qubit, 1, tlist[1]))]
    oct.set_target_states(target_states)
    oct.config(runfolder)
    #oct.Krotov_run('inFidelity')

def Krotov_run(runfolder):
    opt_obj,config = config_job.config_opt(runfolder)
    psi_f = opt_obj.prop.propagate()
    print(J_T_local.inFidelity(psi_f[0],opt_obj.target_states[0]))
    opt_result = opt_obj.Krotov_run(runfolder,'inFidelity')
    opt_obj.prop.update_control(opt_result.optimized_controls)
    psi_f = opt_obj.prop.propagate()
    print(J_T_local.inFidelity(psi_f[0],opt_obj.target_states[0]))

def Krotov_call(num_qubit,T,canoLabel,JT,control_source = None, header = None):
    t_start=0
    t_stop=T
    t_rise=1
    t_fall=1
    lambda_a=10
    H=localTools.Hamiltonian(num_qubit)
    shapes = [localTools.S]*num_qubit
    num_steps = 801
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    target_states = [qutip.Qobj(localTools.rotate_state(localTools.canoGHZGen(num_qubit,canoLabel), num_qubit, 1, T))]
    tlist=np.linspace(0,T,num_steps)
    #tlist = tlist[:3]
    pulse_options={}
    if not control_source:control_source = 5
    control_args = localTools.control_generator(num_qubit,control_source,T,header)
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(lambda_a=lambda_a,update_shape=partial(shapes[i-1],t_start=t_start, t_stop=t_stop, t_rise=t_rise, t_fall=t_fall),args=control_args[i-1])
    objectives=[krotov.Objective(initial_state=initial_states[i],target=target_states[i],H=H) for i in range(len(initial_states))]
    if JT == 0:functional_name = 'inFidelity'
    elif JT == 1:functional_name = 'XN'
    opt_functionals=J_T_local.functional_master(functional_name)
    #propagator=krotov.propagators.expm
    propagator=Krotov_API.KROTOV_CHEBY
    opt_result=krotov.optimize_pulses(
            objectives,pulse_options,tlist,
            propagator=propagator,
            chi_constructor=opt_functionals[0],
            info_hook=krotov.info_hooks.print_table(
                J_T=opt_functionals[1],
                show_g_a_int_per_pulse=True,
                unicode=False),
            check_convergence=krotov.convergence.Or(
                krotov.convergence.value_below(8e-3, name='J_T'),
                krotov.convergence.delta_below(3e-7),
                krotov.convergence.check_monotonic_error),
            iter_stop=0,store_all_pulses=True)
    return opt_result

num_qubit = 4
#T=21.0
#H=localTools.Hamiltonian(num_qubit)
#print(H)
#Krotov_config_runfolder('control_source/rf2/',[0,20,801])
#Krotov_run('control_source/rf2/')
#T = 20.75
#canoLabel = '1+'
#Krotov_call(4,T,canoLabel,0,f'control_source/{T}/','pulse_initial')
#Krotov_call(4,T,canoLabel,0,f'control_source/{T}/','pulse_oct')
opt_obj,config = config_job.config_opt('control_source/20.75/',zero_base = False)
opt_obj.config('control_source/rf111/')
#print(opt_obj.prop.tlist_long)
#print(opt_obj.prop.tlist)
opt_obj.Krotov_run('control_source/rf111/','inFidelity')
#psi_f = opt_obj.prop.propagate()
#print(psi_f)
#print(J_T_local.inFidelity(psi_f[0],opt_obj.target_states[0]))
#opt_result=Krotov_call(num_qubit,T,'0+',0,'control_source/21.0/','pulse_initial')
#opt_result=store_intermediate_state(num_qubit,T,'0+',0,'control_source/21.0/','pulse_oct')

