import krotov,numpy as np,J_T_local,localTools,qutip,config_job
import propagation_API

def S(t,T):
    """Scales the Krotov methods update of the pulse value at the time t"""
    return krotov.shapes.flattop(t, t_start=0.0, t_stop=T, t_rise=1, func='blackman')
def random_guess(t,control_args):
    fit_func=control_args.get('fit_func')
    return fit_func(t)
def random_guess_cos(t,control_args):
    fit_func=control_args.get('fit_func')
    freqX=control_args.get('freqX')
    return fit_func(t)*np.cos(freqX*t)*2

def Krotov_call_0(tlist):
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    prop = config_job.Propagation(H,tlist,'cheby',initial_states,'pulse_initial')
    shapes = [S]*num_qubit
    oct = config_job.Optimization(prop,'krotov',shapes)

def Krotov_call(num_qubit,T,canoLabel,JT,control_source = None, header = None):
    from functools import partial
    lambda_a=10
    H=localTools.Hamiltonian(num_qubit)
    shapes = [S]*num_qubit
    num_steps = 801
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    target_states = [qutip.Qobj(localTools.rotate_state(localTools.canoGHZGen(num_qubit,canoLabel), num_qubit, 1, T))]
    tlist=np.linspace(0,T,num_steps)
    #tlist = tlist[:3]
    pulse_options={}
    if not control_source:control_source = 5
    control_args = localTools.control_generator(num_qubit,control_source,T,header)
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(lambda_a=lambda_a,update_shape=partial(shapes[i-1],T=T),args=control_args[i-1])
    objectives=[krotov.Objective(initial_state=initial_states[i],target=target_states[i],H=H) for i in range(len(initial_states))]
    if JT == 0:functional_name = 'inFidelity'
    elif JT == 1:functional_name = 'XN'
    opt_functionals=J_T_local.functional_master(functional_name)
    #propagator=krotov.propagators.expm
    propagator=propagation_API.KROTOV_CHEBY
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
            iter_stop=100,store_all_pulses=True)
    return opt_result

num_qubit = 4
T=21.0
#H=localTools.Hamiltonian(num_qubit)
#print(H)
opt_result=krotov_call(num_qubit,T,'0+',0,'control_source/21.0/','pulse_initial')
#opt_result=store_intermediate_state(num_qubit,T,'0+',0,'control_source/21.0/','pulse_oct')

