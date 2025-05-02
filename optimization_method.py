import numpy as np,config_job,krotov,matplotlib.pyplot as plt,J_T_local
from functools import partial
opt_obj,config = config_job.config_opt('control_source/rf2/')

def Krotov_forward():
    psi_f = 0
    return psi_f

def Krotov_backward():
    chis_t = 0
    return chis_t

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
    JT_iter.append(JT_ss(psi_T,opt_obj.target_states))
    print('iter    JT')
    print(f'0      {JT_iter[-1]}')
    pulse_options = {}
    for k,v in opt_obj.prop.pulse_options.items():
        pulse_options[k] = {'oct_lambda_a':v['oct_lambda_a'],'shape_function':partial(krotov.shapes.flattop,t_start=opt_obj.prop.tlist[0], t_stop=opt_obj.prop.tlist[1], t_rise=float(v['t_rise']), t_fall=float(v['t_fall']), func='blackman')}
    for iters in range(1000):
        chis_t = opt_obj.prop.propagate(True,True,False,opt_obj.target_states)
        psi_T,new_controls = opt_obj.prop.propagate(False,False,True,chis_t)
        if JT_ss(psi_T,opt_obj.target_states) < JT_iter[-1]:
            JT_iter.append(JT_ss(psi_T,opt_obj.target_states))
            print(f'{iters}      {JT_iter[-1]}')
            opt_obj.prop.update_control(new_controls)
    return 0

Krotov_optimization(opt_obj)