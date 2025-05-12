import read_write,numpy as np,localTools,re,qutip,Krotov_API,krotov,J_T_local,propagation_method,copy,matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigsh

def sort_property(ipt_list):
    text_content = ''
    n_dicts = len(ipt_list)
    if n_dicts == 1:
        return '\n* ' + ', '.join(f'{k} = {v}' for k,v in ipt_list[0].items()) + '\n\n'
    id_keys = [list(ipt_list[i].keys()) for i in range(n_dicts)]
    all_keys = list(ipt_list[0].keys())
    for i in range(1,n_dicts):all_keys += list(ipt_list[1].keys())
    all_keys = list(np.unique(all_keys))
    same_property = {}
    different_property = [{'ipt_list_id':i} for i in range(n_dicts)]
    for key in all_keys:
        if all([key in id_keys[i] for i in range(n_dicts)]):
            if all([ipt_list[0][key] == ipt_list[i][key] for i in range(n_dicts)]):
                same_property[key] = ipt_list[0][key]
            else:
                for i in range(n_dicts):
                    if key in id_keys[i]:
                        different_property[i][key] = ipt_list[i][key]
        else:
            for i in range(n_dicts):
                if key in id_keys[i]:
                    different_property[i][key] = ipt_list[i][key]
    text_content += ', '.join(f'{k} = {v}' for k,v in same_property.items()) + '\n'
    for i in range(n_dicts):
        different_property[i].pop('ipt_list_id')
        text_content += '* ' + ', '.join(f'{k} = {v}' for k,v in different_property[i].items()) + '\n'
    return text_content + '\n'

def dict2string(ipt_dict):
    text_content = ''
    for key in ipt_dict.keys():
        text_content += f'{key}: '
        if isinstance(ipt_dict[key],dict): text_content += ', '.join(f'{k} = {v}' for k,v in ipt_dict[key].items()) + '\n\n'
        elif isinstance(ipt_dict[key],list):
            text_content += sort_property(ipt_dict[key])
        else:
            raise KeyError('Anyway the format of the input dictionary does not match.')
    return text_content

def H_t(Hamiltonian,t,pulse_options,update_table=None):
    assert isinstance(Hamiltonian,list) and len(Hamiltonian) > 0
    if isinstance(Hamiltonian[0],list):
        Ht = Hamiltonian[0][1](t,pulse_options[Hamiltonian[0][1]]['args']) * Hamiltonian[0][0]
    else:
        Ht = Hamiltonian[0]
    for H_i in Hamiltonian[1:]:
        if isinstance(H_i,list):
            if not update_table: 
                update_amp = 0
            else:
                update_amp = update_table[H_i[1]]
            Ht += (H_i[1](t,pulse_options[H_i[1]]['args']) + update_amp) * H_i[0]
        else:
            Ht += H_i
    return Ht

class Propagation:
    def __init__(self,Hamiltonian,tlist,prop_method,initial_states,pulse_name,pulse_options = None):
        self.Hamiltonian = Hamiltonian
        self.tlist = tlist
        self.tlist_long = localTools.half_step_tlist(self.tlist)
        self.prop_method = prop_method
        self.n_states = len(initial_states)
        self.initial_states = initial_states
        self.pulse_name = pulse_name
        self.pulse_options = pulse_options
        self.c_ops = None
        self.krotov_pulse_options()
    
    def add_dissipator(self,c_ops):
        self.c_ops = c_ops

    def krotov_pulse_options(self):
        converted_options = {}
        if len(self.tlist) == 2:
            for k,v in self.pulse_options.items():
                converted_options[k] = dict(lambda_a=float(v['oct_lambda_a']),update_shape=partial(localTools.S,
                                        t_start=0, t_stop=self.tlist[0], t_rise=float(v['t_rise']), t_fall=float(v['t_fall'])),args=v['args'])
                self.pulse_options[k]['update_shape']=converted_options[k]['update_shape']
        if len(self.tlist) == 3:
            for k,v in self.pulse_options.items():
                converted_options[k] = dict(lambda_a=float(v['oct_lambda_a']),update_shape=partial(localTools.S,
                                        t_start=self.tlist[0], t_stop=self.tlist[1], t_rise=float(v['t_rise']), t_fall=float(v['t_fall'])),args=v['args'])
                self.pulse_options[k]['update_shape']=converted_options[k]['update_shape']
        self.Krotov_pulse_ops =  converted_options

    def propagate_sg(self,dt,t,psi_0,backwards=False):
        psi_0 = copy.deepcopy(psi_0)
        dt = self.tlist_long[1] - self.tlist_long[0]
        Ht = H_t(self.Hamiltonian,t,self.pulse_options)
        Ht = Ht.full()
        eig_vals = eigsh(Ht,return_eigenvectors=False)
        E_max = max(eig_vals)
        E_min = min(eig_vals)
        for i in range(self.n_states):
            if isinstance(psi_0[i],qutip.Qobj):psi_0[i]=psi_0[i].full()
            psi_0[i] = qutip.Qobj(propagation_method.Chebyshev(Ht,psi_0[i],E_max,E_min,dt,backwards=backwards))
        return psi_0

    def propagate_sg_update(self,dt,t,psi_0,chis):
        psi_0 = copy.deepcopy(psi_0)
        update_table = {}
        dt = self.tlist_long[1] - self.tlist_long[0]
        for i in range(self.n_states):
            psi_0[i] = psi_0[i].full()
        state_size = psi_0[0].size
        for i in range(self.n_states):
            chis[i] = np.reshape(chis[i].full(),(state_size))
        update_return = []
        ga_return = []
        for k in range(len(self.Hamiltonian)):
            if isinstance(self.Hamiltonian[k],list):
                control_k_update_amp = 0
                Hk = self.Hamiltonian[k][0].full()
                pulse_k = self.Hamiltonian[k][1]
                update_shape_k_t = self.Krotov_pulse_ops[pulse_k]['update_shape'](t)
                lambda_a_k = self.Krotov_pulse_ops[pulse_k]['lambda_a']
                if lambda_a_k and update_shape_k_t != 0:
                    for i in range(self.n_states):
                        Hk_psi = np.matmul(Hk,psi_0[i])
                        control_k_update_amp += np.linalg.norm(chis[i],2) * np.imag(np.inner(np.conjugate(chis[i]),np.reshape(Hk_psi,(state_size))))
                control_k_update_amp *= update_shape_k_t / lambda_a_k
                update_table[pulse_k] = control_k_update_amp
                ga_return.append(control_k_update_amp * dt)
                update_return.append(control_k_update_amp+pulse_k(t,self.Krotov_pulse_ops[pulse_k]['args']))
        Ht = H_t(self.Hamiltonian,t,self.pulse_options,update_table)
        Ht = Ht.full()
        eig_vals = eigsh(Ht,return_eigenvectors=False)
        E_max = max(eig_vals)
        E_min = min(eig_vals)
        for i in range(self.n_states):
            psi_0[i] = qutip.Qobj(propagation_method.Chebyshev(Ht,psi_0[i],E_max,E_min,dt))
        return psi_0,update_return,ga_return

    def propagate(self,backwards=False,store_states = False,update=False,chis_t=None,prop_options: dict = {}):
        if 'initial_states' in prop_options.keys():psi_0 = copy.deepcopy(prop_options['initial_states'])
        else:psi_0 = copy.deepcopy(self.initial_states)
        dt = self.tlist_long[1] - self.tlist_long[0]
        prop_tlist = copy.deepcopy(self.tlist_long)
        if backwards:prop_tlist = np.flip(prop_tlist,0)
        if store_states:psi_t = [psi_0]
        if update:
            new_controls = [[] for _ in range(len(self.pulse_options))]
            ga_int = [0 for _ in range(len(self.pulse_options))]
        for i in range(len(prop_tlist)):
            t = prop_tlist[i]
            if update:
                psi_0,update_return,ga_return = self.propagate_sg_update(dt,t,psi_0,chis_t[i])
                for k in range(len(new_controls)):
                    new_controls[k].append(update_return[k])
                    ga_int[k] += np.abs(ga_return[k])
            else:psi_0 = self.propagate_sg(dt,t,psi_0,backwards=backwards)
            if store_states:psi_t.append(psi_0)
        if update:return psi_0,new_controls,ga_int
        else:
            if store_states:return psi_t
            else:return psi_0

    def update_control(self,new_controls):
        for i in range(len(new_controls)):
            new_fit = interp1d(
                self.tlist_long, new_controls[i], kind="cubic", fill_value="extrapolate")
            self.pulse_options[self.Hamiltonian[i+1][1]]['args']["fit_func"] = new_fit
        self.krotov_pulse_options()

    def plot_pulses(self):
        for i in range(len(self.Hamiltonian)):
            if isinstance(self.Hamiltonian[i],list):
                plt.plot(self.tlist_long,self.Hamiltonian[i][1](self.tlist_long,self.pulse_options[self.Hamiltonian[i][1]]['args']),label=f'pulse {i}')
        plt.legend(loc='best')
        plt.show()

    def config(self,path,write = False):
        path_Path = Path(path)
        path_Path.mkdir(exist_ok=True,parents=True)
        config_dict = {'prop':{'prop_method':self.prop_method},'tgrid':{'t_start':self.tlist[0],'t_stop':self.tlist[1],'nt':self.tlist[2]}}
        Hamiltonian_info = []
        pulse_info = []
        pulse_count = 0
        for i in range(len(self.Hamiltonian)):
            if isinstance(self.Hamiltonian[i],list):
                id_pulse_option = self.pulse_options[self.Hamiltonian[i][1]]
                mat_text = read_write.matrix2text(self.Hamiltonian[i][0])
                pulse_text = read_write.control2text(self.tlist_long,self.Hamiltonian[i][1](self.tlist_long,id_pulse_option['args']))
                with open(path + f'{self.pulse_name}_{pulse_count}.dat','w') as pulse_file:
                    pulse_file.write(pulse_text)
                Hamiltonian_info.append({'dim':self.Hamiltonian[i][0].shape[0],'filename':f'H{i}.dat','pulse_id':pulse_count})
                id_pulse_info = {'pulse_id':pulse_count,'filename':f'{self.pulse_name}_{pulse_count}.dat'}
                for key in id_pulse_option.keys():
                    if key != 'args':
                        id_pulse_info[key] = id_pulse_option[key]
                pulse_info.append(id_pulse_info)
                pulse_count += 1
            else:
                mat_text = read_write.matrix2text(self.Hamiltonian[i])
                Hamiltonian_info.append({'dim':self.Hamiltonian[i].shape[0],'filename':f'H{i}.dat'})
            with open(path + f'H{i}.dat', 'w') as mat_file:
                mat_file.write(mat_text)
        psi_info = []
        if self.n_states == 1:
            state_text = read_write.state2text(self.initial_states[0])
            with open(path + 'psi_initial.dat','w') as state_file:
                state_file.write(state_text)
            psi_info.append({'filename':'psi_initial.dat','label':'initial'})
        else:
            for i in range(self.n_states):
                state_text = read_write.state2text(self.initial_states[i])
                with open(path + f'psi_initial_{i}.dat','w') as state_file:
                    state_file.write(state_text)
                psi_info.append({'filename':f'psi_initial_{i}.dat','label':'initial'})
        config_dict['ham'] = Hamiltonian_info
        config_dict['pulse'] = pulse_info
        config_dict['psi'] = psi_info
        if write:
            config_text = dict2string(config_dict)
            with open(path + 'config', 'w') as config_file:
                config_file.write(config_text)
        return config_dict

class Optimization:
    def __init__(self,prop: Propagation,opt_method,JT_conv,delta_JT_conv,iter_dat,iter_stop):
        self.prop = prop
        self.oct_info = {
            'oct_method':opt_method,
            'JT_conv':JT_conv,
            'delta_JT_conv':delta_JT_conv,
            'iter_dat':iter_dat,
            'iter_stop':iter_stop}
        self.target_states = None
        self.observables = None
        self.stored_controls = []
        self.initial_controls = []

    def set_target_states(self,target_states):
        self.target_states = target_states
    
    def set_observables(self,observables):
        self.observables = observables

    def store_initial_controls(self):
        initial_controls = []
        for Hi in self.prop.Hamiltonian:
            if isinstance(Hi,list):
                initial_controls.append(
                    Hi[1](self.prop.tlist_long,self.prop.pulse_options[Hi[1]]['args'])
                )
        self.initial_controls.append(initial_controls)

    def update_control(self,new_controls,store_key = False):
        if store_key == 'all':
            self.stored_controls.append(new_controls)
        if store_key == 'last':
            if len(self.stored_controls) < 2:
                self.stored_controls.append(new_controls)
            else:self.stored_controls = [self.stored_controls[-1],new_controls]
        self.prop.update_control(new_controls)

    def config(self,path):
        path_Path = Path(path)
        path_Path.mkdir(exist_ok=True,parents=True)
        config_dict = self.prop.config(path,False)
        if self.target_states:
            psi_info = []
            if self.prop.n_states > 1:
                for i in range(len(self.target_states)):
                    psi_text = read_write.state2text(self.target_states[i])
                    with open(path+f'psi_final_{i}.dat', 'w') as psi_file:
                        psi_file.write(psi_text)
                    psi_info.append({'filename':f'psi_final_{i}.dat','label':'final'})
            else: 
                psi_text = read_write.state2text(self.target_states[0])
                with open(path+f'psi_final.dat', 'w') as psi_file:
                    psi_file.write(psi_text)
                psi_info.append({'filename':f'psi_final.dat','label':'final'})
            config_dict['psi'] += psi_info
        if self.observables:
            observables_dict = []
            for i in range(len(self.observables)):
                observable_text = read_write.matrix2text(self.observables[i])
                with open(path+f'O{i}.dat', 'w') as observable_file:
                    observable_file.write(observable_text)
                observables_dict.append({'filename':f'O{i}.dat'})
            config_dict['observables'] = observables_dict
        config_dict['oct'] = self.oct_info
        config_text = dict2string(config_dict)
        with open(path + 'config', 'w') as config_file:
            config_file.write(config_text)
    
    def write2runfolder(self,runfolder,opt_result):
        for i in range(len(opt_result.optimized_controls)):
            control_text = read_write.control2text(self.prop.tlist_long,opt_result.optimized_controls[i])
            with open(runfolder+f'pulse_oct_{i}.dat','w') as pulse_f:
                pulse_f.write(control_text)
        state_text = read_write.state2text(opt_result.states[-1].full())
        with open(runfolder+'psi_final_after_oct.dat','w') as state_f:
            state_f.write(state_text)

    def Krotov_run(self,runfolder,functional_name):
        path_Path = Path(runfolder)
        path_Path.mkdir(exist_ok=True,parents=True)
        if self.oct_info['oct_method'] != 'krotov':
            raise KeyError(f"Currently, only krotov and crab is supported for the optimization. Input prop_method: {self.oct_info['oct_method']}")
        if self.prop.prop_method == 'cheby':propagator=Krotov_API.KROTOV_CHEBY
        elif self.prop.prop_method == 'expm':propagator=krotov.propagators.expm
        else:raise KeyError(f"Currently, only cheby and expm is supported for the propagation. Input prop_method: {self.prop.prop_method}")
        opt_functionals=J_T_local.functional_master(functional_name)
        objectives=[krotov.Objective(initial_state=self.prop.initial_states[i],target=self.target_states[i],H=self.prop.Hamiltonian) for i in range(self.prop.n_states)]
        self.prop.krotov_pulse_options()
        out_file = open(runfolder+self.oct_info['iter_dat'],'w')
        opt_result=krotov.optimize_pulses(
                objectives,self.prop.Krotov_pulse_ops,self.prop.tlist_long,
                propagator=propagator,
                chi_constructor=opt_functionals[0],
                info_hook=krotov.info_hooks.print_table(
                    J_T=opt_functionals[1],
                    show_g_a_int_per_pulse=True,
                    unicode=False,
                    col_formats=('%d', '%.13e', '%.13e', '%.13e', '%.13e', '%.13e', '%.13e', '%d'),
                    out=out_file),
                check_convergence=krotov.convergence.Or(
                    krotov.convergence.value_below(self.oct_info['JT_conv'], name='J_T'),
                    krotov.convergence.delta_below(self.oct_info['delta_JT_conv']),
                    krotov.convergence.check_monotonic_error),
                iter_stop=self.oct_info['iter_stop'],store_all_pulses=False)
        out_file.close()
        self.write2runfolder(runfolder,opt_result)
        return opt_result


    def Krotov_optimization(self,chi_constructor,JT,XN=None):
        self.store_initial_controls()
        JT_iter = []
        psi_T = self.prop.propagate()
        #print(psi_T)
        if XN:JT_iter.append(J_T_local.JT_var(psi_T,self.observables))
        else:JT_iter.append(JT(psi_T,self.target_states))
        print(f'iter    JT,{' ' * 16}    ga_int')
        print(f'0      {JT_iter[-1]}, 0')
        pulse_options = {}
        for k,v in self.prop.pulse_options.items():
            pulse_options[k] = {'oct_lambda_a':v['oct_lambda_a'],'shape_function':partial(krotov.shapes.flattop,t_start=self.prop.tlist[0], t_stop=self.prop.tlist[1], t_rise=float(v['t_rise']), t_fall=float(v['t_fall']), func='blackman')}
        for iters in range(1,self.oct_info['iter_stop'] + 1):
            if XN:chis_T=J_T_local.chis_var(psi_T,self.observables)
            else:chis_T = chi_constructor(psi_T,self.target_states)
            chis_t = self.prop.propagate(True,True,False,prop_options={'initial_states':chis_T})
            chis_t.reverse()
            psi_T,new_controls,ga_int = self.prop.propagate(False,False,True,chis_t)
            if XN:
                JT_iter.append(J_T_local.JT_var(psi_T,self.observables))
                print(J_T_local.cat_res_2(psi_T[0].full(),self.prop.tlist[1],'0-'))
            else:JT_iter.append(JT(psi_T,self.target_states))
            print(f'{iters}      {JT_iter[-1]}    {ga_int}')
            self.update_control(new_controls)
            if JT_iter[-1] < 1e-3 or (JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3:
                #print(f'stop condition met (JT_iter[-1] < 1e-3: {JT_iter[-1] < 1e-3}, (JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3): {(JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3}, break')
                break
        return JT_iter

    def GRAPE_update_pulse(self,psi_t,lambda_t,Hamiltonian,epsilon,tlist,n_states,pulse_options):
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

    def GRAPE(self,target_states,epsilon,iter_stop,JT):
        self.store_initial_controls()
        JT_iter = []
        n_states = len(target_states)
        print(target_states)
        for i in range(iter_stop):
            psi_t = self.prop.propagate(False,True)
            JT_iter.append(JT(psi_t[-1],target_states))
            if i:print(f'iter {i}: {JT_iter[-1]}, ga_int: {ga}')
            else:print(f'iter {i}: {JT_iter[-1]}')
            lambda_t = self.prop.propagate(True,True,prop_options={'initial_states':target_states})
            new_pulses,ga = self.GRAPE_update_pulse(psi_t,lambda_t,self.prop.Hamiltonian,epsilon,self.prop.tlist_long,n_states,self.prop.pulse_options)
            self.update_control(new_pulses)
            if i > 1:
                if JT_iter[-1] > 1 - 1e-3 or (JT_iter[-1] - JT_iter[-2])/JT_iter[-1] < 1e-3:
                    print(f'stop condition met (JT_iter[-1] > 1 - 1e-3: {JT_iter[-1] > 1 - 1e-3}, (JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3): {(JT_iter[-2] - JT_iter[-1])/JT_iter[-1] < 1e-3}, break')
                    break
        self.prop.plot_pulses()
        psi_T = self.prop.propagate(False,False)
        JT_iter.append(JT(psi_T,target_states))
        print(f'iter {i}: {JT_iter[-1]}, ga_int: {ga}')
        print(psi_T)
        plt.plot(JT_iter)
        plt.show()