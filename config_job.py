import read_write,numpy as np,localTools,re,qutip,Krotov_API,krotov,J_T_local,os,propagation_method
from functools import partial
from pathlib import Path
from collections import defaultdict
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

str2float_keys = ['oct_lambda_a','lambda_a','t_start','t_stop','t_rise','f_fall']

def H_t(Hamiltonian,pulse_options,t):
    assert isinstance(Hamiltonian,list) and len(Hamiltonian) >0
    if isinstance(Hamiltonian[0],list):
        Ht = Hamiltonian[0][1](t,pulse_options[Hamiltonian[0][1]]['args']) * Hamiltonian[0][0]
    else:
        Ht = Hamiltonian[0]
    for H_i in Hamiltonian[1:]:
        if isinstance(H_i,list):
            Ht += H_i[1](t,pulse_options[H_i[1]]['args']) * H_i[0]
        else:
            Ht += H_i
    return Ht

class Propagation:
    def __init__(self,Hamiltonian,tlist,prop_method,initial_states,pulse_name,pulse_options = None):
        self.Hamiltonian = Hamiltonian
        self.tlist = tlist
        self.tlist_long = localTools.half_step_tlist(self.tlist)
        self.prop_method = prop_method
        if isinstance(initial_states,list):
            self.n_states = len(initial_states)
        else:
            self.n_states = 1
        self.initial_states = initial_states
        self.pulse_name = pulse_name
        self.pulse_options = pulse_options
        self.c_ops = None
    
    def add_dissipator(self,c_ops):
        self.c_ops = c_ops

    def krotov_pulse_options(self):
        converted_options = {}
        for k,v in self.pulse_options.items():
            converted_options[k] = dict(lambda_a=float(v['oct_lambda_a']),update_shape=partial(localTools.S,
                                    t_start=self.tlist[0], t_stop=self.tlist[1], t_rise=float(v['t_rise']), t_fall=float(v['t_fall'])),args=v['args'])
        return converted_options

    def propagate(self,prop_options: dict = {}):
        psi_0 = self.initial_states
        dt = self.tlist_long[1] - self.tlist_long[0]
        for t in self.tlist_long:
            Ht = H_t(self.Hamiltonian,self.pulse_options,t)
            Ht = Ht.full()
            eig_vals = eigsh(Ht,return_eigenvectors=False)
            E_max = max(eig_vals)
            E_min = min(eig_vals)
            for i in range(self.n_states):
                psi_0[i] = qutip.Qobj(propagation_method.Chebyshev(Ht,psi_0[i].full(),E_max,E_min,dt))
        return psi_0

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

    def update_control(self,new_controls):
        for i in range(len(new_controls)):
            cubicSpline_fit = interp1d(
                self.tlist_long, new_controls[i], kind="cubic", fill_value="extrapolate")
            self.pulse_options[self.Hamiltonian[i+1][1]]['args'] = {"fit_func": cubicSpline_fit}

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

    def set_target_states(self,target_states):
        self.target_states = target_states
    
    def set_observables(self,observables):
        self.observables = observables

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
        krotov_pulse_options = self.prop.krotov_pulse_options()
        out_file = open(runfolder+self.oct_info['iter_dat'],'w')
        opt_result=krotov.optimize_pulses(
                objectives,krotov_pulse_options,self.prop.tlist_long,
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

def combine_relevant_lines(sentence_list):
    combined_lines = [sentence_list[0]+'\n']
    max_line = len(sentence_list)
    current_id = 1
    while current_id < max_line - 1:
        if sentence_list[current_id] == '':
            current_id += 1
        else:
            relevant_info = ''
            while sentence_list[current_id] != '':
                relevant_info += sentence_list[current_id] + '\n'
                current_id += 1
            combined_lines.append(relevant_info)
    return combined_lines

def combine_psi_args(sentence_list):
    combined_lines = ''
    psi_start = 0
    psi_end = len(sentence_list) - 1
    while 'psi' not in sentence_list[psi_start]:
        psi_start += 1
    while 'psi' not in sentence_list[psi_end]:
        psi_end -= 1
    for i in range(psi_start):
        combined_lines += sentence_list[i] + '\n'
    psi_sentence = ''
    for i in range(psi_start,psi_end + 1):
        psi_sentence += sentence_list[i]
    psi_sentence = ''.join(psi_sentence.split('psi:'))
    psi_sentence = ''.join(psi_sentence.split('\n'))
    psi_sentence = 'psi:' + '\n*'.join(psi_sentence.split('*')) + '\n'
    combined_lines += psi_sentence + '\n'
    for i in range(psi_end + 1,len(sentence_list)):
        combined_lines += sentence_list[i] + '\n'
    return combined_lines

def read_config(file_name):
    config_f = open(file_name,'r')
    config_f = config_f.read().split('\n')
    config_f = combine_relevant_lines(config_f)
    n_psi = ''.join(config_f).count('psi:')
    if n_psi>1:
        config_f = combine_psi_args(config_f)
        with open(file_name,'w') as store_f:
            store_f.write(config_f)
    return 0

def parse_config_with_subsections(file_path):
    config = defaultdict(lambda: {"main": [], "subsections": []})
    current_section = None
    current_subsection = None
    read_config(file_path)
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            # Check if the line is a new section (e.g., "tgrid:", "prop:")
            section_match = re.match(r'^(\w+):', line)
            if section_match:
                current_section = section_match.group(1)
                config[current_section] = {"main": {}, "subsections": []}
                current_subsection = None
                line = line[len(current_section)+1:].strip()  # Remove section name
            # Check if the line is a continuation (using "&")
            if '&' in line:
                line = line.replace('&', '').strip()
            # If the line starts with '*', it's a new subsection
            if line.startswith('*'):
                current_subsection = {}
                config[current_section]["subsections"].append(current_subsection)
                line = line[1:].strip()  # Remove '*' from the start
            # Split key-value pairs separated by commas
            pairs = [pair.strip() for pair in line.split(',')]
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key, value = key.strip(), value.strip()
                    # If in a subsection, add to subsection; else add to main section
                    if current_subsection is not None:
                        current_subsection[key] = value
                    else:
                        config[current_section]["main"].update({key: value})
    return config



def config_prop(path,zero_base = True):
    config = parse_config_with_subsections(f'{path}config')
    prop_method = config['prop']['main']['prop_method']
    tlist = [float(config['tgrid']['main']['t_start']),float(config['tgrid']['main']['t_stop']),int(config['tgrid']['main']['nt'])]
    pulses_info = [config['pulse']['subsections'][i] for i in range(len(config['pulse']['subsections']))]
    for i in range(len(pulses_info)):
        for key in config['pulse']['main'].keys():
            pulses_info[i][key] = config['pulse']['main'][key]
        detupleTlist, detupleGuess = read_write.controlReader(
                path + pulses_info[i]['filename'])
        cubicSpline_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate")
        pulses_info[i]['args'] = {"fit_func": cubicSpline_fit}
    Hamiltonian_info = [config['ham']['subsections'][i] for i in range(len(config['ham']['subsections']))]
    Ham_pulse_Table = []
    for i in range(len(Hamiltonian_info)):
        if 'pulse_id' in Hamiltonian_info[i].keys():
            Ham_pulse_Table.append(Hamiltonian_info[i]['pulse_id'])
        else:
            Ham_pulse_Table.append(False)
        for key in config['ham']['main'].keys():
            Hamiltonian_info[i][key] = config['ham']['main'][key]
    Hamiltonian = [[qutip.Qobj(read_write.matrixReader(path+Hamiltonian_info[i]['filename'],int(Hamiltonian_info[i]['dim']),zero_base)),lambda t,args:localTools.random_guess(t,args)
                    ] if 'pulse_id' in Hamiltonian_info[i].keys() else
                    qutip.Qobj(read_write.matrixReader(path+Hamiltonian_info[i]['filename'],int(Hamiltonian_info[i]['dim']),zero_base)
                    )for i in range(len(Hamiltonian_info))]
    pulse_options = {}
    for i in range(len(pulses_info)):
        pulse_id = pulses_info[i]['pulse_id']
        pulses_info[i].pop('pulse_id')
        pulses_info[i].pop('filename')
        pulse_options[Hamiltonian[Ham_pulse_Table.index(pulse_id)][1]] = pulses_info[i]
    initial_states = []
    for i in range(len(config['psi']['subsections'])):
        if config['psi']['subsections'][i]['label'] == 'initial':
            state = read_write.stateReader(path+config['psi']['subsections'][i]['filename'],int(config['ham']['main']['dim']),zero_base)
            initial_states.append(qutip.Qobj(state))
    prop_obj = Propagation(Hamiltonian,tlist,prop_method,initial_states,'pulse_initial',pulse_options)
    prop_obj.tlist_long = detupleTlist
    return prop_obj,config

def config_opt(path,zero_base = True):
    prop_obj,config = config_prop(path,zero_base)
    opt_info = config['oct']['main']
    opt_method = opt_info['oct_method']
    JT_conv = float(opt_info['JT_conv'])
    delta_JT_conv = float(opt_info['delta_JT_conv'])
    iter_dat = opt_info['iter_dat']
    iter_stop = int(opt_info['iter_stop'])
    opt_obj = Optimization(prop_obj,opt_method,JT_conv,delta_JT_conv,iter_dat,iter_stop)
    target_states = []
    for i in range(len(config['psi']['subsections'])):
        if config['psi']['subsections'][i]['label'] == 'final':
            state = read_write.stateReader(path+config['psi']['subsections'][i]['filename'],int(config['ham']['main']['dim']),zero_base)
            target_states.append(qutip.Qobj(state))
    if len(target_states):
        opt_obj.set_target_states(target_states)
    if 'observables' in config.keys():
        observables = []
        for i in range(len(config['observables']['subsections'])):
            observable = read_write.matrixReader(path+config['observables']['subsections'][i]['filename'],int(config['ham']['main']['dim']),zero_base)
        observables.append(qutip.Qobj(observable))
        opt_obj.set_observables(observables)
    return opt_obj,config

#print(parse_config_with_subsections(f'control_source/21.0/config'))
#config_opt('control_source/rf0/')
#config_task(f'control_source/21.0/')