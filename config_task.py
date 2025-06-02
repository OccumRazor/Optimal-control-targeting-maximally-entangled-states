import read_write,numpy as np,localTools,re,qutip,task_obj
from collections import defaultdict
from scipy.interpolate import interp1d

str2float_keys = ['oct_lambda_a','lambda_a','t_start','t_stop','t_rise','f_fall']


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
            if key in str2float_keys:pulses_info[i][key] = float(config['pulse']['main'][key])
            else:pulses_info[i][key] = config['pulse']['main'][key]
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
    prop_obj = task_obj.Propagation(Hamiltonian,tlist,prop_method,initial_states,'pulse_initial',pulse_options)
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
    opt_obj = task_obj.Optimization(prop_obj,opt_method,JT_conv,delta_JT_conv,iter_dat,iter_stop)
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
        observables.append(observable)
        opt_obj.set_observables(observables)
    return opt_obj,config

#print(parse_config_with_subsections(f'control_source/21.0/config'))
#config_opt('control_source/rf0/')
#config_task(f'control_source/21.0/')