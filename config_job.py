import read_write,numpy as np

class Propagation:
    def __init__(self,Hamiltonian,tlist,prop_method,initial_states,pulse_name):
        self.Hamiltonian = Hamiltonian
        self.tlist = tlist
        self.prop_method = prop_method
        if isinstance(initial_states,list):
            self.n_states = len(initial_states)
        else:
            self.n_states = 1
        self.initial_states = initial_states
        self.pulse_name = pulse_name
        return 0
    
    def config(self,path,write = False):
        tlist_long = np.linspace(self.tlist[0],self.tlist[1],self.tlist[2])
        config_dict = {'prop':{'prop_method':self.prop_method},'tgrid':{'t_start':self.tlist[0],'t_stop':self.tlist[1],'nt':self.tlist[2]}}
        if isinstance(self.Hamiltonian[0],list):
            Hamiltonian_dict = {'dim':len(self.Hamiltonian[0][0])}
        else:
            Hamiltonian_dict = {'dim':len(self.Hamiltonian[0])}
        pulse_dict = {}
        pulse_count = 0
        for i in range(len(self.Hamiltonian)):
            if isinstance(self.Hamiltonian[i],list):
                mat_text = read_write.matrix2text(self.Hamiltonian[i][0])
                pulse_text = read_write.control2text(tlist_long,self.Hamiltonian[i][1](tlist_long))
                Hamiltonian_text += f'* filename = H{i}.dat, pulse_id = {pulse_count}{
                                    ',\n' if i < len(self.Hamiltonian) -1 else '\n\n'}'
                with open(path + f'{self.pulse_name}_{pulse_count}.dat','w') as pulse_file:
                    pulse_file.write(pulse_text)
                pulse_count += 1
            else:
                mat_text = read_write.matrix2text(self.Hamiltonian[i])
            with open(path + f'H_{i}.dat', 'w') as mat_file:
                mat_file.write(mat_text)
        psi_dict = {}
        if self.n_states == 1:
            state_text = read_write.state2text(self.initial_states[0])
            with open(path + 'psi_initial.dat','w') as state_file:
                state_file.write(state_text)
        else:
            for i in range(self.n_states):
                state_text = read_write.state2text(self.initial_states[i])
                with open(path + f'psi_initial_{i}.dat','w') as state_file:
                    state_file.write(state_text)
        config_dict['ham'] = Hamiltonian_dict
        config_dict['pulse'] = pulse_dict
        config_dict['pdi'] = psi_dict
        if write:
            with open(path + 'config', 'w') as config_file:
                config_file.write(config_text)
        return config_text

class Optimization:
    def __init__(self,prop,opt_method,shape_function):
        self.prop = prop
        self.opt_method = opt_method
        self.psi_final = None
        self.shape_function = shape_function
        return 0

    def config(self,path):
        config_dict = self.prop.config(self.prop,path,False)
        config_dict['oct'] = {'oct_method':self.opt_method}
        if self.psi_final:
            if isinstance(self.psi_final,list):
                for i in range(len(self.psi_final)):
                    psi_text = read_write.state2text(self.psi_final[i])
                    with open(path+f'psi_final_{i}.dat', 'w') as psi_file:
                        psi_file.write(psi_text)
            else: 
                psi_text = read_write.state2text(self.psi_final)
                with open(path+f'psi_final.dat', 'w') as psi_file:
                    psi_file.write(psi_text)
        with open(path + 'config', 'w') as config_file:
            config_file.write(config_text)
        return 0


def read_config(path):
    config_info = {}
    return config_info