import os,numpy as np,qutip,matplotlib.pyplot as plt,types,title_routine,read_write
from scipy.sparse import coo_matrix,coo_array

def matrix_verifier(ipt_matrix):
    if isinstance(ipt_matrix,qutip.Qobj):ipt_matrix = ipt_matrix.full()
    if not isinstance(ipt_matrix,np.ndarray):ipt_matrix = np.array(ipt_matrix)
    return ipt_matrix

def const_pulse(t):
    return 1
E0 = 2
def sin2(T,t):
    return 0.5*E0*np.sin(np.pi*t/T)**2

class Pulse_oct_para:
    def __init__(self,info_dict):
        self.oct_shape = info_dict['oct_shape']
        self.oct_increase_factor = info_dict['oct_increase_factor']
        self.oct_pulse_min = info_dict['oct_pulse_min']
        self.oct_pulse_max = info_dict['oct_pulse_max']
        self.is_complex = info_dict['is_complex']
        self.t_rise = info_dict['t_rise']
        self.t_fall = info_dict['t_fall']
        self.oct_lambda_a = float()

class Pulse:
    def __init__(self,*args):
        if len(args) == 2:
            self.tlist = args[0]
            self.amp = args[1]
            if isinstance(self.amp,types.FunctionType):
                self.amp = self.amp(self.tlist)
        elif  len(args) == 1:
            info_dict = args[0]
            self.pulse_id = int(info_dict['id'])
            self.runfolder = info_dict['runfolder']
            self.filename = info_dict['filename']
            self.tlist,self.amp = read_write.controlReader(self.runfolder + self.filename)
            self.oct_shape = info_dict['oct_shape']
            if self.oct_shape != 'zero':
                self.pulse_oct_para = Pulse_oct_para(info_dict)
            else:self.pulse_oct_para={}

    def add_oct_paramters(self,pulse_id,filename,info_dict):
        self.pulse_id = pulse_id
        self.filename = filename
        self.oct_shape = info_dict['oct_shape']
        if self.oct_shape != 'zero':
            self.pulse_oct_para = Pulse_oct_para(info_dict)
        else:self.pulse_oct_para={}

    def store2file(self,runfolder):
        text_content = read_write.control2text(self.tlist,self.amp)
        with open(runfolder + self.filename, 'w') as store_f:store_f.write(text_content)
    
    def report(self):
        text_content = f'* pulse_id = {self.pulse_id}, filename = {self.filename}'
        if isinstance(self.pulse_oct_para,dict): return text_content + '\n'
        else:
            text_content += ', '
            for k, v in self.pulse_oct_para.__dict__.items():
                text_content += f'{k} = {v}, '
            text_content = text_content[:-2] + '\n'
        return text_content

class State:
    def __init__(self,*args):
        if len(args) == 3:
            state_mat = read_write.stateReader(args[0] + args[1], args[2])
            self.filename = args[1]
        else:state_mat = args[0]
        if not isinstance(state_mat,coo_array):state_mat = coo_array(state_mat)
        self.data = state_mat.data
        self.row = state_mat.row
        self.nnz = len(self.row)
        self.shape = state_mat.shape
        self._len_max_row = len(str(self.shape[0]))
        #self._norm = state_norm(self._state_mat)
    def add_label(self,label):
        self.label = label

    def add_filename(self,filename):
        self.filename = filename

    def store2file(self,runfolder):
        with open(runfolder + self.filename,'w') as store_f:
            store_f.write(self.text_format())

    def report(self):
        print(self.text_format())

    def report_2(self):
        print(np.abs(self._state_mat[0][0])**2)



class Ham:
    def __init__(self,*args):
        self.terms = []
        if len(args) == 1:
            self.max_dim = 0
            args = args[0]
            for i in range(len(args)):
                if isinstance(args[i],np.ndarray):mat_dim = len(args[i])
                else:mat_dim = len(args[i][0])
                if mat_dim > self.max_dim:self.max_dim = mat_dim
                self.terms.append(args[i])
            self.resolve_pulses()
        else:
            self.ham_info = args[0]
            self.pulse_info = args[1]
            self.max_dim = args[2]
            self.runfolder = args[3]
            for i in range(len(self.ham_info)):
                H_mat = read_write.matrixReader(self.runfolder + self.ham_info[i]['filename'],self.max_dim)
                if "pulse_id" in self.ham_info[i].keys():
                    pulse_id = int(self.ham_info[i]['pulse_id'])
                    for j in range(len(self.pulse_info)):
                        if int(self.pulse_info[j]['id']) == pulse_id:
                            self.terms.append([H_mat,Pulse(self.pulse_info[j])])
                            break
                else:self.terms.append(H_mat)

    def add_tlist(self,qspoc_tlist):
        self.qspoc_tlist = qspoc_tlist

    def resolve_pulses(self):
        for i in range(len(self.terms)):
            if isinstance(self.terms[i],list) and len(self.terms[i]) == 3:
                self.terms[i] = [self.terms[i][0],Pulse(self.terms[i][1],self.terms[i][2])]
    
    def add_pulse_info(self,pulse_info):
        if len(pulse_info) != len(self.terms):
            raise IndexError(f'Length of input variable pulse_info ({len(pulse_info)}) does not equal length of {type(self).__name__}.terms ({len(self.terms)}).\nUse Placeholder (False) to fix this.')
        n_pulses = 0
        for i in range(len(pulse_info)):
            if not pulse_info[i]:pulse_info[i] = {'oct_shape':'zero'}
            if len(self.terms[i]) == 2:
                n_pulses += 1
                self.terms[i][1].add_oct_paramters(n_pulses,f'pulses_{i}.dat',pulse_info[i])
        return 1

    def pulse_report(self):
        text_content = 'pulse_sec:\n'
        for term in self.terms:
            if isinstance(term,list):
                text_content += term[1].report()
        return text_content

    def report(self):
        text_content = f'{self.pulse_report()}\nham_sec: max_dim = {self.max_dim}\n'
        for i in range(len(self.terms)):
            text_content += f'* filename = H{i}.dat'
            if isinstance(self.terms[i],list):
                text_content += f', pulse_id: {self.terms[i][1].pulse_id}\n'
            else:text_content += '\n'
        return text_content + '\n'

    def store2file(self,runfolder):
        for i in range(len(self.terms)):
            if not isinstance(self.terms[i],list):
                text_content = read_write.matrix2text(self.terms[i])
                with open(runfolder + f'H{i}.dat','w') as matrix_file:
                    matrix_file.write(text_content)
                #read_write.matrix2text(self.terms[i],runfolder + f'H{i}.dat')
            else:
                text_content = read_write.matrix2text(self.terms[i][0])
                with open(runfolder + f'H{i}.dat','w') as matrix_file:
                    matrix_file.write(text_content)
                #read_write.matrix2text(self.terms[i][0],runfolder + f'H{i}.dat')
                self.terms[i][1].store2file(runfolder)

class Hamiltonian:
    def __init__(self,*args):
        self._matrices = []
        self._pulses = []
        self._controllabel = []
        self._highest_eigenvalue = []
        self._lowest_eigenvalue = []
        if not args:
            raise ValueError("At least one Hermitian matrix is required.")
        for arg in args:
            if isinstance(arg,tuple) and len(arg) == 2:
                self._matrices.append(matrix_verifier(arg[0]))
                self._pulses.append(arg[1])
                self._controllabel.append(len(self._pulses) - 1)
            elif any([isinstance(arg,data_type) for data_type in [qutip.Qobj,np.ndarray,list]]):
                self._matrices.append(matrix_verifier(arg))
                self._pulses.append(const_pulse)
            else:
                raise KeyError("Hamiltonian: Number or type of keys does not match.")
        self._n_controls = len(self._controllabel)
        self._space_dimension = len(self._matrices[0])
        if self._space_dimension > 1024:
            for matrix in self._matrices:
                self._highest_eigenvalue.append(Arnoldi.Lanczos(matrix,1))
        else:
            for matrix in self._matrices:
                eig_val,eig_vec = np.linalg.eig(matrix)
                self._highest_eigenvalue.append(max(eig_val))
                self._lowest_eigenvalue.append(min(eig_val))
    
    def report(self):
        print(f'Dimension of the Hilbert Space: {self._space_dimension}')
        if self._space_dimension > 64: est_method = '(estimated by Lanczos iteration)'
        else: est_method = ''
        for obj_id,matrix,pulse,highest_eig,lowest_eig in zip(range(len(self._matrices)),self._matrices,self._pulses,self._highest_eigenvalue,self._lowest_eigenvalue):
            print(f'Object {obj_id}: ')
            if pulse == None:print(f'Static term, matrix: \n{matrix}')
            else:print(f'Time dependent term, matrix: \n{matrix}')
            print(f'Highest eigenvalue{est_method}: {highest_eig}')
            print(f'Lowest eigenvalue{est_method}: {lowest_eig}')
    
    def draw_controls(self,tlist,title):
        for i in self._controllabel:
            plt.plot(tlist,self._pulses[i](tlist),label = f'control_{i}')
        plt.legend(loc='best')
        plt.title(title)
        plt.savefig(title+'.png')
        plt.clf()

    def energy_est(self,amps):
        highest_eig = sum(self._highest_eigenvalue[i]*amps[i] for i in range(len(amps)))
        lowest_eig = sum(self._lowest_eigenvalue[i]*amps[i] for i in range(len(amps)))
        return np.real(highest_eig),np.real(lowest_eig)

    def at_t(self,t):
        H_t = np.zeros([self._space_dimension,self._space_dimension],dtype = np.complex128)
        highest_eig_t = 0
        lowest_eig_t = 0
        for matrix,pulse,eig_val_h, eig_val_l in zip(self._matrices,self._pulses,self._highest_eigenvalue,self._lowest_eigenvalue):
            H_t += matrix * pulse(t)
        est_amps = [1] * len(self._matrices)
        for i in range(len(self._matrices)):
            if i in self._controllabel:
                est_amps[i] = self._pulses[i](t)
        highest_eig_t, lowest_eig_t = self.energy_est(est_amps)
        return csc_matrix(H_t), highest_eig_t, lowest_eig_t
    
    def at_amps(self,t,amps):
        H_t = np.zeros([self._space_dimension,self._space_dimension],dtype = np.complex128)
        highest_eig_t = 0
        lowest_eig_t = 0
        est_amps = [1] * len(self._matrices)
        amp_i = 0
        for matrix,pulse_id in zip(self._matrices,range(len(self._matrices))):
            if pulse_id in self._controllabel: 
                H_t += matrix * amps[amp_i]
                est_amps[amp_i] = amps[amp_i]
                amp_i += 1
            else: 
                H_t += matrix * self._pulses[pulse_id](t)
                est_amps[pulse_id] = self._pulses[pulse_id](t)
        highest_eig_t, lowest_eig_t = self.energy_est(est_amps)
        return csc_matrix(H_t), highest_eig_t, lowest_eig_t

    def dev_t(self,t,pulse_id):
        h = 20/3000
        h2 = h * 2
        return (-self._pulses[pulse_id](t + h2) + 8*self._pulses[pulse_id](t + h) - 8*self._pulses[pulse_id](t 
                        - h) + self._pulses[pulse_id](t - h2)) / (12 * h)

    def store2file(self,file_address):
        if not os.path.exists(file_address): os.mkdir(file_address)
        for i in range(len(self._matrices)):
            text_content = read_write.matrix2text(self._matrices[i])
            with open(file_address + f'H{i}.dat','w') as store_f:
                store_f.write(text_content)

    def update_control(self,new_pulses):
        for i,new_pulse in zip(self._controllabel,new_pulses):
            self._pulses[i] = new_pulse

class Lindbladian:
    def __init__(self,H,*args):
        self._H = H
        self._dissipators = []
        self._damping_rates = []
        if not args:
            raise ValueError("At least one dissipator is required.")
        for arg in args:
            if isinstance(arg,tuple) and len(arg) == 2:
                self._dissipators.append(matrix_verifier(arg[0]))
                self._damping_rates.append(arg[2])
            elif any([isinstance(arg,data_type) for data_type in [qutip.Qobj,np.ndarray,list]]):
                self._dissipators.append(matrix_verifier(arg))
                self._damping_rates.append(1)
            else:
                raise KeyError("Hamiltonian: Number or type of keys does not match.")
    
    def at_t(self,t):
        return 0
    
    def at_amp(self,amps):
        return 0
    
    def update_H(self,H):
        self._H = H

def state_norm(state):
    result = 0
    for i in range(len(state)):
        result += np.conj(state[i]) * state[i]
    return np.abs(result[0])

def vector2text(vector):
    text_content = '# col ' + ' ' * 20 +'real    ' + ' ' * 17 + 'imag\n'
    space_label = ' '
    max_length = len(str(len(vector)))
    for i in range(len(vector)):
        imag_formatted = "{: .16E}".format(np.imag(vector[i][0]))
        #if np.imag(self.state_mat[i])[0] < 0 : imag_formatted = ' ' + imag_formatted
        real_formatted = "{: .16E}".format(np.real(vector[i][0]))
        #if np.real(self.state_mat[i])[0] < 0 : real_formatted = ' ' + real_formatted
        text_content += f'   {space_label * (max_length-len(str(i)))}{i}  {real_formatted}  {imag_formatted}\n'
    return text_content

def isunitary(mat):
    print(np.matmul(np.conj(np.transpose(mat)),mat))

