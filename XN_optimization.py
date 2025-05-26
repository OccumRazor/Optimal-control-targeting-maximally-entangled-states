import numpy as np,csv,config_task,matplotlib.pyplot as plt,J_T_local,localTools,task_obj

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

def best_info(T):
    csv_name = 'data/XN.csv'
    Tlist = []
    cats = []
    with open(csv_name, mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            #Tlist.append(float(line['T']))
            Tlist.append(line['T'])
            cats.append(line['Label'])
    T_id = Tlist.index(str(T))
    return cats[T_id]

def read_O2s(file_name):
    O2_data = np.loadtxt(file_name)
    O2s = []
    n_cols = len(O2_data[0])
    for i in range(len(O2_data)):
        O2 = np.zeros([2 * n_cols,2 * n_cols])
        for j in range(n_cols):
            O2[j][j] = O2_data[i][j]
            O2[2 * n_cols - j - 1][2 * n_cols - j - 1] = O2_data[i][j]
        O2s.append(O2)
    return O2s


def Krotov_config_task(canoLabel,tlist,lambda_a,JT_conv,dJT_conv,iter_stop):
    num_qubit = 4
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    pulse_options={}
    control_source = 1
    control_args = localTools.control_generator(num_qubit,control_source,tlist[1])
    for i in range(1,len(H)):
        pulse_options[H[i][1]]=dict(lambda_a = lambda_a, t_rise = 1, t_fall = 1, args=control_args[i-1])
    prop = task_obj.Propagation(H,tlist,'cheby',initial_states,'pulse_initial',pulse_options)
    oct = task_obj.Optimization(prop,'krotov',JT_conv=JT_conv,delta_JT_conv=dJT_conv,iter_dat='oct_iter.dat',iter_stop=iter_stop)
    if canoLabel:
        target_states = [localTools.rotate_state(localTools.canoGHZGen(num_qubit,canoLabel), num_qubit, 1, tlist[1])]
        oct.set_target_states(target_states)
    return oct

def monotonic(JT_iter):
    return all([JT_iter[i+1]<JT_iter[i] for i in range(len(JT_iter)-1)])

def XN_reproduce(T,task_folder,source_folder):
    best_cat_at_T = best_info(T)
    X_folder = task_folder+'stage_X/'
    XN_folder = task_folder+'stage_XN/'
    target_functional = 1
    num_qubit = 4
    JX = 1
    while JX > 0.01:
        opt_obj,_ = config_task.config_opt(source_folder + 'stage_X/')
        opt_obj.config(X_folder)
        for i in range(1):
            JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional,monotonic=True)
        JX = JT_iter[-1]
    O1 = opt_obj.observables[0]
    opt_obj.store_result(X_folder,psi_T)
    opt_obj,_ = config_task.config_opt(source_folder + 'stage_XN/')
    opt_obj.config(XN_folder)
    O2_info = ''
    csv_file = XN_folder + 'iter_info.csv'
    O2_file = XN_folder + 'O2s.data'
    data_dict = []
    O2s = read_O2s(source_folder + 'stage_XN/O2s.data')
    iter_stop = len(O2s)
    for i in range(iter_stop):
        last_success_controls = opt_obj.prop.obtain_pulse()
        O2 = O2s[i]
        opt_obj.set_observables([O1,O2])
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional)
        if i%5 == 0 or i == iter_stop - 1:
            plt.imshow(np.abs(localTools.densityMatrix(psi_T[0])))
            plt.savefig(f'states/{i}.pdf')
        if not monotonic(JT_iter):
            print('JT increased, resume to last step.')
            opt_obj.update_control(last_success_controls)
        else:
            for i in range(2**(num_qubit-1)):
                O2_info += f'{O2[i][i]} '
            O2_info += '\n'
            best_cat,JT_ss_b,JT_ss_m,varX_i,varN_i = J_T_local.cat_res_2(psi_T[0].full(),T,best_cat_at_T)
            lambda_as = opt_obj.prop.obtain_lambda_a()
            data_dict.append({'JT_ss_b':JT_ss_b,'JT_ss_m':JT_ss_m,'var_X':varX_i,'var_N':varN_i,'lambda_a':lambda_as[0]})
            with open(csv_file, 'w', newline='') as csvfile:
                fieldnames = ['JT_ss_b','JT_ss_m','var_X','var_N','lambda_a']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data_dict)
            with open(O2_file,'w') as O2_f:
                O2_f.write(O2_info)
    opt_obj.store_result(XN_folder,psi_T)
    return 0

def XN_opt(T,task_folder):
    best_cat_at_T = best_info(T)
    X_folder = task_folder+'stage_X/'
    XN_folder = task_folder+'stage_XN/'
    target_functional = 1
    JX = 1
    while JX > 0.01:
        if target_functional:opt_obj = Krotov_config_task('6+',[0,T,801],25,1e-2,1e-7,300)
        else:opt_obj = Krotov_config_task('6+',[0,T,801],0.1,1e-2,1e-5,50)
        num_qubit = 4
        O1 = J_T_local.mat_X(num_qubit,1)
        O1 = localTools.rotate_matrix(O1,num_qubit,1,T)
        opt_obj.set_observables([O1])
        opt_obj.config(X_folder)
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional,monotonic=True)
        JX = JT_iter[-1]
    opt_obj.store_result(X_folder,psi_T)
    opt_obj.oct_info['iter_stop'] = 1
    opt_obj.prop.change_lambda_a(100,0)
    opt_obj.config(XN_folder)
    iter_stop = 200
    O2_info = ''
    csv_file = XN_folder + 'iter_info.csv'
    O2_file = XN_folder + 'O2s.data'
    data_dict = []
    for i in range(iter_stop):
        last_success_controls = opt_obj.prop.obtain_pulse()
        O2 = J_T_local.mat_N(4,1,True)
        opt_obj.set_observables([O1,O2])
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional)
        if i%5 == 0 or i == iter_stop - 1:
            plt.imshow(np.abs(localTools.densityMatrix(psi_T[0])))
            plt.savefig(f'states/{i}.pdf')
        if not monotonic(JT_iter):
            print('JT increased, resume to last step.')
            opt_obj.update_control(last_success_controls)
        else:
            for i in range(2**(num_qubit-1)):
                O2_info += f'{O2[i][i]} '
            O2_info += '\n'
            best_cat,JT_ss_b,JT_ss_m,varX_i,varN_i = J_T_local.cat_res_2(psi_T[0].full(),T,best_cat_at_T)
            lambda_as = opt_obj.prop.obtain_lambda_a()
            data_dict.append({'JT_ss_b':JT_ss_b,'JT_ss_m':JT_ss_m,'var_X':varX_i,'var_N':varN_i,'lambda_a':lambda_as[0]})
            with open(csv_file, 'w', newline='') as csvfile:
                fieldnames = ['JT_ss_b','JT_ss_m','var_X','var_N','lambda_a']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data_dict)
            with open(O2_file,'w') as O2_f:
                O2_f.write(O2_info)
    opt_obj.store_result(XN_folder,psi_T)






T = 45.0
task_folder = 'states/45.0_4/'
source_folder = 'states/45.0_3/'
#XN_opt(T,task_folder)
XN_reproduce(T,task_folder,source_folder)
