import numpy as np,csv,config_task,matplotlib.pyplot as plt,J_T_local,localTools,task_obj,shutil,copy

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

def read_lambda_a(file_name):
    lambda_as = []
    with open(file_name, mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            lambda_as.append(float(line['lambda_a']))
    return lambda_as

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

def Krotov_config_task(canoLabel,tlist,lambda_a,JT_conv,dJT_conv,iter_stop,control_source=None,header=None):
    num_qubit = 4
    H=localTools.Hamiltonian(num_qubit)
    initial_states = [localTools.canoGHZGen(num_qubit,'0'*num_qubit)]
    pulse_options={}
    if not control_source:
        control_source = 1
        header = None
    control_args = localTools.control_generator(num_qubit,control_source,tlist[1],header)
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

def convergent(JXN_iter,dJT_conv):
    sampling_steps = 10
    if len(JXN_iter) > sampling_steps:
        JT_relative = [(JXN_iter[i][0]-JXN_iter[i][1])/JXN_iter[i][0] for i in range(len(JXN_iter)-sampling_steps,len(JXN_iter))]
        if sum(JT_relative)/sampling_steps < dJT_conv:
            return True
    return False

def XN_reproduce(T,task_folder,source_folder):
    best_cat_at_T = best_info(T)
    X_folder = task_folder+'stage_X/'
    XN_folder = task_folder+'stage_XN/'
    target_functional = 1
    num_qubit = 4
    opt_obj,_ = config_task.config_opt(source_folder + 'stage_X/')
    opt_obj.config(X_folder)
    for i in range(1):
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional,monotonic=True,out_file = f'{X_folder}/oct_iters.dat')
    JX = JT_iter[-1]
    opt_obj.store_result(X_folder,psi_T)
    opt_obj,_ = config_task.config_opt(source_folder + 'stage_XN/')
    O1 = opt_obj.observables[0]
    opt_obj.config(XN_folder)
    O2_info = ''
    csv_file = XN_folder + 'iter_info.csv'
    O2_file = XN_folder + 'O2s.dat'
    data_dict = []
    O2s = read_O2s(source_folder + 'stage_XN/O2s.dat')
    lambda_as = read_lambda_a(source_folder + 'stage_XN/iter_info.csv')
    iter_stop = len(O2s)
    JXN_iter = []
    csv_file = open(csv_file,'w',newline='')
    O2_file = open(O2_file,'w')
    fieldnames = ['JT_ss_b','JT_ss_m','var_X','var_N','lambda_a']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    i = 0
    while all([i < iter_stop,JX < 0.95,not convergent(JXN_iter,1e-6)]):
        last_success_controls = opt_obj.prop.obtain_pulse()
        if i <len(O2s):O2 = O2s[i]
        else:O2 = J_T_local.mat_N(4,1,True)
        opt_obj.set_observables([O1,O2])
        if i <len(lambda_as):opt_obj.prop.change_lambda_a(lambda_as[i],0)
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional)
        if not monotonic(JT_iter):
            print('JT increased, resume to last step.')
            opt_obj.update_control(last_success_controls)
        else:
            O2_info = ''
            for j in range(2**(num_qubit-1)):
                O2_info += f'{O2[j][j]} '
            O2_info += '\n'
            best_cat,JT_ss_b,JT_ss_m,JX,JN = J_T_local.cat_res_2(psi_T[0].full(),T,best_cat_at_T)
            lambda_as = opt_obj.prop.obtain_lambda_a()
            data_dict.append({'JT_ss_b':JT_ss_b,'JT_ss_m':JT_ss_m,'var_X':JX,'var_N':JN,'lambda_a':lambda_as[0]})
            writer.writerow(data_dict[-1])
            O2_file.write(O2_info)
            csv_file.flush()
            O2_file.flush()
        i += 1
    csv_file.close()
    O2_file.close()
    opt_obj.store_result(XN_folder,psi_T)
    return 0

def XN_opt(T,task_folder,lambda_X,JX_goal,iter_stop_X,iter_stop_XN):
    best_cat_at_T = best_info(T)
    X_folder = task_folder+'stage_X/'
    XN_folder = task_folder+'stage_XN/'
    target_functional = 1
    JX = 100
    while JX > JX_goal:
        #opt_obj = Krotov_config_task('6+',[0,T,801],25,JX_goal,1e-7,iter_stop_X)
        opt_obj = Krotov_config_task('6+',[0,T,801],25,1e-2,1e-7,iter_stop_X)
        num_qubit = 4
        O1 = J_T_local.mat_X(num_qubit,1)
        O1 = localTools.rotate_matrix(O1,num_qubit,1,T)
        opt_obj.set_observables([O1])
        opt_obj.config(X_folder)
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional,monotonic=True,out_file = f'{X_folder}/oct_iters.dat')
        JX = JT_iter[-1]
    psi_T_last = copy.deepcopy(psi_T)
    opt_obj.store_result(X_folder,psi_T)
    opt_obj.oct_info['iter_stop'] = 1
    opt_obj.prop.change_lambda_a(100,0)
    opt_obj.config(XN_folder)
    csv_file = XN_folder + 'iter_info.csv'
    O2_file = XN_folder + 'O2s.dat'
    data_dict = []
    i = 0
    JXN_iter = []
    csv_file = open(csv_file,'w',newline='')
    O2_file = open(O2_file,'w')
    fieldnames = ['JT_ss_b','JT_ss_m','var_X','var_N','lambda_a']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    O1 = J_T_local.mat_X(num_qubit,lambda_X)
    O1 = localTools.rotate_matrix(O1,num_qubit,1,T)
    #while all([i < iter_stop_XN,JX < 0.6,not convergent(JXN_iter,1e-6)]):
    while all([i < iter_stop_XN,JX < 0.95,not convergent(JXN_iter,1e-7)]):
        last_success_controls = opt_obj.prop.obtain_pulse()
        O2 = J_T_local.mat_N(4,1,True)
        opt_obj.set_observables([O1,O2])
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional)
        best_cat,JT_ss_b,JT_ss_m,JX,JN = J_T_local.cat_res_2(psi_T[0].full(),T,best_cat_at_T)
        if not monotonic(JT_iter):
            psi_T = copy.deepcopy(psi_T_last)
            print(f'JT increased: {not monotonic(JT_iter)}, resume to last step.')
            opt_obj.update_control(last_success_controls)
        else:
            psi_T_last = copy.deepcopy(psi_T)
            JXN_iter.append(JT_iter)
            O2_info = ''
            for j in range(2**(num_qubit-1)):
                O2_info += f'{O2[j][j]} '
            O2_info += '\n'
            lambda_as = opt_obj.prop.obtain_lambda_a()
            data_dict.append({'JT_ss_b':JT_ss_b,'JT_ss_m':JT_ss_m,'var_X':JX,'var_N':JN,'lambda_a':lambda_as[0]})
            writer.writerow(data_dict[-1])
            O2_file.write(O2_info)
            csv_file.flush()
            O2_file.flush()
        i += 1
    csv_file.close()
    O2_file.close()
    opt_obj.store_result(XN_folder,psi_T)

def XN_opt_continue(T,task_folder,lambda_X,control_source,iter_stop_XN):
    best_cat_at_T = best_info(T)
    X_folder = task_folder+'stage_X/'
    XN_folder = task_folder+'stage_XN/'
    target_functional = 1
    opt_obj = Krotov_config_task('6+',[0,T,801],25,1e-2,1e-7,1,control_source,'pulse_oct')
    num_qubit = 4
    JX = 0
    opt_obj.prop.change_lambda_a(100,0)
    opt_obj.config(XN_folder)
    shutil.copytree(control_source,X_folder)
    csv_file = XN_folder + 'iter_info.csv'
    O2_file = XN_folder + 'O2s.dat'
    data_dict = []
    JXN_iter = []
    JX_iter = []
    JT_b_iter = []
    JT_m_iter = []
    JN_iter = []
    csv_file = open(csv_file,'w',newline='')
    O2_file = open(O2_file,'w')
    fieldnames = ['JT_ss_b','JT_ss_m','var_X','var_N','lambda_a']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    O1 = J_T_local.mat_X(num_qubit,lambda_X)
    O1 = localTools.rotate_matrix(O1,num_qubit,1,T)
    O2 = J_T_local.mat_N(4,1,True)
    O2_ori = [8.0, 7.0, 6.0, 3.0, 2.0, 4.0, 5.0, 1.0]
    O2 = np.zeros([16,16])
    for i in range(8):
        O2[i][i] = O2_ori[i]
        O2[15-i][15-i] = O2_ori[i]
    opt_obj.set_observables([O1,O2])
    i = 0
    while all([i < iter_stop_XN,JX < 0.95,not convergent(JXN_iter,1e-6)]):
        last_success_controls = opt_obj.prop.obtain_pulse()
        JT_iter,psi_T = opt_obj.Krotov_optimization(target_functional)
        if i%10 == 0 or i == iter_stop_XN - 1:
            plt.imshow(np.abs(localTools.densityMatrix(psi_T[0])))
            plt.savefig(f'{task_folder}{i}.pdf')
        if not monotonic(JT_iter):
            print(f'JT increased: {not monotonic(JT_iter)}, resume to last step.')
            opt_obj.update_control(last_success_controls)
        else:
            min_cat,JT_ss_b,JT_ss_m,JX,JN = J_T_local.cat_res_2(psi_T[0],T,best_cat_at_T)
            JN = J_T_local.opVar(O2,psi_T[0].full())
            JX_iter.append(JX)
            JN_iter.append(JN)
            JT_b_iter.append(JT_ss_b)
            JT_m_iter.append(JT_ss_m)
            JXN_iter.append(JT_iter)
            O2_info = ''
            for j in range(2**(num_qubit-1)):
                O2_info += f'{O2[j][j]} '
            O2_info += '\n'
            lambda_as = opt_obj.prop.obtain_lambda_a()
            data_dict.append({'JT_ss_b':JT_ss_b,'JT_ss_m':JT_ss_m,'var_X':JX,'var_N':JN,'lambda_a':lambda_as[0]})
            writer.writerow(data_dict[-1])
            O2_file.write(O2_info)
            csv_file.flush()
            O2_file.flush()
        i += 1
    print(f'{i < iter_stop_XN},{JX < 0.95},{not convergent(JXN_iter,1e-6)}')
    csv_file.close()
    O2_file.close()
    opt_obj.config(XN_folder)
    opt_obj.store_result(XN_folder,psi_T)

import os
for T in [25.0,30.0,35.0,40.03,45.0,50.03]:
    i = 0
    while os.path.exists(f'reproduce_{T}_{i}/'): i += 1
    task_folder = f'reproduce_{T}_{i}/'
    if T == 40.03:control_source = f'opt_examples/{T}_local_minima/'
    else:control_source = f'opt_examples/{T}/'
    XN_reproduce(T,task_folder,control_source)