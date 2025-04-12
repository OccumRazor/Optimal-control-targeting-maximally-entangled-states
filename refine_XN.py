import os,shutil,subprocess,csv,matplotlib.pyplot as plt

universal_files= [f'H{i}.dat' for i in range(5)] + ['config','None','O1.dat','O2.dat','oct_iters.dat','oct_params.dat','psi_final_after_oct.dat','psi_final.dat','psi_initial.dat'] + [
        f'pulse_initial_{i}.dat' for i in range(4)] + [f'pulse_oct_{i}.dat' for i in range(4)]

def search_n_run(O_folder):
    files = os.listdir(O_folder)
    guess_id = len(files)
    while f'O2_{guess_id}.dat' not in files:
        guess_id -= 1
    return guess_id + 1

def write_csv(csv_name,data_dict):
    fieldnames = ['current_run','current_JT_ss (best cat)','current_JT_ss (min cat)','var_X','var_N']
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dict)

def read_csv(csv_name,run0=0):
    fieldnames = ['current_run','current_JT_ss (best cat)','current_JT_ss (min cat)','var_X','var_N']
    data_dict = []
    with open(csv_name, mode ='r') as file:
        csvFile = csv.DictReader(file)
        for line in csvFile:
            line_dict = {}
            keys = list(line.keys())
            if 'current_JT_ss (min cat)' not in keys:
                fieldnames[1] = 'current_JT_ss'
            for f_name in fieldnames:
                if f_name in keys:
                    if f_name == 'current_run':
                        line_dict[f_name] = int(line[f_name]) + run0
                    else:
                        line_dict[f_name] = float(line[f_name])
                else:
                    if 'JT' in f_name:line_dict[f_name] = 1.0
                    else:line_dict[f_name] = 100.0
            data_dict.append(line_dict)
    return data_dict

def orgainse_decreased_steps(src,dst):
    csv_name = src + 'iter_info.csv'
    data_dict = read_csv(csv_name)
    run_IDs = [0]
    JTs = [float(data_dict[0]['current_JT_ss (min cat)'])]
    for i in range(1,len(data_dict)):
        if float(data_dict[i-1]['current_JT_ss (min cat)']) > float(data_dict[i]['current_JT_ss (min cat)']):
            run_IDs.append(i)
            JTs.append(float(data_dict[i]['current_JT_ss (min cat)']))
    shutil.copytree(src,dst)
    shutil.rmtree(dst+'former_opt_2/')
    os.mkdir(dst+'former_opt_2/')
    os.mkdir(dst+'former_opt_2/former_pulses/')
    for i in range(len(run_IDs)):
        shutil.copy(src+f'former_opt_2/O2_{run_IDs[i]}.dat',dst+f'former_opt_2/O2_{i}.dat')
    return 0

def combine_all(path_main,OP_folders_id,runfolders_id):
    backup_path = path_main[:len(path_main)-1]+'_copy/'
    shutil.copytree(path_main,backup_path)
    OP0 = path_main+f'former_opt_{OP_folders_id[0]}/'
    OP0_P = OP0 + 'former_pulses/'
    n_run0 = search_n_run(OP0)
    csv_name = path_main+f'former_opt_{runfolders_id[0]}/iter_info.csv'
    data_dict = read_csv(csv_name,0)
    for op_rf_id in range(1,len(OP_folders_id)):
        op_id = OP_folders_id[op_rf_id]
        rf_id = runfolders_id[op_rf_id]
        n_run1 = search_n_run(path_main+f'former_opt_{op_id}/')
        for i in range(n_run1):
            shutil.copy(path_main+f'former_opt_{op_id}/O2_{i}.dat',OP0+f'O2_{n_run0 + i}.dat')
            for j in range(4):
                shutil.copy(path_main+f'former_opt_{op_id}/former_pulses/pulse_oct_{i}_{j}.dat',OP0_P+f'pulse_oct_{n_run0 + i}_{j}.dat')
        if rf_id == -1:csv_name = path_main+'iter_info.csv'
        else:csv_name = path_main+f'former_opt_{rf_id}/iter_info.csv'
        data_dict += read_csv(csv_name,n_run0)
        n_run0 += n_run1
    csv_name = path_main+'iter_info.csv'
    write_csv(csv_name,data_dict)
    runs = []
    JT_ms = []
    JT_bs = []
    ks = list(data_dict[0].keys())
    if 'current_JT_ss (min cat)' in ks:
        JT_m_key = 'current_JT_ss (min cat)'
        JT_b_key = 'current_JT_ss (best cat)'
    else: 
        JT_m_key = 'current_JT_ss'
        JT_b_key = 'current_JT_ss'
    for i in range(len(data_dict)):
        runs.append(data_dict[i]['current_run'])
        JT_ms.append(data_dict[i][JT_m_key])
        JT_bs.append(data_dict[i][JT_b_key])
    plt.plot(runs,JT_ms,label='min')
    plt.plot(runs,JT_bs,label='best')
    plt.legend(loc='best')
    plt.savefig(path_main+'JT_per_run.pdf')
    plt.clf()
    for i in range(1,len(OP_folders_id)):
        shutil.rmtree(path_main+f'former_opt_{OP_folders_id[i]}/')
        shutil.rmtree(path_main+f'former_opt_{runfolders_id[i-1]}/')
    shutil.rmtree(backup_path)
    #delete_backup = input(f'T for delete backup folder ({backup_path})')
    #if delete_backup:
    #    shutil.rmtree(backup_path)
    #    print(f'{backup_path} deleted')
    #else:print(f'{backup_path} reserved')
    return 0

def delete_bad_res(path_main,former_opt_label,run_id):
    backup_path = path_main[:len(path_main)-1]+'_copy/'
    shutil.copytree(path_main,backup_path)
    o2_folder = path_main + f'former_opt_{former_opt_label}/'
    pulse_folder = o2_folder + 'former_pulses/'
    if os.path.exists(o2_folder + 'O2s.dat'):
        with open(o2_folder + 'O2s.dat','r') as info_f:
            text_info = ''
            for i in range(run_id):
                text_info += info_f.readline()
        os.remove(o2_folder + 'O2s.dat')
        with open(o2_folder + 'O2s.dat','w') as info_f:
            info_f.write(text_info)
        return 0
    else:
        delete_num = 0
        while os.path.exists(o2_folder + f'O2_{run_id+delete_num}.dat'):
            os.remove(o2_folder + f'O2_{run_id+delete_num}.dat')
            for i in range(4):
                os.remove(pulse_folder+f'pulse_oct_{run_id+delete_num}_{i}.dat')
            delete_num += 1
    csv_name = path_main+'/iter_info.csv'
    data_dict = read_csv(csv_name,0)[:run_id]
    write_csv(csv_name,data_dict)
    runs = []
    JT_ms = []
    JT_bs = []
    ks = list(data_dict[0].keys())
    if 'current_JT_ss (min cat)' in ks:
        JT_m_key = 'current_JT_ss (min cat)'
        JT_b_key = 'current_JT_ss (best cat)'
    else: 
        JT_m_key = 'current_JT_ss'
        JT_b_key = 'current_JT_ss'
    for i in range(len(data_dict)):
        runs.append(data_dict[i]['current_run'])
        JT_ms.append(data_dict[i][JT_m_key])
        JT_bs.append(data_dict[i][JT_b_key])
    plt.plot(runs,JT_ms,label='min')
    plt.plot(runs,JT_bs,label='best')
    plt.legend(loc='best')
    plt.savefig(path_main+'JT_per_run.pdf')
    plt.clf()
    shutil.rmtree(backup_path)
    #delete_backup = input(f'T for delete backup folder ({backup_path})')
    #if delete_backup:
    #    shutil.rmtree(backup_path)
    #    print(f'{backup_path} deleted')
    #else:print(f'{backup_path} reserved')
    return 0
Tlist = [20.84,20.86,20.88,20.9,20.92,20.94,20.96,20.98,21.0]

