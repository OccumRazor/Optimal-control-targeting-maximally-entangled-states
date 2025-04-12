import QSL_search,localTools,os,J_T_local,csv,matplotlib.pyplot as plt,shutil,csv,sys,QDYN_config
import numpy as np,datetime,read_config,tracemalloc,gc

def best_info(T):
    csv_name = 'store_result_ZZ/cat_at_T.csv'
    Tlist = []
    JTs = []
    cats = []
    with open(csv_name, mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            Tlist.append(line['T'])
            JTs.append(float(line['JT']))
            cats.append(line['cat'])
    T_id = Tlist.index(str(T))
    return cats[T_id],JTs[T_id]

def monotonicity_check(folder):
    this_JTS = np.loadtxt(folder+'oct_iters.dat', usecols=(1))
    return all([this_JTS[i+1]<this_JTS[i] for i in range(len(this_JTS)-1)])

def store_monotonic_pulse(folder,src_path = False):
    if src_path:
        for i in range(4):
            shutil.copy(src_path+f'pulse_oct_{i}.dat',folder+f'store_monotonic_pulse/pulse_oct_{i}.dat')
    else:
        for i in range(4):
            shutil.copy(folder+f'pulse_oct_{i}.dat',folder+f'store_monotonic_pulse/pulse_oct_{i}.dat')

def resume_monotonic_pulse(folder):
    for i in range(4):
        shutil.copy(folder+f'store_monotonic_pulse/pulse_oct_{i}.dat',folder+f'pulse_oct_{i}.dat')

def conditions(initial_cat,best_cat,current_JT_ss,best_JT_ss,current_run,max_run,txt_return = False):
    if txt_return:
        text_content = f'current cat: {initial_cat}, best cat:{best_cat}({initial_cat == best_cat})\n'
        text_content += f'current_JT_ss: {current_JT_ss}, best_JT_ss: {best_JT_ss} ({current_JT_ss>best_JT_ss})\n'
        text_content += f'current run : {current_run}, max run: {max_run} ({current_run<max_run})\n'
        return text_content
    else:return all([current_JT_ss>best_JT_ss,current_run<max_run])

def store_success_run(main_folder,run_ID,former_opt_label,iter_stop,O2_mat):
    store_folder = main_folder + f'former_opt_{former_opt_label}/'
    if not os.path.exists(store_folder+'former_pulses/'):os.mkdir(store_folder+'former_pulses/')
    with open(store_folder + 'O2s.dat','w') as O2_rec:
        O2_mat_diag_1st = [O2_mat[i][i] for i in range(int(len(O2_mat)/2))]
        O2_rec.write(str(O2_mat_diag_1st) + '\n')
    if iter_stop >= 50:
        for i in range(4):
            shutil.copy(main_folder+f'pulse_oct_{i}.dat',store_folder + f'former_pulses/pulse_oct_{run_ID}_{i}.dat')

def restore(main_folder,folder,last_success_run,former_opt_label,iter_stop):
    store_folder = main_folder + f'former_opt_{former_opt_label}/former_pulses/'
    if iter_stop >= 50:
        for i in range(4):
            shutil.copy(store_folder + f'pulse_oct_{last_success_run}_{i}.dat', folder+f'pulse_oct_{i}.dat') 


def prepare_pulse(folder):
    for i in range(4):
        os.remove(folder+f'pulse_initial_{i}.dat')
        os.rename(folder+f'pulse_oct_{i}.dat',folder+f'pulse_initial_{i}.dat')

def draw_iters(runfolder,JT_m_per_run,JT_b_per_run,best_JT_ss,X_i,N_i,cat_i,best_cat):
    fig,ax = plt.subplots(2,1,figsize=(12,9))
    ax[0].plot(JT_m_per_run,label='min')
    ax[0].plot(JT_b_per_run,label='best')
    ax[0].plot([0,len(JT_b_per_run)-1],[min(JT_m_per_run)]*2,'--',alpha=0.5)
    ax[0].plot([0,len(JT_b_per_run)-1],[best_JT_ss]*2,'--',alpha=0.5,label='goal')
    ax[1].plot(X_i,label = 'Var(X)')
    ax[1].plot(N_i,label = 'Var(N)')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    ax[0].grid()
    ax[1].grid()
    ax[0].set_ylabel('inFidelity')
    fig.supxlabel('iters')
    plt.title(f'min cat: {cat_i}, best cat: {best_cat},JT_min (at {JT_m_per_run.index(min(JT_m_per_run))}):{min(JT_m_per_run)}')
    plt.legend(loc='best')
    plt.savefig(runfolder+f'JT_per_run.pdf')
    file_rank = runfolder.split('/')
    task_id = file_rank[-2]
    file_rank=file_rank[:-2]
    folder_main = '/'.join(file_rank) + '/'
    plt.savefig(folder_main+f'_{task_id}JT_per_run.pdf')

def app():
    lt = []
    for i in range(0, 100000):
        lt.append(i)

def cat_0_info(runfolder,src_f,T,best_cat):
    num_qubit = 4
    Ham_num = 3
    t_step = 801
    initial_state = localTools.canoGHZGen(num_qubit, '0').full()
    QDYN_config.qdyn_prop(Ham_num,(T,t_step),initial_state,runfolder,src_f,'pulse_oct',num_qubit = num_qubit)
    initial_cat , current_JT_ss_b,current_JT_ss_m ,varX_i,varN_i = J_T_local.cat_res(runfolder,T,best_cat)
    return initial_cat , current_JT_ss_b,current_JT_ss_m ,varX_i,varN_i

def trial_run(src_path,T,runfolder,iter_stop,N_factor = 1.5):
    if not os.path.exists(runfolder):os.mkdir(runfolder)
    tracemalloc.start()
    localTools.store_control_source(src_path,runfolder)
    run_method = {"immediate_return":False,"slurm":False,"store":False,"t_sleep":1,'N_factor':N_factor}
    files = os.listdir(runfolder)
    max_run = 1000
    max_run = int(15000/iter_stop)
    if max_run > 15000:max_run = 15000
    if 'former_opt_1' in files:
        former_opt_label = 1
        while f'former_opt_{former_opt_label}' in files and former_opt_label < max_run + 15:
            if os.path.exists(f'{runfolder}former_opt_{former_opt_label}/former_pulses/'):
                shutil.rmtree(f'{runfolder}former_opt_{former_opt_label}/former_pulses/')
            former_opt_label += 1
    else:former_opt_label = 1
    os.mkdir(runfolder+f'former_opt_{former_opt_label}/')
    os.mkdir(runfolder+f'former_opt_{former_opt_label}/former_pulses/')
    os.mkdir(runfolder+'store_monotonic_pulse/')
    record_file = runfolder + 'opt_record_trial_run.log'
    best_cat , best_JT_ss = best_info(T)
    best_JT_ss *= 1.05
    current_run = 0
    initial_cat , current_JT_ss_b,current_JT_ss_m ,_,_ = cat_0_info(runfolder,src_path,T,best_cat)
    text_content = f'Goal for T = {T}: best_cat: {best_cat}, best_JT_ss: {best_JT_ss}\nsource: {src_path},\n'
    text_content += f'source_info: source_JT_ss: {current_JT_ss_b}\n iter_stop for each opt trial: {iter_stop}\n'
    text_content += f'max_run: {max_run}, results shown in iter_info.csv\n'
    csv_file = runfolder + 'iter_info.csv'
    with open(record_file,'w') as log_f:
        log_f.write(text_content)
    JT_m_per_run = [current_JT_ss_m]
    JT_b_per_run = [current_JT_ss_b]
    X_i = []
    N_i = []
    data_dict = []
    JTs = []
    JT_min = [current_JT_ss_m]
    while conditions(initial_cat,best_cat,current_JT_ss_m,best_JT_ss,current_run,max_run):
        if current_run == 0:
            std_source = src_path
            store_monotonic_pulse(runfolder,src_path)
            QSL_search.state_task(T,4,3,'ZZ','0+',2,iter_stop,runfolder_main=runfolder,control_source = std_source,run_method = run_method)
        else:
            std_source = runfolder
            if current_run == 1:
                read_config.kill_obsevable(str(runfolder))
            O2_mat = J_T_local.mat_N(4,N_factor,True)
            mat_text = localTools.matrix2text(O2_mat)
            with open(runfolder + 'O2.dat','w') as o2_f:o2_f.write(mat_text)
            prepare_pulse(runfolder)
            QDYN_config.runner(runfolder,0,0,0,4,n_cpu=1,run_method = run_method)
        cat_i,current_JT_ss_b,current_JT_ss_m,varX_i,varN_i = J_T_local.cat_res(runfolder,T,best_cat)
        JT_min.append(current_JT_ss_m)
        if monotonicity_check(runfolder):
            JTs.append(current_JT_ss_m)
            X_i.append(varX_i)
            N_i.append(varN_i)
            app()
            text_content += f'run {current_run}/{max_run}, current_JT_ss: {current_JT_ss_m}, mem_usage: {tracemalloc.get_traced_memory()}\n'
            with open(record_file,'w') as log_f:log_f.write(text_content)
            data_dict.append({'current_run':current_run,'current_JT_ss (best cat)':current_JT_ss_b,'current_JT_ss (min cat)':current_JT_ss_m,'var_X':varX_i,'var_N':varN_i,'complete_time':datetime.datetime.now()})
            with open(csv_file, 'w', newline='') as csvfile:
                fieldnames = ['current_run','current_JT_ss (best cat)','current_JT_ss (min cat)','var_X','var_N','complete_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data_dict)
            if current_run == 0:O2_mat = localTools.matrixReader(str(runfolder)+'/O2.dat',16)
            store_success_run(runfolder,current_run,former_opt_label,iter_stop,O2_mat)
            JT_m_per_run.append(current_JT_ss_m)
            JT_b_per_run.append(current_JT_ss_b)
            store_monotonic_pulse(runfolder)
        else:
            resume_monotonic_pulse(runfolder)
            JT_min = JT_min[:len(JT_min)-1]
            text_content += f'run {current_run}/{max_run}, not decreasing monotonically, restored\n'
            with open(record_file,'w') as log_f:log_f.write(text_content)
            current_run -= 1
        current_run += 1
        if not conditions(initial_cat,best_cat,current_JT_ss_m,best_JT_ss,current_run,max_run):
            condition_text = conditions(initial_cat,best_cat,current_JT_ss_m,best_JT_ss,current_run,max_run)
            text_content += str(condition_text) + '\n'
            with open(record_file,'w') as log_f:log_f.write(text_content)
        gc.collect()
    shutil.rmtree(runfolder+'store_monotonic_pulse/')
    tracemalloc.stop()
    draw_iters(runfolder,JT_m_per_run,JT_b_per_run,best_JT_ss,X_i,N_i,cat_i,best_cat)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        src_path = sys.argv[1]
        T = float(sys.argv[2])
        runfolder = sys.argv[3]
        iter_stop = int(sys.argv[4])
        trial_run(src_path,T,runfolder,iter_stop)
    elif len(sys.argv) == 6:
        src_path = sys.argv[1]
        T = float(sys.argv[2])
        runfolder = sys.argv[3]
        iter_stop = int(sys.argv[4])
        N_factor = float(sys.argv[5])
        trial_run(src_path,T,runfolder,iter_stop,N_factor)
    else:
        print("Error: Number of Keys do not match.")
        sys.exit(1) 