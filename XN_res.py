import localTools,J_T_local,os,csv,QSL_search,shutil,matplotlib.pyplot as plt,numpy as np,subprocess,random

def text_content_XN_trial_run(src_folder,T,runfolder_main,job_id,iter_stop,N_factor):
    job_name = f'{runfolder_main[3]}{runfolder_main[8:]}{str(T)[2:]}_{job_id}'
    if N_factor:
        default_slurm_script = """#!/bin/bash
#SBATCH --exclusive=user
#SBATCH --job-name={job_name} # Job name, will show up in squeue output
#SBATCH --ntasks=1                                 # 
#SBATCH --nodes=1                                  # Ensure that all cores are on one machine
#SBATCH --cpus-per-task=1                          # Number of cores
#SBATCH --time=2-12:00:00 # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=1500 # Memory per cpu in MB (see also --mem) 
#SBATCH --output={runfolder_main}{T}/{job_id}stb.out  # File to which standard out will be written
#SBATCH --error={runfolder_main}{T}/{job_id}stb.err   # File to which standard err will be written


echo $SLURM_JOBID
echo $PWD
scontrol show job $SLURM_JOBID
source activate py312
module load intel/oneapi/2024.2.0

srun --cpu-bind=cores python autoXN.py {src_folder} {T} {runfolder_main}{T}/{job_id}/ {iter_stop} {N_factor}""".format(
    job_name=job_name,src_folder = src_folder,T=T,runfolder_main=runfolder_main,job_id=job_id,iter_stop=iter_stop,N_factor=N_factor)
    else:
        default_slurm_script = """#!/bin/bash
#SBATCH --exclusive=user
#SBATCH --job-name={job_name} # Job name, will show up in squeue output
#SBATCH --ntasks=1                                 # 
#SBATCH --nodes=1                                  # Ensure that all cores are on one machine
#SBATCH --cpus-per-task=1                          # Number of cores
#SBATCH --time=2-12:00:00 # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=1500 # Memory per cpu in MB (see also --mem) 
#SBATCH --output={runfolder_main}{T}/{job_id}stb.out  # File to which standard out will be written
#SBATCH --error={runfolder_main}{T}/{job_id}stb.err   # File to which standard err will be written

# store job info in output file
echo $SLURM_JOBID
echo $PWD
scontrol show job $SLURM_JOBID
source activate py312
module load intel/oneapi/2024.2.0

srun --cpu-bind=cores python autoXN.py {src_folder} {T} {runfolder_main}{T}/{job_id}/ {iter_stop}""".format(
    job_name=job_name,src_folder = src_folder,T=T,runfolder_main=runfolder_main,job_id=job_id,iter_stop=iter_stop)
    return default_slurm_script

def cat_res(folder,T):
    if os.path.exists(f'{folder}/psi_final_after_oct.dat'):
        state = localTools.stateReader(f'{folder}/psi_final_after_oct.dat',4)
        state = localTools.rotate_state(state,4,0,float(T))
        cat_i,JT_i = J_T_local.bestCat(state)
        varX_i = J_T_local.opVarX(state,4)
        varN_i = J_T_local.opVarN(state,4)
        return cat_i,JT_i,varX_i,varN_i
    else:return None,1,None,None

def read_source_info(folder):
    if os.path.exists(f'{folder}/source_info.csv'):
        with open(f'{folder}/source_info.csv','r') as info_f:
            csvFile = csv.DictReader(info_f)
            for line in csvFile:
                src_JT_ss = line['JT_i']
        return float(src_JT_ss)
    else: return 1

def read_iter_info(folder):
    if os.path.exists(f'{folder}/iter_info.csv'):
        JT_ss = []
        n_iter = []
        with open(f'{folder}/iter_info.csv','r') as info_f:
            csvFile = csv.DictReader(info_f)
            for line in csvFile:
                n_iter.append(int(line['current_run']))
                JT_ss.append(float(line['current_JT_ss']))
        return n_iter,JT_ss
    else: return 0,0

def restore(folder):
    files = os.listdir(folder)
    if f'former_opt_1' in files:
        former_opt_label = 2
        while f'former_opt_{former_opt_label}' in files:
            former_opt_label += 1
        former_opt_label -= 1
        former_path = folder + f'former_opt_{former_opt_label}/'
        for file in files:
            if 'former_opt' not in file:
                os.remove(folder + file)
        files = os.listdir(former_path)
        for file in files:  
            shutil.move(former_path + file, folder) 
        os.rmdir(former_path)

def store(source_folder,tgt_folder_T):
    if os.path.exists(tgt_folder_T):
        shutil.rmtree(tgt_folder_T)
    shutil.copytree(source_folder,tgt_folder_T)

def compare_store(candidate_folder,cat_candidate,JT_candidate,T):
    tgt_main = 'store_result_XN/'
    tgt_folder = tgt_main + str(T) + '/'
    if not os.path.exists(tgt_folder):
        store(candidate_folder,tgt_folder)
        return 0
    else:
        cat_best,JT_best,X_best,N_best = cat_res(tgt_folder,T)
        if cat_best == cat_candidate and JT_candidate < JT_best:
            store(candidate_folder,tgt_folder)

def plain_submit(src_folders,task_id,Tlist,opt_parameters,name_tag,task_label=False,N_factor = 1.5):
    #iter_stop = 200
    iter_stop = opt_parameters['iter_stop']
    ids = opt_parameters['ids']
    for src_f in src_folders:
        if isinstance(task_label,bool) and not task_label:
            task_label = 0
            while os.path.exists(f'rf/{task_id}XN_{name_tag}{task_label}/'): task_label += 1
        runfolder_main = f'rf/{task_id}XN_{name_tag}{task_label}/'
        if not os.path.exists(runfolder_main):os.mkdir(runfolder_main)
        for T in Tlist:
            if not os.path.exists(runfolder_main+f'{T}/'):os.mkdir(runfolder_main+f'{T}/')
            if ids:
                std_source = [src_f+f'{T}/{i}/' for i in ids]
            else:
                files = os.listdir(src_f+f'{T}/')
                std_source = []
                for file in files:
                    if os.path.isdir(src_f+f'{T}/{file}'):
                        std_source.append(src_f+f'{T}/{file}/')
            print(runfolder_main+f'{T}/ ({len(std_source)} jobs)')
            for i in range(len(std_source)):
                os.mkdir(runfolder_main+f'{T}/{i}/')
                slrm_scrpt = text_content_XN_trial_run(std_source[i],T,runfolder_main,i,iter_stop,N_factor)
                with open(f'{runfolder_main}/{T}/run_rf_{i}.sh','w') as f:
                    f.write(slrm_scrpt)
                fortran_result = subprocess.run(["sbatch", f'{runfolder_main}/{T}/run_rf_{i}.sh'])

def gather_result(src_folders,c_opt=False):
    csv_name = 'store_result_ZZ/cat_at_T.csv'
    just_run = True
    iter_stop = 100
    Tlist = []
    JTs = []
    cats = []
    T_max = 20.50
    T_min = 20.0
    with open(csv_name, mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            #Tlist.append(float(line['T']))
            Tlist.append(line['T'])
            JTs.append(float(line['JT']))
            cats.append(line['cat'])
    for src_f in src_folders:
        if c_opt:
            task_label = 0
            while os.path.exists(f'rf/0+_ZZ{task_label}/'): task_label += 1
            runfolder_main = f'rf/0+_ZZ{task_label}/'
            if not os.path.exists(runfolder_main):os.mkdir(runfolder_main)
        text_content = ''
        folders = os.listdir(src_f)
        Ts = []
        for folder in folders:
            try:
                Ts.append(float(folder))
            except ValueError:
                pass
        Ts.sort()
        if Ts[-1] > T_max:
            max_T_id = 0
            while Ts[max_T_id] <= T_max:max_T_id += 1
            Ts = Ts[:max_T_id]
        if Ts[0] < T_min:
            min_T_id = 0
            while Ts[min_T_id] <= T_min and min_T_id < len(Ts)-1:min_T_id += 1
            Ts = Ts[min_T_id:]
        n_m_all = 0
        n_dm_all = 0
        n_m_g = 0
        n_dm_g = 0
        JT_ss_best = []
        min_JT_ss_new = []
        min_JT_ss_old  = []
        for T in Ts:
            Tid = Tlist.index(str(T))
            JT_ss_best.append(JTs[Tid])
            best_cat = cats[Tid]
            text_content += f'\n{T} res, best cat: {best_cat} (JT_ss = {JTs[Tid]}):\n'
            #print(f'best cat for T = {file}: {best_cat}')
            match_res = ''
            n_match = 0
            dismatch_res = ''
            n_dismatch = 0
            sub_f = src_f+'/'+str(T)+'/'
            all_candidates = os.listdir(sub_f)
            matched_candidates = []
            matched_JT = []
            matched_id = []
            JT_ss_T_new = []
            JT_ss_T_old = []
            for each_candidate in all_candidates:
                cat_i,JT_i,varX_i,varN_i = cat_res(sub_f+each_candidate+'/',T)
                n_iter,JT_ss = read_iter_info(sub_f+each_candidate+'/')
                if n_iter:JT_i = min(JT_ss)
                src_JT_ss = read_source_info(sub_f+each_candidate+'/')
                JT_ss_T_old.append(src_JT_ss)
                JT_ss_T_new.append(JT_i)
                if cat_i:
                    if cat_i == best_cat: 
                        matched_JT.append(JT_i)
                        matched_id.append(each_candidate)
                        if n_iter:match_res += f'\t{each_candidate}: {cat_i}, {JT_i} at iter {n_iter[JT_ss.index(JT_i)]}\n'
                        else:match_res += f'\t{each_candidate}: {cat_i}, {JT_i}\tVar(X): {varX_i}, Var(N): {varN_i}\n'
                        match_res += f'\t\tsource JT_ss: {src_JT_ss}\n'
                        if JT_i > src_JT_ss:
                            restore(sub_f+each_candidate+'/')
                            match_res += '\t\t Restored to former result\n'
                        elif JT_i == src_JT_ss:match_res += '\t\t Nothing changed\n'
                        n_match += 1
                        matched_candidates.append(each_candidate)
                    else: 
                        #if JT_i != src_JT_ss:
                        #    restore(sub_f+each_candidate+'/')
                        #    dismatch_res += f'\t{each_candidate}: {cat_i}, {JT_i}\n\t\tRestored to former result\n'
                        #else:dismatch_res += f'\t{each_candidate}: {cat_i}, {JT_i}\n\t\t Nothing changed\n'
                        n_dismatch += 1
            if match_res or just_run: 
                n_m_g += 1
                min_JT_ss = min(matched_JT)
                candidate_folder = f'{src_f}/{T}/{matched_id[matched_JT.index(min_JT_ss)]}/'
                compare_store(candidate_folder,best_cat,min_JT_ss,T)
                match_res = f'min JT_ss: {min_JT_ss},  corresponding job: {matched_id[matched_JT.index(min_JT_ss)]}\n' + match_res
                if c_opt:
                    std_source = [f'{src_f}/{T}/{matched_candidate}/' for matched_candidate in matched_candidates]
                    print(f'configuring {runfolder_main}{T} ({len(std_source)}) jobs.')
                    os.mkdir(runfolder_main+str(T))
                    with open(runfolder_main+f'{T}/source_info.log','w') as log_f:
                        log_f.write(match_res)
                    if False:
                        QSL_search.state_task(T,4,3,'ZZ','0+',2,runfolder_main=runfolder_main,control_source = std_source)
                    for i in range(len(std_source)):
                        os.mkdir(runfolder_main+f'{T}/{i}/')
                        slrm_scrpt = text_content_XN_trial_run(std_source[i],T,runfolder_main,i,iter_stop)
                        with open(f'{runfolder_main}/{T}/run_rf_{i}.sh','w') as f:
                            f.write(slrm_scrpt)
                        fortran_result = subprocess.run(["sbatch", f'{runfolder_main}/{T}/run_rf_{i}.sh'])
            else: n_dm_g += 1
            n_m_all += n_match
            n_dm_all += n_dismatch
            text_content += f'match rate: {n_match}/{n_match + n_dismatch}\n' + match_res + '\n' + dismatch_res
            min_JT_ss_new.append(min(JT_ss_T_new))
            min_JT_ss_old.append(min(JT_ss_T_old))
        if n_m_all == 0 and c_opt:os.rmdir(runfolder_main)
        over_all = f'Over all match rate:\n\t{n_m_all}/{n_m_all + n_dm_all}\nGroupwise match rate:\n\t{n_m_g}/{n_m_g+n_dm_g}\n'
        with open(src_f+'/cat_res.log','w') as log_f: 
            log_f.write(over_all + text_content)
        plt.plot(Ts,min_JT_ss_new,'.',label = 'new')
        plt.plot(Ts,min_JT_ss_old,'.',label = 'old')
        plt.plot(Ts,JT_ss_best,'.',label = 'best')
        plt.legend(loc='best')
        for i in range(len(Ts)):
            JT_ss_pair = [min_JT_ss_old[i],min_JT_ss_new[i]]
            plt.plot([Ts[i],Ts[i]],[min(JT_ss_pair),max(JT_ss_pair)],'r--')
        plt.savefig(src_f+'/cat_JTss_T.pdf')
        plt.clf()


cat_dict = {20.01:['6+'],20.75:['6+'],20.76:['6+'],20.77:['6+','7+'],20.78:['0+'],20.79:['0+'],20.8:['6+'],20.81:['6+'],20.82:['6+'],20.83:['6+'],20.84:['6+','7+'],20.85:['7+'],20.86:['7+'],20.87:['7+'],20.88:['2+'],
            20.89:['2+'],20.9:['2+'],20.91:['6+'],20.92:['6+'],20.93:['0+'],20.94:['0+'],20.95:['4+'],20.96:['4+',],20.97:['4+'],20.97:['4+'],20.98:['4+'],
            20.99:['4+'],21.0:['6+'],21.01:['6+'],21.02:['1+'],21.03:['1+'],21.04:['1+'],21.05:['1+'],21.06:['1+'],21.07:['6+'],21.08:['0+','6+'],21.09:['0+'],21.1:['0+'],
            21.11:['6+'],21.12:['6+'],21.13:['6+'],21.14:['6+'],21.15:['6+','7+'],21.16:['7+'],21.17:['2+','7+'],21.18:['2+','7+'],21.19:['2+','7+'],
            21.2:['2+','7+'],21.21:['2+'],21.22:['2+','6+'],21.23:['0+','6+'],21.24:['0+'],21.25:['0+'],22.0:['1+'],23.0:['2+','3+'],24.0:['3+'],27.0:['0+'],40.03:['1+']}

def mv_chosen(src_main,src_ids,tgt):
    if isinstance(src_ids,int):
        src_ids = [src_ids]
    if not os.path.exists(tgt):os.mkdir(tgt)
    tgt_files = os.listdir(tgt)
    for src_id in src_ids:
        src_id = str(src_id)
        if src_id in tgt_files or f'_{src_id}JT_per_run.pdf' in tgt_files:
            assigned_id = str(random.randint(100,10000))
            while assigned_id in tgt_files:assigned_id = str(random.randint(100,10000))
            print(f'{src_main}:src_id {src_id} exists, new id {assigned_id} assgined')
        else:
            assigned_id = src_id
            print(f'{src_main}:src_id {src_id} moved to tgt')
        subprocess.run(['mv',f'{src_main}{src_id}',f'{tgt}{assigned_id}'])
        subprocess.run(['mv',f'{src_main}_{src_id}JT_per_run.pdf',f'{tgt}_{assigned_id}JT_per_run.pdf'])
    return 0

def non_optimal(path,tgt = None):
    files = os.listdir(path)
    Tlist = [float(file) for file in files]        
    Tlist.sort()
    if tgt:
        if not os.path.exists(tgt):
            os.mkdir(tgt)
    for T in Tlist:
        src_main = path+f'{T}/'
        tgt_T = tgt + f'{T}/'
        best_cats = cat_dict[T]
        files = os.listdir(path+f'{T}/')
        for file in files:
            id_path = path+f'{T}/{file}/'
            if os.path.isdir(id_path):
                cat_i,JT_i,varX_i,varN_i = cat_res(id_path,T)
                if cat_i in best_cats:
                    if tgt:
                        mv_chosen(src_main,file,tgt_T)
                    else:
                        print(id_path+f'optimal {cat_i}')
                else:
                    shutil.rmtree(id_path)
    return 0



for i in range(0):
    non_optimal(f'rf/{i}XN_ZZ1/','rf/6XN_ZZ0/')
    #subprocess.run(['rm','-r',f'rf/{i}XN_ZZ1/'])
#non_optimal('rf/2XN_ZZ2/')
#non_optimal('rf/2XN_ZZ1/')
name_tag = 'ZZ'
i_range = 0
task_id = 0
for task_id in [4]:
    if os.path.exists(f'rf/{task_id}XN_{name_tag}{i_range}/'):
        files = os.listdir(f'rf/{task_id}XN_{name_tag}{i_range}/')
        Tlist = [float(file) for file in files]
    Tlist.sort()
    for i in range(5):
        task_label = 0
        while os.path.exists(f'rf/{task_id}XN_{name_tag}{task_label}/'): task_label += 1
        for T in Tlist:
            iter_stop = 10
            N_factor = 5.1 + 0.1 * i
            opt_parameters = {'iter_stop':iter_stop,'ids':False}
            plain_submit([f'rf/{task_id}XN_{name_tag}{i_range}/'],task_id,[T],opt_parameters,name_tag,task_label=task_label,N_factor=N_factor)



def best_info(T):
    csv_name = 'store_result_ZZ/cat_at_T.csv'
    Tlist = []
    JTs = []
    cats = []
    with open(csv_name, mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            #Tlist.append(float(line['T']))
            Tlist.append(line['T'])
            JTs.append(float(line['JT']))
            cats.append(line['cat'])
    T_id = Tlist.index(str(T))
    return cats[T_id],JTs[T_id]




for j in range(0):
    if j > 0:
        folder0 = f'rf/0+_ZZ0/'
        if not os.path.exists(folder0):
            os.mkdir(folder0)
    #print('-'*10+str(T)+'-'*10)
    files = os.listdir(f'rf/0+_ZZ{j}/')
    Tlist = [float(file) for file in files]
    for T in Tlist:
        cats = cat_dict[T]
        for i in range(16):
            folder = f'rf/0+_ZZ{j}/{T}/{i}/'
            cat_i,JT_i,varX_i,varN_i = cat_res(folder,T)
            #print(f'{j}_{T}_{i}: {cat_i}, {JT_i}, {varX_i}, {varN_i}')
            if cat_i in cats:
                print(f'{j}_{T}_{i}: {cat_i}, {JT_i}, {varX_i}, {varN_i}')
                if j > 0:
                    if not os.path.exists(folder0+f'{T}/'):
                        os.mkdir(folder0+f'{T}/')
                    if os.path.exists(folder0+f'{T}/{i}/'):
                        k = random.randint(33,1000)
                        while os.path.exists(folder0+f'{T}/{k}/'):
                            k += 1
                    else:
                        k = i
                    shutil.copytree(folder,folder0+f'{T}/{k}/')
            else:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
        if len(os.listdir(f'rf/0+_ZZ{j}/{T}/')) == 0:os.rmdir(f'rf/0+_ZZ{j}/{T}/')