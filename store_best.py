import os ,numpy as np,shutil,matplotlib.pyplot as plt,localTools,read_config,subprocess,csv,QSL_search,re,read_write
from scipy.interpolate import interp1d
from J_T_local import bestCat

canoLabel = ['0+','1+','2+','3+','4+','5+','6+','7+']

def store(source_folder,tgt_folder_T):
    if os.path.exists(tgt_folder_T):
        shutil.rmtree(tgt_folder_T)
    shutil.copytree(source_folder,tgt_folder_T)

def iter_plot(working_folder):
    if not os.path.exists(working_folder+'/oct_iters.dat'):
        print(working_folder)
        return [1.02,1.01,1.0],[0,0,0],[1,1,1]
    iters,JTs = np.split(np.loadtxt(working_folder+'/oct_iters.dat', usecols=(0, 1)), 2, axis=1)
    files = os.listdir(working_folder)
    lambda_recode = []
    if any(['former_opt_' in file for file in files]):
        c_opt_label = 1
        all_iters = []
        all_JTs = []
        while os.path.exists(working_folder+f'former_opt_{c_opt_label}/'):
            iters_i,JTs_i = np.split(np.loadtxt(working_folder+f'former_opt_{c_opt_label}/oct_iters.dat', usecols=(0, 1)), 2, axis=1)
            if c_opt_label == 1:loc_0=0
            else:loc_0=1
            all_iters += list(len(all_iters) + iters_i.T[0])[loc_0:]
            all_JTs += list(JTs_i.T[0])[loc_0:]
            lambda_recode.append([len(all_iters),read_config.read_oct_lambda_a(working_folder+f'former_opt_{c_opt_label}/')])
            c_opt_label += 1
        all_iters += list(len(all_iters) + iters.T[0])[1:]
        all_JTs += list(JTs.T[0])[1:]
        lambda_recode.append([len(all_iters),read_config.read_oct_lambda_a(working_folder)])
        return all_iters,all_JTs,lambda_recode
    lambda_recode.append([len(iters.T[0]),read_config.read_oct_lambda_a(working_folder)])
    return iters.T[0],JTs.T[0],lambda_recode

def plot_stored_results(tgt_folder,task_name):
    folders = os.listdir(tgt_folder)
    Ts = []
    for folder in folders:
        try:
            Ts.append(float(folder))
        except ValueError:
            pass
    Ts.sort()
    oct_info = [localTools.read_oct_iters(tgt_folder + f'{T}/',True) for T in Ts]
    for T in Ts:
        files = os.listdir(tgt_folder + f'{T}/')
        if not any(['control_plot_' in file for file in files]):
            try:
                localTools.frequency_monitor(tgt_folder + f'{T}/',T)
            except ValueError:
                pass
    delta_JTs = [oct_info[i][1] for i in range(len(Ts))]
    JTs = [oct_info[i][0] for i in range(len(Ts))]
    data_dict = [{'T':Ts[i],'JT':JTs[i],'dJT':delta_JTs[i]} for i in range(len(Ts))]
    with open(f'{tgt_folder}/{task_name}.csv', 'w', newline='') as csvfile:
        fieldnames = ['T','JT','dJT']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dict)
    plt.plot(Ts,JTs)
    plt.plot(Ts,JTs,'.')
    plt.plot([min(Ts),max(Ts)],[0.01]*2,'--',alpha=0.5)
    for i in range(len(Ts)):
        if JTs[i] > 1e-2:plt.plot([Ts[i],Ts[i]],[JTs[i],JTs[i]-delta_JTs[i]*10],'b--',alpha=0.5)
    plt.savefig(tgt_folder+f'{task_name}.pdf')
    plt.clf()
    with open(tgt_folder+'summary.log','a') as log_f:
        log_f.write(f'\nAll stored results:\nT{task_name} = {Ts}\nJT{task_name} = {JTs}\n')
        for T,JT,delta_JT in zip(Ts,JTs,delta_JTs):
            log_f.write(f'{T} {JT} {delta_JT}\n')

def compare_store(source_folders,tgt_folder,task_name):
    if not os.path.exists(tgt_folder):os.mkdir(tgt_folder)
    text_log = ''
    JTs = []
    Tgrid = []
    for src_f in source_folders:
        if os.path.exists(src_f):
            sub_fodlers = os.listdir(src_f)
            for folder in sub_fodlers:
                try:
                    this_T = float(folder)
                    if not this_T in Tgrid:
                        Tgrid.append(this_T)
                except ValueError:
                    pass
    Tgrid.sort()
    valid_Tgrid = []
    for T in Tgrid:
        min_JT_i = []
        min_id_i = []
        for i in range(len(source_folders)):
            cur_wd = source_folders[i]+f'{T}/'
            if os.path.exists(cur_wd):
                contents = os.listdir(cur_wd)
                if os.path.isdir(cur_wd+contents[0]):
                    this_JTs = [localTools.read_oct_iters(cur_wd+contents[j]) for j in range(len(contents))]
                    min_JT_i.append(min(this_JTs))
                    min_id_i.append(int(contents[this_JTs.index(min_JT_i[-1])]))
                else:
                    min_JT_i.append(localTools.read_oct_iters(cur_wd))
                    min_id_i.append(True)
            else:
                min_JT_i.append(np.inf)
                min_id_i.append(False)
        if not all([isinstance(min_id_i[i],bool) and min_id_i[i] == False for i in range(len(source_folders))]):
            valid_Tgrid.append(T)
            JTs.append(min(min_JT_i))
            best_source_id = min_JT_i.index(JTs[-1])
            best_source_folder = f'{source_folders[best_source_id]}{T}/'
            text_log += f'Best result for T = {T} occurs at {best_source_folder}'
            if isinstance(min_id_i[best_source_id],bool):
                text_log += '\n'
            else:
                best_source_folder += f'{min_id_i[best_source_id]}/'
                text_log += f'{min_id_i[best_source_id]}/\n'
            tgt_folder_T = tgt_folder + f'{T}/'
            if os.path.exists(tgt_folder_T+'oct_iters.dat'):
                former_result = localTools.read_oct_iters(tgt_folder_T)
                if former_result > JTs[-1]:
                    text_log += f'Former result of T = {T}: {former_result}\n'
                    store(best_source_folder,tgt_folder_T)
                else:text_log += f'T = {T}: original JT {former_result} <= new JT {JTs[-1]}, pass\n'
            else:store(best_source_folder,tgt_folder_T)
    with open(tgt_folder+'summary.log','w') as log_f:
        log_f.write(f'source_folders: {source_folders}\n')
        log_f.write(f'T{task_name} = {valid_Tgrid}\nJT{task_name} = {JTs}\n')
        log_f.write(text_log)
        for T,JT in zip(valid_Tgrid,JTs):
            log_f.write(f'{T} {JT}\n')
    plot_stored_results(tgt_folder,task_name)
    return 0

def ax_lambda(ax_i,lambda_record_best,id_lambda_record):
    x0 = ax_i.lines[0].get_xdata()[0]
    y0 = ax_i.lines[0].get_ydata()
    y0_min = min(y0)
    y0_max = max(y0)
    y1 = ax_i.lines[1].get_ydata()
    y1_min = min(y1)
    y1_max = max(y1)
    for i in range(len(id_lambda_record)):
        if i == 0:ax_i.text(x0,y0_max,rf"$\lambda_n$:{id_lambda_record[i][1]}")
        else:ax_i.text(id_lambda_record[i-1][0]+10,y0_max,rf"$\lambda_n$:{id_lambda_record[i][1]}")
        ax_i.plot([id_lambda_record[i][0],id_lambda_record[i][0]],[y0_max,y0_min],'r--',alpha=0.1)
    for i in range(len(lambda_record_best)):
        if i==0:ax_i.text(x0,y1_min,rf"$\lambda_s$:{lambda_record_best[i][1]}")
        else:ax_i.text(lambda_record_best[i-1][0]+10,y1_min,rf"$\lambda_s$:{lambda_record_best[i][1]}")
        ax_i.plot([lambda_record_best[i][0],lambda_record_best[i][0]],[y1_max,y1_min],'b--',alpha=0.1)

def plot_opt_iter(iter_best,JTs_best,lambda_record_best,new_opt_folder,denote_lambda):
    id_iter,id_JTs,id_lambda_record = iter_plot(new_opt_folder)
    p05_val = max(id_JTs[-1],JTs_best[-1])*1.15
    p05_loc = localTools.binary_search(id_JTs,p05_val,new_opt_folder)
    dot_flag = False
    if JTs_best[-1]<id_JTs[-1]:
        step = localTools.binary_search(JTs_best,id_JTs[-1],'JT_best')
        dot_flag = True
    fig, axs = plt.subplots(4,1, figsize=(10, 8))
    ax0=axs[0]
    ax0.plot(iter_best,JTs_best,label='best')
    ax0.plot(id_iter,id_JTs,'--',label='new')
    ax0.set_ylabel('JT')
    ax1=axs[1]
    grad_best = np.log10([JTs_best[i]-JTs_best[i+1] for i in range(len(JTs_best)-1)])
    ax1.plot(iter_best[1:],grad_best,label = 'best')
    grad_new = np.log10([id_JTs[i]-id_JTs[i+1] for i in range(len(id_JTs)-1)])
    ax1.plot(id_iter[1:],grad_new,'--',label='new')
    if dot_flag: 
        ax0.plot([iter_best[step]],[JTs_best[step]],'*',label = 'best[i] == new[-1]')
        ax1.plot([iter_best[step]],[grad_best[step-1]],'*',label = 'best[i] == new[-1]')
    ax0.legend(loc='best')
    ax1.set_ylabel('lg(delta_JT)')
    if denote_lambda:
        ax_lambda(ax0,lambda_record_best,id_lambda_record)
        ax_lambda(ax1,lambda_record_best,id_lambda_record)
    id_iter=id_iter[p05_loc:]
    id_JTs=id_JTs[p05_loc:]
    seg_label = localTools.binary_search(id_lambda_record[:][0],p05_loc)
    id_lambda_record=id_lambda_record[seg_label:]
    if p05_loc<len(iter_best):
        iter_best=iter_best[p05_loc:]
        JTs_best=JTs_best[p05_loc:]
        seg_label = localTools.binary_search(lambda_record_best[:][0],p05_loc)
        lambda_record_best=lambda_record_best[seg_label:]
    ax2=axs[2]
    ax2.plot(iter_best[:min(2*len(id_iter),len(iter_best))],JTs_best[:min(2*len(id_iter),len(iter_best))],label='best')
    ax2.plot(id_iter,id_JTs,'--',label='new')
    ax2.set_ylabel('JT')
    ax3=axs[3]
    grad_best = np.log10([JTs_best[i]-JTs_best[i+1] for i in range(len(JTs_best)-1)])
    ax3.plot(iter_best[1:min(2*len(id_iter),len(iter_best))],grad_best[:min(2*len(id_iter),len(iter_best))-1],label = 'best')
    grad_new = np.log10([id_JTs[i]-id_JTs[i+1] for i in range(len(id_JTs)-1)])
    ax3.plot(id_iter[1:],grad_new,'--',label='new')
    ax3.set_ylabel('lg(delta_JT)')
    ax3.set_xlabel('iter')
    if denote_lambda:
        ax_lambda(ax2,lambda_record_best,id_lambda_record)
        ax_lambda(ax3,lambda_record_best,id_lambda_record)
    plt.savefig(new_opt_folder+f'/compare_opt.pdf')
    plt.clf()
    plt.close()

def find_grad(best_res_fit,JT,n_iters):
    step = 1
    if best_res_fit(n_iters) > JT: return ''
    #step = localTools.binary_search(best_res_fit,JT)
    while best_res_fit(step) > JT:step += 1
    return (best_res_fit(step) - best_res_fit(step+1))/best_res_fit(step)

def grad_comment(grad_best,grad_new):
    if isinstance(grad_best,str):return 'JT of the new result are better than the stored result'
    text_content = 'the new result might be '
    rate = grad_new / grad_best
    categories = [
        (10, "Outstanding"),
        (5, "Impressive"),
        (2, "Favorable"),
        (1, "Neutral"),
        (1/2, "Poor"),
        (1/5, "Dismal"),
        (1/10, "Catastrophic"),
        (0, "Abyssal")]
    for threshold, adjective in categories:
        if rate >= threshold:
            return text_content + f'{adjective} compared with the stored result (>={threshold})'

def empty_folder(folder):
    files = os.listdir(folder)
    if 'oct_iters.dat' not in files:return True
    else: return False

def c_opt_iter_stop(grads):
    if isinstance(grads,list):
        if len(grads) == 0:return 0
        grads = max(grads)
    if grads > 2e-3:return 1500
    elif grads > 1e-3:return 1000
    elif grads > 8e-4:return 800
    elif grads > 6e-4:return 600
    elif grads > 2e-4:return 500
    else: return 0

def separate_strings(s):
    match  = re.match(r"([a-zA-Z]+)(\d+)",s)
    if match:return match.groups()
    else: return None, None

def compare_with_best(source_folders,tgt_folder,denote_lambda=True,c_opt=False):
    text_logs = [f'log of {source_folders[i]}\n' for i in range(len(source_folders))]
    Tgrid = []
    for src_f in source_folders:
        if os.path.exists(src_f):
            sub_fodlers = os.listdir(src_f)
            for folder in sub_fodlers:
                try:
                    this_T = float(folder)
                    if not this_T in Tgrid:
                        Tgrid.append(this_T)
                except ValueError:
                    pass
    Tgrid.sort()
    for T in Tgrid:
        for i in range(len(source_folders)):
            str_T = str(T)
            tgt_folder_T = tgt_folder + f'{T}/'
            #print(tgt_folder_T)
            iter_best, JTs_best,lambda_record_best = iter_plot(tgt_folder_T)
            bets_res_fit = interp1d(iter_best, JTs_best, kind="cubic", fill_value="extrapolate")
            best_res_JT,best_res_delta_JT = localTools.read_oct_iters(tgt_folder_T,True)
            cur_wd = source_folders[i]+f'{T}/'
            if os.path.exists(cur_wd):
                text_logs[i] += '-'*20 + '\n' + '-' * int(np.floor(10-len(str_T)/2)) + str_T + '-' * int(np.ceil(10-len(str_T)/2)) +  '\n'+ '-'*20+'\n'
                text_logs[i] += f'Stored result info:\n n_iter {iter_best[-1]}, JT {best_res_JT}, delta_JT {best_res_delta_JT}\n'
                read_data = ''
                contents = os.listdir(cur_wd)
                grads = []
                grads_r = []
                c_opt_list = []
                if os.path.isdir(cur_wd+contents[0]):
                    new_contents = []
                    for content in contents:
                        if not empty_folder(cur_wd+content):new_contents.append(content)
                    contents = new_contents
                    #print(cur_wd)
                    if len(contents):
                        lambda_ai = read_config.read_oct_lambda_a(cur_wd+contents[0]+'/')
                        this_JTs = [localTools.read_oct_iters(cur_wd+contents[j],True) for j in range(len(contents))]
                        all_info = sorted([[contents[j],this_JTs[j][0],this_JTs[j][1],find_grad(bets_res_fit,this_JTs[j][0],iter_best[-1])] for j in range(len(contents))], key=lambda x:x[1])
                        for j in range(len(contents)):
                            if any([all([all_info[j][1] <= JTs_best[-1], all_info[j][2] > 2e-4]),all_info[j][2] >= all_info[j][3] * 0.95]) and all_info[j][1] > 9.8e-3:
                                c_opt_list.append(f'{cur_wd}{all_info[j][0]}/')
                                grads.append(all_info[j][2])
                                grads_r.append(all_info[j][2]/all_info[j][3])
                            read_data += f'{all_info[j][0]}: {all_info[j][1]}, {grad_comment(all_info[j][3],all_info[j][2])},\n\t\t\t\tgrad: {all_info[j][2]}, best_res_grad: {all_info[j][3]}\n'
                            plot_opt_iter(iter_best,JTs_best,lambda_record_best,cur_wd+all_info[j][0]+'/',denote_lambda)
                else:
                    if not empty_folder(cur_wd):
                        lambda_ai = read_config.read_oct_lambda_a(cur_wd)
                        this_JT = localTools.read_oct_iters(cur_wd,True)
                        read_data += f'{cur_wd}: {this_JT[0]}, {this_JT[1]}\n'
                        plot_opt_iter(iter_best,JTs_best,lambda_record_best,cur_wd,denote_lambda)
                        best_grad = find_grad(bets_res_fit,this_JT[0],iter_best[-1])
                        if this_JT[1] >= best_grad * 0.95 and this_JT[0] > 9.8e-3:
                            c_opt_list.append(cur_wd)
                            grads.append(this_JT[1])
                            grads_r.append(this_JT[1]/best_grad)
                iter_stop = c_opt_iter_stop(grads)
                if iter_stop and c_opt:
                    redo_info = c_opt_list[0].split('/')[1].split('_')
                    if redo_info[0][0] == 'D':
                        dissipation = True
                        gate_name = redo_info[0][1:]
                    else:
                        dissipation = False
                        gate_name = redo_info[0]
                    logical_qubit,_ = separate_strings(redo_info[1])
                    if gate_name not in ['l0','l1'] and gate_name not in canoLabel:
                        QSL_search.gate_task([T],gate_name,logical_qubit,iter_stop,lambda_a = None,dissipation = dissipation, control_source = c_opt_list)
                    else:
                        if gate_name in ['l0','l1']:
                            num_qubit = 1
                            Ham_num = 7
                            JT = 0
                        else:
                            num_qubit = 4
                            Ham_num = 3
                            JT = 1
                        QSL_search.state_task([T],num_qubit,Ham_num,logical_qubit,gate_name,JT,iter_stop,lambda_a = None,dissipation = dissipation, control_source = c_opt_list)
                    text_logs[i] += f'redo info: iter_stop = {iter_stop},\n'
                    for j in range(len(c_opt_list)):
                        text_logs[i] += f'{c_opt_list[j].split('/')[-2]}, '
                        #text_logs[i] += f'{c_opt_list[j]},\n{grads[j]}\n{grads_r[j]}\n'
                    text_logs[i] += '\n'
                text_logs[i] += f'lambda_a: {lambda_ai}\n' + read_data
    for i in range(len(source_folders)):
        with open(source_folders[i]+'compare_summary.log','w') as log_f:
            log_f.write(text_logs[i])
    return 0

def del_former(src):
    folders = os.listdir(src)
    for folder in folders:
        if '.log' not in folder:
            path_sub = src + folder + '/'
            ff_names = os.listdir(path_sub)
            for ff_name in ff_names:
                if 'former_opt' in ff_name:
                    subprocess.run(['rm','-r',path_sub+ff_name]) 

def cat_res(folder):
    items = os.listdir(folder)
    text_content = ''
    for item in items:
        if all(['.log' not in item, '.pdf' not in item]):
            sub_items = os.listdir(folder + item)
            for sub_item in sub_items:
                state = read_write.stateReader(f'{folder}{item}/{sub_item}/psi_final_after_oct.dat',16)
                state = localTools.rotate_state(state,4,0,float(item))
                cat_inFidelity = bestCat(state)
                min_JTss = cat_inFidelity
                text_content += f'JT_ss: {min_JTss}'
    with open(folder+'cat.log','w') as log_f:log_f.write(text_content)

def best_cat_at_T():
    canoLabels = ['0+','1+','2+','4+','7+']
    import csv
    source_main = f'store_result_ZZ/'
    csv_names = [source_main + f'{canoLabel}_ZZ/{canoLabel}_ZZ.csv' for canoLabel in canoLabels]
    Tlist = []
    JT_cats = []
    for canoLabel,csv_name in zip(canoLabels,csv_names):
        with open(csv_name, mode ='r') as file:    
            this_cats = []
            csvFile = csv.DictReader(file)
            for line in csvFile:
                this_cats.append(float(line['JT']))
            JT_cats.append(this_cats)
        if canoLabel == '0+':
            with open(csv_name, mode ='r') as file:
                csvFile = csv.DictReader(file)
                for line in csvFile:
                    Tlist.append(float(line['T']))
    JT_cats=list(np.array(JT_cats).T)
    new_JT_cat = []
    for i in range(len(JT_cats)):
        new_JT_cat.append(list(JT_cats[i]))
    JT_cats = new_JT_cat
    data_dict = []
    for i in range(len(Tlist)):
        JT_at_T = min(JT_cats[:][i])
        cat_at_T = canoLabels[JT_cats[:][i].index(JT_at_T)]
        data_dict.append({'T':Tlist[i],'JT':JT_at_T,'cat':cat_at_T})
    with open(f'store_result_ZZ/cat_at_T.csv', 'w', newline='') as csvfile:
        fieldnames = ['T','JT','cat']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dict)

q1_gates = ['H','T','X','Z']
q2_gates = ['BG','CX','CZ','siS']
dq1_gates = ['DH','DT','DX','DZ']
dq2_gates = ['DBG','DCX','DCZ','DsiS']
canoLabel = ['0+','1+','2+','3+','4+','5+','6+','7+']
canoLabel_F = ['0+','1+','2+','4+','7+']
logical_states = ['l0','l1']
dls = ['Dl0','Dl1']
logical_qubits = ['GKP','cat','bnm','ZZ']
logical_qubit = logical_qubits[:3]
gate_names = q2_gates
#gate_names = ['siS']
if isinstance(logical_qubit,str):logical_qubit = [logical_qubit]
if isinstance(gate_names,str):gate_names = [gate_names]
task_ids = [1,2,3,4]
for gate_name in gate_names:
    for lg_q in logical_qubit:
        all_folders = os.listdir('rf/')
        #all_folders = []
        source_folders = []
        for each_folder in all_folders:
            keywords = each_folder.split('_')
            if all([gate_name == keywords[0],lg_q in keywords[1]]):source_folders.append('rf/'+each_folder+'/')
        print(source_folders)
        #task_name = f'{gate_name}_{lg_q}'
        #tgt_folder = f'store_result_{lg_q}/{task_name}/'
        #if gate_name[0] == 'D':tgt_folder = f'store_result_{lg_q}_D/{task_name}/'
        #compare_store(source_folders,tgt_folder,task_name)
        if len(source_folders):
            task_name = f'{gate_name}_{lg_q}'
            #source_folders = [f'rf/{task_name}{i}/' for i in task_ids]
            tgt_folder = f'store_result_{lg_q}/{task_name}/'
            if gate_name[0] == 'D':
                tgt_folder = f'store_result_{lg_q}_D/{task_name}/'
            #src =source_folders[0]
            #del_former(src)
            compare_store(source_folders,tgt_folder,task_name)
            #if gate_name in q2_gates:
            compare_with_best(source_folders,tgt_folder,denote_lambda=False,c_opt=True)


