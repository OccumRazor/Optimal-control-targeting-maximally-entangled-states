import os,matplotlib.pyplot as plt,localTools,numpy as np,J_T_local,csv,read_write
from scipy.interpolate import interp1d


line_style_fit = {'0+':['darkblue','--',''],'1+':['Orange','--',''],'2+':['darkred','--',''],'3+':['darkorange','--',''],'4+':['darkcyan','--',''],'5+':['Brown','--',''],'6+':['orchid','--',''],'7+':['Black','--','']}
line_style_num = {'0+':['darkblue','','*'],'1+':['Orange','','+'],'2+':['darkred','','x'],'3+':['darkorange','','x'],'4+':['darkcyan','','1'],'5+':['Brown','','2'],'6+':['orchid','','h'],'7+':['Black','','.'],'XN':['limegreen','','^']}

def cano_XN(goal):
    if goal == 'cano_s':
        #Tlist = [20.01,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,30.93,32.0,33.0,34.0,35.0,36.03,37.0,38.0,39.0,40.03,40.98,41.95,43.02,44.03,45.0,46.0,47.0,48.0,48.97,50.03]
        Tlist = [20.01,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,30.93,32.0,33.0,34.0,35.0,36.03,37.0,38.0,39.0,40.03,40.98,41.95,43.02,44.03,45.0,46.0,47.0,48.0,48.97,50.03]
    if goal == 'cano_f':
        Tlist = [20.75, 20.76, 20.77, 20.78, 20.79, 20.8, 20.81, 20.82, 20.83, 20.84, 20.85, 20.86, 20.87, 20.88, 20.89, 20.9, 20.91, 20.92, 20.93, 20.94, 20.95, 20.96, 20.97, 20.98, 20.99, 21.0, 21.01, 21.017, 21.02, 21.03, 21.04, 21.05, 21.06, 21.07, 21.08, 21.09, 21.1, 21.11, 21.12, 21.13, 21.14, 21.15, 21.16, 21.17, 21.18, 21.19, 21.2, 21.21, 21.22, 21.23, 21.24, 21.25]
    Tgrid = []
    JT_ss = []
    for T in Tlist:
        if os.path.exists(f'store_result_XN/{T}/'):
            _,_,current_JT_ss_m,_,_ = J_T_local.cat_res(f'store_result_XN/{T}/',T,'0+')
            Tgrid.append(T)
            JT_ss.append(current_JT_ss_m)
    return Tgrid,JT_ss

def cano_fit(Tgrid):
    freqs,center_locs = localTools.cano_freq()
    Tgrid = np.array(Tgrid)
    JT_T = []
    for T in Tgrid:
        JT_T.append(min([
            0.5-np.abs(np.cos(freqs[i]*(T - center_locs[i])))*0.5
        for i in range(8)]))
    fig,ax = plt.subplots()
    ax.plot(Tgrid,JT_T,label='min')
    ax.set_ylabel('inFidelity')
    #ax1=ax.twinx()
    #ax1.set_ylabel(r'$\phi$/$\pi$')
    for i in [3,5,6]:
        canoLabel = f'{i}+'
        ax.plot(Tgrid,0.5-np.abs(np.cos(freqs[i]*(Tgrid - center_locs[i])))*0.5,label=rf'$|{canoLabel}\rangle$')
        #x1.plot(Tgrid,localTools.cat_phi(Tgrid,canoLabel),'-.',color=color)
    ax.legend(loc='best')
    ax.set_xlabel('T')
    plt.savefig('store_result_ZZ/predict.pdf')


def plot_stored_results(tgt_folder,logical_qubit,goals):
    if goals == 'cano_f': gate_names = ['0+','1+','2+','4+','7+']
    elif goals == 'cano_s': gate_names = ['3+','5+','6+']
    elif isinstance(goals,str):gate_names = [goals]
    if 'cano' in goals: 
        Tgrid, JT_ss = cano_XN(goal)
        plt.plot(Tgrid,JT_ss,label='XN',color=line_style_num['XN'][0],linestyle=line_style_num['XN'][1],marker=line_style_num['XN'][2])
    for gate_name in gate_names:
        folders = os.listdir(tgt_folder+f'{gate_name}_{logical_qubit}/')
        Ts = []
        for folder in folders:
            try:
                Ts.append(float(folder))
            except ValueError:
                pass
        Ts.sort()
        if gate_name == '6+' and goals == 'cano_s':
            new_T = []
            allow_list_6 = [20.0,20.95]
            for T in Ts:
                if T < 22.0 and T not in allow_list_6:
                    pass
                else:
                    new_T.append(T)
            Ts = new_T
        JTs = [localTools.read_oct_iters(tgt_folder + f'{gate_name}_{logical_qubit}/{T}/',True) for T in Ts]
        JTs = [JTs[i][0] for i in range(len(Ts))]
        if 'cano' in goals:
            fit_T = np.linspace(Ts[0],Ts[-1],100*len(Ts))
            plt.plot(fit_T,cano_fit_id(fit_T,gate_name),color=line_style_fit[gate_name][0],linestyle=line_style_fit[gate_name][1],marker=line_style_fit[gate_name][2])
            plt.plot(Ts,JTs,label = rf'|{gate_name}$\rangle$',color=line_style_num[gate_name][0],linestyle=line_style_num[gate_name][1],marker=line_style_num[gate_name][2])
        else:
            plt.plot(Ts,JTs,label = gate_name,color=line_style_num[gate_name][0],linestyle=line_style_num[gate_name][1],marker=line_style_num[gate_name][2])
        #plt.plot(Ts,JTs,'.')
    if goals == 'cano_f':
        folders = os.listdir(tgt_folder+f'6+_{logical_qubit}/')
        Ts = []
        for folder in folders:
            try:
                Ts.append(float(folder))
            except ValueError:
                pass
        Ts.sort()
        Tlist = []
        for T in Ts:
            if T <= 21.25 and T >= 20.75:Tlist.append(T)
        JTs = [localTools.read_oct_iters(tgt_folder + f'6+_{logical_qubit}/{T}/',True) for T in Tlist]
        JTs = [JTs[i][0] for i in range(len(Tlist))]
        plt.plot(Tlist,JTs,label = r'|6+$\rangle$',color=line_style_fit['6+'][0],linestyle=line_style_fit['6+'][1],marker=line_style_fit['6+'][2])
    csv_name = 'store_result_ZZ/cat_at_T.csv'
    with open(csv_name, mode ='r') as file:
        this_cats = []
        Tlist = []    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            T = float(line['T'])
            if T >= 20.75 and T <= 21.25:
                Tlist.append(float(line['T']))
                this_cats.append(1.05*float(line['JT']))
    #plt.plot(Tlist,this_cats,'--')
    plt.legend(loc='best')
    #plt.plot([min(Ts),max(Ts)],[0.01]*2,'--',alpha=0.5)
    plt.xlabel('ns')
    plt.ylabel('inFidelity')
    if goals == 'cano_s': goals = 'cano_2nd'
    if goals == 'cano_f': goals = 'cano_1st'
    plt.savefig(tgt_folder+f'{logical_qubit}{goals}.pdf')
    plt.clf()



def cano_f_all():
    csv_name = 'store_result_ZZ/6+_ZZ/6+_ZZ.csv'
    with open(csv_name, mode ='r') as file:
        this_cats = []
        Tlist = []    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            Tlist.append(float(line['T']))
            this_cats.append(float(line['JT']))
    fit_cats_s = interp1d(Tlist, this_cats, kind="cubic", fill_value="extrapolate")
    cats_f = []
    canoLabels = ['0+','1+','2+','4+','7+']
    for canoLabel in canoLabels:
        csv_name = f'store_result_ZZ/{canoLabel}_ZZ/{canoLabel}_ZZ.csv'
        with open(csv_name, mode ='r') as file:
            this_cats = []
            Tlist = []    
            csvFile = csv.DictReader(file)
            for line in csvFile:
                Tlist.append(float(line['T']))
                this_cats.append(float(line['JT']))
        cats_f.append(interp1d(Tlist, this_cats, kind="cubic", fill_value="extrapolate"))
    Tlist = np.linspace(20.75,21.25,51)
    text_content = ''
    for i in range(len(Tlist)):
        text_content += f'{Tlist[i]}\n'
    text_content += '-' * 20 + '\n'
    for i in range(5):
        for T in Tlist:
            text_content += f'{cats_f[i](T)}\n'
        text_content += '-' * 20 + '\n'
    for T in Tlist:
        text_content += f'{fit_cats_s(T)}\n'
    with open(f'store_result_ZZ/cats_at_T.log', 'w') as log_f:
        log_f.write(text_content)
    data_dict = []
    for T in Tlist:
        data_dict.append({'T':T,'0+':cats_f[0](T),'1+':cats_f[1](T),'2+':cats_f[2](T),'4+':cats_f[3](T),'6+':fit_cats_s(T),'7+':cats_f[4](T)})
    fieldnames = ['T','0+','1+','2+','4+','6+','7+']
    with open(f'store_result_ZZ/cats_at_T.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dict)
    Tlist=np.linspace(20.75,21.25,501)
    for i in range(len(canoLabels)):
        plt.plot(Tlist,cats_f[i](Tlist),label=canoLabels[i])
    plt.plot(Tlist,fit_cats_s(Tlist),label='6+')
    plt.legend(loc='best')
    #plt.plot([min(Ts),max(Ts)],[0.01]*2,'--',alpha=0.5)
    plt.xlabel('ns')
    plt.ylabel('inFidelity')
    plt.savefig('store_result_ZZ/ZZcano_f.pdf')
    plt.clf()

def plt_rf_res(rf):
    files = os.listdir(rf)
    pop_list = ['res.pdf','res_oct_iter.pdf','res_with_phase.pdf']
    for candidate in pop_list:
        if candidate in files:
            files.pop(files.index(candidate))
    Tlist = [float(file) for file in files]
    Tlist.sort()
    JT = [localTools.read_oct_iters(f'{rf}{T}/0/',False) for T in Tlist]
    plt.plot(Tlist,JT,label='JT')
    #plt.plot(Tlist,phase,label=r'$\phi$/$\pi$')
    plt.legend(loc='best')
    plt.savefig(rf+'res_oct_iter.pdf')
    plt.clf()
    JT = []
    phase = []
    for T in Tlist:
        state = read_write.stateReader(f'{rf}{T}/0/psi_final_after_oct.dat',2**4)
        state = localTools.rotate_state(state,4,0,T)
        JT_T,phase_T = J_T_local.bestCat_phase(state,False,True)
        JT.append(JT_T)
        if phase_T > 1: phase_T -= 2
        elif phase_T < -1: phase_T += 2
        phase.append(phase_T)
    plt.plot(Tlist,JT,label='JT')
    plt.plot(Tlist,phase,label=r'$\phi$/$\pi$')
    plt.legend(loc='best')
    plt.savefig(rf+'res_with_phase.pdf')
    plt.clf()

logical_qubits = ['ZZ']
Tgrid = np.linspace(20.0,50.0,10001)
for goal in ['cano_f']:
    for logical_qubit in logical_qubits:
        if goal[0]=='d':tgt_folder = f'store_result_{logical_qubit}_D/'
        else:tgt_folder = f'store_result_{logical_qubit}/'
        plot_stored_results(tgt_folder,logical_qubit,goal)