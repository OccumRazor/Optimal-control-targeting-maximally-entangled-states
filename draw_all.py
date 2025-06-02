import matplotlib.pyplot as plt,localTools,numpy as np,csv,matplotlib

font = {'size'   : 15}
matplotlib.rc('font', **font)

line_style_fit = {'0+':['darkblue','--',''],'1+':['Orange','--',''],'2+':['darkred','--',''],'3+':['darkorange','--',''],'4+':['darkcyan','--',''],'5+':['Brown','--',''],'6+':['orchid','--',''],'7+':['Black','--','']}
line_style_num = {'0+':['darkblue','','*'],'1+':['Orange','','+'],'2+':['darkred','','x'],'3+':['darkorange','','x'],'4+':['darkcyan','','1'],'5+':['Brown','','2'],'6+':['orchid','','h'],'7+':['Black','','.'],'XN':['limegreen','','^']}

def read_dict(canoLabe,goal):
    if goal == 'cano_s':
        T_min = 20.01
        T_max = 50.03
        #Tlist = [20.01,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,30.93,32.0,33.0,34.0,35.0,36.03,37.0,38.0,39.0,40.03,40.98,41.95,43.02,44.03,45.0,46.0,47.0,48.0,48.97,50.03]
    if goal == 'cano_f':
        T_min = 20.75
        T_max = 21.25
        #Tlist = [20.75, 20.76, 20.77, 20.78, 20.79, 20.8, 20.81, 20.82, 20.83, 20.84, 20.85, 20.86, 20.87, 20.88, 20.89, 20.9, 20.91, 20.92, 20.93, 20.94, 20.95, 20.96, 20.97, 20.98, 20.99, 21.0, 21.01, 21.017, 21.02, 21.03, 21.04, 21.05, 21.06, 21.07, 21.08, 21.09, 21.1, 21.11, 21.12, 21.13, 21.14, 21.15, 21.16, 21.17, 21.18, 21.19, 21.2, 21.21, 21.22, 21.23, 21.24, 21.25]
    if canoLabe=='XN':csv_name = 'data/XN.csv'
    else:csv_name = f'data/{canoLabe}.csv'
    JT_ss = []
    Tlist = []
    with open(csv_name, mode ='r') as file:
        csvFile = csv.DictReader(file)
        for line in csvFile:
            T = float(line['T'])
            if T >= T_min and T <= T_max:
                Tlist.append(T)
                JT_ss.append(float(line['JT']))
    return Tlist,JT_ss

def cano_fit_id(Tgrid,canoLabel):
    freqs,center_locs = localTools.cano_freq()
    cano_id = int(canoLabel[0])
    Tgrid = np.array(Tgrid)
    freq = freqs[cano_id]
    center_loc = center_locs[cano_id]
    JT_T = 0.5-np.abs(np.cos(freq*(Tgrid - center_loc)))*0.5
    return JT_T

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
    ax.set_ylabel('infidelity')
    for i in [3,5,6]:
        canoLabel = f'{i}+'
        ax.plot(Tgrid,0.5-np.abs(np.cos(freqs[i]*(Tgrid - center_locs[i])))*0.5,label=rf'$|{canoLabel}\rangle$')
    ax.legend(loc='best')
    ax.set_xlabel('ns')
    plt.savefig('data/predict.pdf')
    plt.clf()

def plot_stored_results(goals):
    if goals == 'cano_f': canoLabels = ['0+','1+','2+','4+','7+']
    elif goals == 'cano_s': canoLabels = ['3+','5+','6+']
    elif isinstance(goals,str):canoLabels = [goals]
    Tgrid, JT_ss = read_dict('XN',goals)
    plt.plot(Tgrid,JT_ss,label='XN',color=line_style_num['XN'][0],linestyle=line_style_num['XN'][1],marker=line_style_num['XN'][2])
    for canoLabel in canoLabels:
        Ts,JTs = read_dict(canoLabel,goals)
        fit_T = np.linspace(Ts[0],Ts[-1],100*len(Ts))
        plt.plot(fit_T,cano_fit_id(fit_T,canoLabel),color=line_style_fit[canoLabel][0],linestyle=line_style_fit[canoLabel][1],marker=line_style_fit[canoLabel][2])
        plt.plot(Ts,JTs,label = rf'|{canoLabel}$\rangle$',color=line_style_num[canoLabel][0],linestyle=line_style_num[canoLabel][1],marker=line_style_num[canoLabel][2])
    if goals == 'cano_f':
        Ts,JTs = read_dict('6+',goals)
        plt.plot(Ts,JTs,label = r'|6+$\rangle$',color=line_style_fit['6+'][0],linestyle=line_style_fit['6+'][1],marker=line_style_fit['6+'][2])
    plt.legend(loc='best')
    plt.xlabel('ns')
    plt.ylabel('infidelity')
    if goals == 'cano_s': fig_name = 'ZZcano_2nd'
    if goals == 'cano_f': fig_name = 'ZZcano_1st'
    plt.savefig(f'data/{fig_name}.pdf')
    plt.clf()

def draw_iters(csv_name,best_label,fig_num):
    JT_m_per_run = []
    JT_b_per_run = []
    X_i = []
    N_i = []
    with open(csv_name, mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for line in csvFile:
            JT_b_per_run.append(float(line['JT_ss_b']))
            JT_m_per_run.append(float(line['JT_ss_m']))
            X_i.append(float(line['var_X']))
            N_i.append(float(line['var_N']))
    fig,ax = plt.subplots(2,1,figsize=(12,9))
    ax[0].plot(X_i,'r')
    ax2 = ax[0].twinx()
    ax2.plot(N_i,'b')
    ax[1].plot(JT_m_per_run,label='min')
    ax[1].plot(JT_b_per_run,label=rf'$|{best_label}\rangle$')
    ax[0].set_ylabel(r'$J_X$')
    ax[0].yaxis.label.set_color('r')
    ax[0].tick_params(axis='y',colors='r')
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y',colors='b')
    ax2.set_ylabel(r'$J_N$')
    ax[1].legend(loc='best')
    if fig_num == 3:
        ax[0].set_xticks([int(100*i) for i in range(9)])
        ax[1].set_xticks([int(100*i) for i in range(9)])
        ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ylim = ax[0].get_ylim()
        ax2.set_ylim(ylim[0]*10,ylim[1]*10)
    elif fig_num == 4:
        ax2.set_yticks([0.0,0.2,0.4,0.6,0.8])
        ylim = ax[0].get_ylim()
        ax2.set_ylim(ylim[0],ylim[1])
        ax[0].set_xticks([int(100*i) for i in range(7)])
        ax[1].set_xticks([int(100*i) for i in range(7)])
    ax[0].grid()
    ax[1].grid()
    ax[1].set_ylabel('infidelity')
    fig.supxlabel('iters')
    plt.savefig(f'data/Fig_{fig_num}.pdf')
    plt.clf()

plot_stored_results('cano_s')
plot_stored_results('cano_f')
draw_iters('data/iter_info_fig3.csv','0-',3)
draw_iters('data/iter_info_fig4.csv','1+',4)
cano_fit(np.linspace(20,50,10**5))