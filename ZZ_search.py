import QDYN_config, numpy as np,os,multiprocessing
from localTools import canoGHZGen,cat_phi
from functools import partial

def indi_job(runfolder,Ham_num,T,t_step,target_state,initial_state,num_qubit,t_rise,JT=1,dissipation = False,control_source = False,run_method = None):
    if not control_source:
        control_source = 2
        continue_from = False
        header = None
    else:
        if isinstance(control_source,list):
            src_id = int(runfolder.split('/')[-2])
            control_source = control_source[src_id]
        continue_from = True
        header = 'pulse_oct'
    if JT:lambda_a = 100
    else:lambda_a = 2e-1
    opt_result = QDYN_config.qdyn_model(Ham_num,(T,t_step),target_state=target_state,runfolder=runfolder
        ,control_source=control_source,initial_state=initial_state,header=header,num_qubit=num_qubit,lambda_a=lambda_a,iter_stop=500,JT=JT,t_rise=t_rise,dissipation=dissipation,continue_from=continue_from,run_method =run_method )
    return opt_result



def task(Tgrid,Ham_num,canoLabel,JT=1,dissipation = False, control_source = False):
    num_qubit = 4
    if isinstance(Tgrid,int) or isinstance(Tgrid,float):Tgrid=[Tgrid]
    initial_state = canoGHZGen(num_qubit, '0').full()
    target_state = canoGHZGen(num_qubit,canoLabel).full()
    if control_source:
        n_jobs = 1
        if isinstance(control_source,list):n_jobs = len(control_source)
    else:
        n_jobs = 1
    t_rise=1
    Tlist = []
    task_label = 0
    while os.path.exists(f'rf/{canoLabel}_ZZ{task_label}/'): task_label += 1
    task_label = f'{canoLabel}_ZZ{task_label}'
    for T in Tgrid:
        phi = cat_phi(T,canoLabel)
        run_method = {'phi':cat_phi(T,canoLabel),'slurm':True}
        target_state = canoGHZGen(num_qubit,canoLabel,np.exp(1j*np.pi*phi)).full()
        print(f'{T} jobs.')
        Tlist.append(T)
        t_rise = 1
        t_step = 801
        partial_run = partial(indi_job,Ham_num=Ham_num,T=T,t_step=t_step,target_state=target_state,initial_state=initial_state,num_qubit=num_qubit,t_rise=t_rise,JT=JT,dissipation = dissipation,control_source=control_source,run_method =run_method )
        runfolders = [f'rf/{task_label}/{T}/{i}/' for i in range(n_jobs)]
        with multiprocessing.Pool() as pool:
            results = pool.map(partial_run,runfolders)
        for i in range(len(results)):
            if isinstance(results[i],str):results[i] = i+2


if __name__ == "__main__":
    error_flag = True
    Ham_num = 3
    num_qubit = 4
    canoLabel = '0+'
    Tgrid = np.linspace(20.0,50.0,31)
    task(Tgrid,Ham_num,'0+',JT=0,control_source=None)


