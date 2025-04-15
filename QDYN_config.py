from pathlib import Path
import time, qutip, qdyn, qdyn.model, qdyn.pulse, os, subprocess, numpy as np,platform,localTools,J_T_local,csv,read_write
import datetime,read_config

def addHam2model(num_qubit,Ham_num,tgrid,control_source,endTime,header,t_rise,lambda_a,n_levels = False,dissipation=False,continue_from=False):
    ham_info = localTools.hamiltonian_info(num_qubit, Ham_num)
    num_correct_pulses = sum(ham_info) - ham_info[0]
    num_pulses = 0
    while num_pulses != num_correct_pulses:
        H = localTools.hamiltonian(num_qubit, Ham_num=Ham_num,n_levels = n_levels,dissipation = dissipation)
        if continue_from:
            control_args = localTools.control_generator_S2L(
                num_qubit, Ham_num, control_source, endTime, header)
        else:
            control_args = localTools.control_generator(
                num_qubit, Ham_num, control_source, endTime, header)
        model = qdyn.model.LevelModel()
        if ham_info[0]:
            model.add_ham(H[0], op_unit="iu")
        for i in range(ham_info[0], ham_info[0] + ham_info[1]):
            model.add_ham(
                H[i][0],
                pulse=qdyn.pulse.Pulse(
                    tgrid,
                    amplitude=H[i][1](tgrid, control_args[i - ham_info[0]]),
                    time_unit="iu",
                    ampl_unit="iu",
                    # is_complex="True",
                    config_attribs={
                        "filename": f"dummy_pulse_initial_{i-ham_info[0]}.dat",
                        "oct_shape": "zero",
                        "t_rise": t_rise,
                        "t_fall": t_rise,
                        "oct_increase_factor": 50,
                        "oct_pulse_min": -2,
                        "oct_pulse_max": 2,  # 1.0e10,
                        "oct_lambda_a": lambda_a,
                        "oct_outfile": f"dummy_pulse_oct_{i-ham_info[0]}.dat",}),
                    op_unit="iu")
        for i in range(ham_info[0] + ham_info[1], sum(ham_info)):
            model.add_ham(
                H[i][0],
                pulse=qdyn.pulse.Pulse(
                    tgrid,
                    amplitude=H[i][1](tgrid, control_args[i - ham_info[0] - ham_info[1]]),
                    time_unit="iu",
                    ampl_unit="iu",
                    config_attribs={
                        "filename": f"pulse_initial_{i-ham_info[0]-ham_info[1]}.dat",
                        "oct_shape": "flattop",
                        "t_rise": t_rise,
                        "t_fall": t_rise,
                        "oct_increase_factor": 20,
                        "oct_pulse_min": -500,
                        "oct_pulse_max": 500,
                        "oct_lambda_a": lambda_a,
                        "oct_outfile": f"pulse_oct_{i-ham_info[0]-ham_info[1]}.dat"}),
                    op_unit="iu")
        num_pulses = len(model._pulse_ids)
    return model

def default_run_method():
    return {"immediate_return":True,"slurm":True,"store":True}

def runner(runfolder, jobname,runtime,mem,program_ID,n_cpu=1,run_method = None):
    if run_method:
        if 'immediate_return' in run_method:
            immediate_return = run_method["immediate_return"]
        else:immediate_return = True
        if 'slurm' in run_method:
            slurm = run_method["slurm"]
        else:slurm = True
        if 't_sleep' in run_method:
            t_sleep = run_method["t_sleep"]
        else:t_sleep = 10
    else:
        immediate_return = True
        slurm = True
        t_sleep = 10
    env = {**os.environ, "OMP_NUM_THREADS": str(n_cpu)}
    executable = ["inFidelity", "inFidelity_gate1", "inFidelity_gate2","JT_X","JT_XN","prop"]
    localTools.regulate_pulses_in_folder(runfolder)
    os_system = platform.platform()
    if 'Linux' in os_system:
        if slurm:
            slrm_scrpt = localTools.text_content_local_run(jobname,n_cpu,runtime,mem,runner=f'./runner/{executable[program_ID]}',runfolder=runfolder)
            with open(f'{runfolder}/run_rf.sh','w') as f:
                f.write(slrm_scrpt)
            fortran_result = subprocess.run(["sbatch", f"{runfolder}/run_rf.sh"])
        else:
            fortran_result = subprocess.run(['./runner/'+executable[program_ID], runfolder], capture_output=True, text=True, env=env)       
        if immediate_return:
            time.sleep(0.1)
            return 0
        print(f'{runfolder} program starts at {datetime.datetime.now()}')
        while not any([os.path.exists(runfolder + 'psi_final_after_oct.dat'),os.path.exists(runfolder + 'psi0_final_after_oct.dat')]):
            time.sleep(t_sleep)
        if executable[program_ID] != "prop":
            iters, J_T = np.split(np.loadtxt(runfolder + "oct_iters.dat", usecols=(0, 1)), 2, axis=1)
            runner_result = J_T[-1][0]
        else:runner_result = 1
    else: runner_result = 'OS system is not Linux, pass'
    return runner_result

def mem_routine(num_qubit,n_states,t_step):
    if num_qubit == 4:return 70
    mem_1q = 4e-3 # here looks suspicious
    mem_2q = 1e-2
    if num_qubit == 1 :return int(n_states * mem_1q * t_step * 1.1) + 2 * n_states
    if num_qubit == 2 :return int(n_states * mem_2q * t_step * 1.1) + 2 * n_states

def runtime_routine():
    return '2-12:00:00'

def rotate_state_call(target_state,num_qubit,T,direction = 1):
    return qutip.Qobj(
        localTools.rotate_state(target_state, num_qubit, direction, T))

def rotate_matrix_call(lab_mat,num_qubit,T,direction = 1):
    return qutip.Qobj(
        localTools.rotate_matrix(lab_mat, num_qubit, direction, T))

def qdyn_prop(
    Ham_num,  # Hamiltonian
    qdyn_tlist,  # tuple of the form (T, Nt)
    initial_state,
    runfolder,
    control_source,
    header=None,
    num_qubit=3,
    dissipation = False,
    **user_kwargs,
):
    runfolder=Path(runfolder)
    runfolder.mkdir(parents=True, exist_ok=True)
    dt = (qdyn_tlist[0]) / (qdyn_tlist[1] - 1)
    tgrid = np.linspace(
        float(dt / 2),
        float(qdyn_tlist[0] - dt / 2),
        qdyn_tlist[1] - 1,
        dtype=np.float64,
    )  #! here the default is np.float 64, it has been changed manually, also in QDYN python package
    if "control" in user_kwargs:
        user_kwargs.pop("control")
    if isinstance(initial_state, str):
        initial_state = localTools.canoGHZGen(num_qubit, initial_state)
    model = addHam2model(num_qubit,Ham_num,tgrid,control_source,qdyn_tlist[0],header,1,1,2,dissipation = dissipation)
    print('Ham added to model')
    model.set_propagation(
        qdyn_tlist[0], qdyn_tlist[1], time_unit="iu", prop_method="cheby")
    model.add_state(initial_state, "initial")
    user_variables = {}
    if not os.path.exists(runfolder):
        os.mkdir(runfolder)
    user_variables["runfolder"] = str(Path(runfolder))
    user_variables.update(user_kwargs)
    model.user_data = user_variables
    model.write_to_runfolder(str(runfolder))  # write everything to runfolder
    mem = mem_routine(num_qubit,1,qdyn_tlist[1])
    runner(str(runfolder)+'/','prop','00:10:00',mem,program_ID=5,n_cpu=1,run_method = {"immediate_return":False,"slurm":False})
    return model



def opt_para(state_path,num_qubit,T,JT,src_path = None):
    if JT == 1:return 2,0,200
    return 2,1.5,50


def qdyn_model(
    Ham_num,  # Hamiltonian
    qdyn_tlist,  # tuple of the form (T, Nt)
    target_state,
    runfolder,
    control_source,
    initial_state,
    header=None,
    num_qubit=3,
    lambda_a=10,
    iter_stop=100,
    JT=0,
    t_rise=1,
    dissipation = False,
    run_method = False,
    **user_kwargs,
):
    if not run_method:run_method=default_run_method()
    if 'N_factor' in run_method.keys():N_factor = run_method['N_factor']
    else:N_factor = False
    n_levels = int(len(target_state) ** (1/num_qubit))
    if type(initial_state) != qutip.Qobj:
        initial_state = qutip.Qobj(initial_state)
        target_state = qutip.Qobj(target_state)
    if 'phi' in run_method.keys():
        phi = run_method['phi']
    else:
        phi = 0
    if isinstance(target_state, str):
        target_state = localTools.canoGHZGen(num_qubit, target_state,phi)
    target_state = rotate_state_call(target_state,Ham_num,num_qubit,qdyn_tlist[0])
    dt = (qdyn_tlist[0]) / (qdyn_tlist[1] - 1)
    tgrid = np.linspace(
        float(dt / 2),
        float(qdyn_tlist[0] - dt / 2),
        qdyn_tlist[1] - 1,
        dtype=np.float64,
    )  #! here the default is np.float 64, it has been changed manually, also in QDYN python package
    # Initialize model
    if JT:
        if N_factor:X_factor,_,lambda_a = opt_para(control_source,num_qubit,qdyn_tlist[0],JT,control_source)
        else:X_factor,N_factor,lambda_a = opt_para(control_source,num_qubit,qdyn_tlist[0],JT,control_source)
    model = addHam2model(num_qubit,Ham_num,tgrid,control_source,qdyn_tlist[0],header,t_rise,lambda_a,n_levels,dissipation = dissipation)
    obj_path = Path(runfolder)
    obj_path.mkdir(parents=True, exist_ok=True)
    model.set_propagation(
        qdyn_tlist[0], qdyn_tlist[1], time_unit="iu", prop_method="cheby")
    if JT:
        J_T_conv = 9.8e-5
        delta_J_T_conv = 1e-10
    else:
        J_T_conv = 9.8e-3
        delta_J_T_conv = 1e-5
    mem = mem_routine(num_qubit,1,qdyn_tlist[1])
    model.set_oct(
        method="krotovpk",
        max_ram_mb=mem,
        J_T_conv=J_T_conv,
        delta_J_T_conv=delta_J_T_conv,
        iter_dat="oct_iters.dat",
        continue_=False,
        params_file="oct_params.dat",
        limit_pulses=True,
        # keep_pulses="prev",
        iter_stop=iter_stop)
    if JT:
        program_ID = 3
        if JT == 2:program_ID = 4
        X = qutip.Qobj(rotate_matrix_call(J_T_local.mat_X(num_qubit,X_factor),Ham_num,num_qubit,qdyn_tlist[0]))
        model.add_observable(
            X,
            None,#"T_X.dat",
            exp_unit="iu",
            time_unit="iu",
            col_label="<X>",
            square="<Xsq>")
        if JT == 2:
            N = J_T_local.mat_N(num_qubit,N_factor,True)
            model.add_observable(
                N,
                None,#"T_N.dat",
                exp_unit="iu",
                time_unit="iu",
                col_label="<N>",
                square="<Nsq>")
    model.add_state(initial_state, "initial")
    model.add_state(target_state, "final")
    user_variables = {}
    user_variables["runfolder"] = str(Path(runfolder))
    user_variables.update(user_kwargs)
    model.user_data = user_variables
    model.write_to_runfolder(str(runfolder))
    jobname = runfolder.split('/')[1]
    runtime = runtime_routine()
    if JT:
        if JT == 2:read_config.kill_obsevable(str(Path(runfolder)))
        fieldnames = ['X_factor','N_factor','lambda_a']
        if isinstance(control_source,str):
            src_state = read_write.stateReader(f'{control_source}/psi_final_after_oct.dat',16)
            src_state = localTools.rotate_state(src_state,4,0,qdyn_tlist[0])
            cat_i,JT_i = J_T_local.bestCat(src_state)
            if "store" in run_method:
                if run_method["store"]:
                    localTools.store_control_source(control_source,runfolder)
            data_dict = [{'X_factor':X_factor,'N_factor':N_factor,'lambda_a':lambda_a,'control_source':control_source,'cat_i':cat_i,'JT_i':JT_i}]
            fieldnames += ['control_source','cat_i','JT_i']
        else:data_dict = [{'X_factor':X_factor,'N_factor':N_factor,'lambda_a':lambda_a}]
        with open(str(Path(runfolder))+'/source_info.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_dict)
    else: program_ID = 0
    runner_result = runner(runfolder,jobname,runtime,mem = mem + 10,program_ID=program_ID,run_method = run_method)
    return runner_result

