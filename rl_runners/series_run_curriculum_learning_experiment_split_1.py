from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_curriculum_learn_experiment_eis_p1(savepath, continuing = False):
    # without curriculum learning:
    difficulties = [0.0, 0.05, 0.1, 0.15, 0.2]

    if not continuing:
        current_spec = 0
    else:
        with open(savepath + "/current_spec", "r") as f:
            current_spec = int(f.read())

    for difficulty in difficulties[current_spec:]:
        with open(savepath + "/current_spec", "w") as f:
            f.write(f"{current_spec}")
        run_save_path = f"{savepath}/non-curriculum-{difficulty}"

        if continuing:
            try:
                with open(run_save_path+ "/epochs_completed", "r") as f:
                    epochs_completed = int(f.read())
            except FileNotFoundError:
                continuing = False
        
        if continuing:
            run_hac(run_save_path, 50, difficulty, False, num_cpu=11, nn_size=256, loadpath = run_save_path + "/policy_latest.pkl", epoch_num = epochs_completed + 1, include_env_in_state = True)

            continuing = False
        else:
            run_hac(run_save_path, 50, difficulty, False, num_cpu=11, nn_size = 256, include_env_in_state = True)

        current_spec += 1


    with open(savepath + "/current_spec", "w") as f:
        f.write(f"{len(difficulties)}")

    run_save_path = f"{savepath}/curriculum"
    
    if continuing:
        try:
            with open(run_save_path+ "/epochs_completed", "r") as f:
                epochs_completed = int(f.read())
        except FileNotFoundError:
            continuing = False
        
    if continuing:
        run_hac(run_save_path, 50, 0.0, True, num_cpu=11, nn_size=256, loadpath = run_save_path + "/policy_latest.pkl", epoch_num = epochs_completed + 1, include_env_in_state = True)
    
    else:
        run_hac(run_save_path, 50, 0.0, True, num_cpu=11, nn_size = 256, include_env_in_state = True)