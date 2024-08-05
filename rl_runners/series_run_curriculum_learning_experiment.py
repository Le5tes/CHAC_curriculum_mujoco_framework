from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_curriculum_learn_experiment(savepath, continuing = False):
    # without curriculum learning:
    difficulties = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

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
            with open(run_save_path+ "/epochs_completed", "r") as f:
                epochs_completed = int(f.read())
            
            run_hac(run_save_path, 50, difficulty, False, num_cpu=10, nn_size=256, loadpath = run_save_path + "/policy_latest.pkl", epoch_num = epochs_completed + 1)

            continuing = False
        else:
            run_hac(run_save_path, 50, difficulty, False, num_cpu=10, nn_size = 256)


    with open(savepath + "/current_spec", "w") as f:
        f.write(f"{len(difficulties)}")

    run_save_path = f"{savepath}/curriculum"
    
    if continuing:
        with open(run_save_path+ "/epochs_completed", "r") as f:
            epochs_completed = int(f.read())
        
        run_hac(run_save_path, 50, 0.0, True, num_cpu=10, nn_size=256, loadpath = run_save_path + "/policy_latest.pkl", epoch_num = epochs_completed + 1)
    
    else:
        run_hac(run_save_path, 50, 0.0, True, num_cpu=10, nn_size = 256)