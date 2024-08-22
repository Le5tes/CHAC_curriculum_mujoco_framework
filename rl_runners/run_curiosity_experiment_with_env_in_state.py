from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_curiosity_experiment_with_eis(savepath, continuing = False):
    level_specs = [False, True]

    if not continuing:
        current_spec = 0
    else:
        with open(savepath + "/current_spec", "r") as f:
            current_spec = int(f.read())

    for spec in level_specs[current_spec:]:
        with open(savepath + "/current_spec", "w") as f:
            f.write(f"{current_spec}")
        
        run_save_path = f"{savepath}/curiosity-{spec}"

        if continuing:
            try:
                with open(run_save_path+ "/epochs_completed", "r") as f:
                    epochs_completed = int(f.read())
            except FileNotFoundError:
                continuing = False

        if continuing:
            run_hac(run_save_path, 50, 0.0, increasing_difficulty = True,  num_cpu=11, nn_size=256, loadpath = run_save_path + "/policy_latest.pkl", epoch_num = epochs_completed + 1, include_env_in_state = True, use_curiosity=spec)

            continuing = False
        else:
            run_hac(run_save_path, 50, 0.0, increasing_difficulty = True, num_cpu=11, nn_size=256, include_env_in_state = True, use_curiosity=spec)

        current_spec += 1
