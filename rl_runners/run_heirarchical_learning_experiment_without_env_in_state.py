from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_hierarchical_learning_experiment_w_eis(savepath, continuing = False):
    level_specs = [(1, (730,)),(2, (27,27)),(3,(9,9,9)),(4,(5,5,5,6)),(5,(4,4,4,4,3))]

    if not continuing:
        current_spec = 0
    else:
        with open(savepath + "/current_spec", "r") as f:
            current_spec = int(f.read())

    for num_levels, time_horizon in level_specs[current_spec:]:
        with open(savepath + "/current_spec", "w") as f:
            f.write(f"{current_spec}")
        
        run_save_path = f"{savepath}/non-hierachies-{num_levels}"

        if continuing:
            try:
                with open(run_save_path+ "/epochs_completed", "r") as f:
                    epochs_completed = int(f.read())
            except FileNotFoundError:
                continuing = False

        if continuing:
            run_hac(run_save_path, 50, 0.0, increasing_difficulty = True, time_horizon=time_horizon, num_cpu=10, nn_size=256, loadpath = run_save_path + "/policy_latest.pkl", epoch_num = epochs_completed + 1)

            continuing = False
        else:
            run_hac(run_save_path, 50, 0.0, increasing_difficulty = True, time_horizon=time_horizon, num_cpu=10, nn_size=256)

        current_spec += 1
