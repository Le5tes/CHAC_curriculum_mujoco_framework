from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_hierarchical_learning_experiment(savepath):
    level_specs = [(1, (730,)),(2, (27,27)),(3,(9,9,9)),(4,(5,5,5,6)),(5,(4,4,4,4,3))]

    for num_levels, time_horizon in level_specs:
        run_hac(f"{savepath}/non-hierachies-{num_levels}", 50, 0.0, increasing_difficulty = True, time_horizon=time_horizon, num_cpu=10, nn_size=256)