from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_heirarchical_learning_experiment_with_env_in_state(savepath):
    level_nums = [1,2,3,4,5]

    for num_levels in level_nums:
        run_hac(f"{savepath}/non-hierachies-{num_levels}", 50, 0.0, increasing_difficulty = True, num_cpu=10, nn_size=256)