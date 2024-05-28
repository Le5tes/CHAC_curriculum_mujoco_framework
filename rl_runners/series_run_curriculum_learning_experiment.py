from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_curriculum_learn_experiment(savepath):
    # without curriculum learning:
    difficulties = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for difficulty in difficulties:
       run_hac(f"{savepath}/non-curriculum-{difficulty}", 100, difficulty, False, num_cpu=10, bind_core=0)

    run_hac(f"{savepath}/curriculum-{difficulty}", 100, 0.0, True, num_cpu=10, bind_core=0)