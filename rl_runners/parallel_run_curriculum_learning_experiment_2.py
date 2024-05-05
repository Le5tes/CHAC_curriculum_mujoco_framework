from multiprocessing import Process

from rl_runners.rl_baselines_hac_mujoco import run_hac

def run_curriculum_learn_experiment_part_2(savepath):
    processes = []
    # withoit curriculum learning:
    difficulties = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for difficulty in difficulties:
        args = (f"{savepath}/non-curriculum-{difficulty}", 500, difficulty, False)
        process = Process(target=run_hac, args = args)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
