import sys

from rl_runners.series_run_curriculum_learning_experiment import run_curriculum_learn_experiment

if __name__ == "__main__":
    if sys.argv[2] == "curriculum_experiment":
        run_curriculum_learn_experiment(sys.argv[1])
