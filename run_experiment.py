import sys

from rl_runners.parallel_run_curriculum_learning_experiment_1 import run_curriculum_learn_experiment
from rl_runners.parallel_run_curriculum_learning_experiment_2 import run_curriculum_learn_experiment_part_2

if __name__ == "__main__":
    if sys.argv[2] == "curriculum_experiment":
        run_curriculum_learn_experiment(sys.argv[1])
    elif sys.argv[2] == "curriculum_experiment_2":
        run_curriculum_learn_experiment_part_2(sys.argv[1])