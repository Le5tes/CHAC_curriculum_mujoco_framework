import sys

from rl_runners.run_curiosity_experiment_with_env_in_state import run_curiosity_experiment_with_eis
from rl_runners.run_curiosity_experiment_without_env_in_state import run_curiosity_experiment_without_eis
from rl_runners.run_heirarchical_learning_experiment_with_env_in_state import run_hierarchical_learning_experiment
from rl_runners.run_heirarchical_learning_experiment_without_env_in_state import run_hierarchical_learning_experiment_w_eis
from rl_runners.series_run_curriculum_learning_experiment_without_env_in_state import run_curriculum_learn_experiment
from rl_runners.series_run_curriculum_learning_experiment_split_1 import run_curriculum_learn_experiment_eis_p1
from rl_runners.series_run_curriculum_learning_experiment_split_2 import run_curriculum_learn_experiment_eis_p2

if __name__ == "__main__":
    if sys.argv[2] == "curriculum_experiment":
        run_curriculum_learn_experiment(sys.argv[1])
    if sys.argv[2] == "curriculum_experiment_eis_p1":
        run_curriculum_learn_experiment_eis_p1(sys.argv[1])
    if sys.argv[2] == "curriculum_experiment_eis_p2":
        run_curriculum_learn_experiment_eis_p2(sys.argv[1])
    if sys.argv[2] == "hierarchy_experiment":
        run_hierarchical_learning_experiment(sys.argv[1])
    if sys.argv[2] == "hierarchy_experiment_w_eis":
        run_hierarchical_learning_experiment_w_eis(sys.argv[1])
    if sys.argv[2] == "curiosity_experiment_with_eis":
        run_curiosity_experiment_with_eis(sys.argv[1])
    if sys.argv[2] == "curiosity_experiment_without_eis":
        run_curiosity_experiment_without_eis(sys.argv[1])
    
