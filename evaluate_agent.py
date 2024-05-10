import sys

from test_runners.assess_trained_model import run_assess_hac_ant_mujoco

if __name__ == "__main__":
   run_assess_hac_ant_mujoco(sys.argv[1], sys.argv[2])
   