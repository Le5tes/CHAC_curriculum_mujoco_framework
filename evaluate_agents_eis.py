import sys

from test_runners.assess_trained_model import run_assess_hac_ant_mujoco

if __name__ == "__main__":
   path = sys.argv[1]
   for arg in sys.argv[2:]:
      run_assess_hac_ant_mujoco(path + arg, path + arg + ".results.json", include_env_in_state=True)
   