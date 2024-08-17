from rl_runners.rl_baselines_hac_mujoco import run_hac
import sys

if __name__ == "__main__":
   run_hac(sys.argv[1],200, num_cpu=11, increasing_difficulty = True, nn_size = 256, include_env_in_state = True)