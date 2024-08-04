from rl_runners.rl_baselines_hac_mujoco import run_hac
import sys

save_path = sys.argv[1]

with open(save_path+ "/epochs_completed", "r") as f:
   epochs_completed = int(f.read())

print("continuing from epoch", epochs_completed + 1)

if __name__ == "__main__":
   run_hac(save_path,200, num_cpu=10, increasing_difficulty = True, nn_size = 256, loadpath = sys.argv[2], epoch_num = epochs_completed + 1)