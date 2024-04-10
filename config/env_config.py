from config.Config import Config
import numpy as np

class DebugLogsConfig(Config):
    default_config = {
        "env": False,
        "sim": False,
        "robot": False
    }

class GCBMujocoConfig(Config):
    default_config = {
        "time_scales": "32,32,32",
        "dims": {
            'o': 27,
            'u': 8,
            'g': 27,
        },
        "start_intensity": 0.01,
        "rate": 500,
        "record_path": "",
        "debug_logs": DebugLogsConfig(),
        "arena_size": 10,
        "reward_fn": None,
        "state_goal_indices": np.array([0, 1]),
        "end_goal_thresholds": np.array([0.8,0.8]),
        "sub_goal_thresholds": np.array([0.8, 0.8, 0.5, 0.5]), # len_threshold, len_threshold, height_threshold, orientation_thresholds, velo_theshold, velo_theshold, z_velo_threshold
        # "sub_goal_thresholds": np.array([1.0, 1.0]), # len_threshold, len_threshold, height_threshold, orientation_thresholds, velo_theshold, velo_theshold, z_velo_threshold
        "state_subgoal_indices": np.array([0, 1, 7, 8]),
        # "state_subgoal_indices": np.array([0, 1]),
        "terrain_encoder_path": None,
        "terrain_embeddings_size": 0,
        "bounded_terrain": False,
        "render": False,
        "increasing_difficulty": False,
        "num_successes_to_increment": 10,
        "intensity_increment": 0.01,
        "step_size": 1,
        "max_episode_length": 1000
    }
