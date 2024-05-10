import json
import pickle
from environment.GCB_wrapper import GCB_Wrapper
from config.env_config import DebugLogsConfig, GCBMujocoConfig
from environment.mujoco_env import MujocoEnvironment
from goal_conditioned_baselines import logger
from goal_conditioned_baselines.chac import config
from goal_conditioned_baselines.chac.chac_policy import set_env_to_load
from goal_conditioned_baselines.chac.rollout import RolloutWorker

from robot.ant_robot import AntSmallF

num_episodes = 20

def get_env(robot, env_config):
    return GCB_Wrapper(MujocoEnvironment(robot, env_config, logger), env_config)

def run_assess_hac_ant_mujoco(load_path, save_path, time_horizon = 27, max_ep_length=700, step_size=15):
    robot = AntSmallF
    include_env_in_state = False

    env_config = GCBMujocoConfig({
        "dims":{
            'o': robot.get_state_length() + (include_env_in_state * 21*21),
            'u': len(robot.joints),
            'g': 2,
        },
        "debug_logs": DebugLogsConfig({"robot": False, "env": False, "sim": False}),
        "time_scales": f"{time_horizon},{time_horizon}",
        "bounded_terrain": True,
        "fall_on_turn_over": False,
        "render": False,
        "step_size": step_size,
        "increasing_difficulty": False,
        "include_env_in_state": include_env_in_state,
        "include_larger_features": False,
        "max_episode_length": max_ep_length
    })


    params = config.DEFAULT_PARAMS
    params['env_name'] = 'Ant-mujoco'
    params['num_threads'] = 1
    #  params = config.prepare_params(params)
    params['T'] =  max_ep_length
    params['gamma'] = 1.0 - 1.0/params['T']
    params['chac_params'] = dict()
    # dims = config.configure_dims(params)

    env = get_env(robot, env_config)
    def make_env():
        return env

    params['make_env'] = make_env
    set_env_to_load(env)

    eval_params = config.EVAL_PARAMS

    for name in config.ROLLOUT_PARAMS_LIST:
        eval_params[name] = params[name]

    f = open(load_path, 'rb')
    f.close()

    with open(load_path, 'rb') as policy_file:
        policy = pickle.load(policy_file)

    evaluator = RolloutWorker(params['make_env'], policy,  env_config.dims, logger, render = False, **eval_params)

    difficulty = 0.0
    results = {}
    while difficulty < 1.0:
        env.wrapped_env.intensity = difficulty
        successes = []
        for _ in range(num_episodes):
            evaluator.generate_rollouts()
            successes.append(bool(evaluator.success_history[-1]))
        success_rate = sum(successes)/len(successes)
        results[difficulty] = {"successes":successes, "success_rate": success_rate}
        print(f"Difficulty: {difficulty}, Success Rate: {success_rate}")
        difficulty += 0.05
 
    with open(save_path, "w") as results_file:
        results_file.write(json.dumps(results))

