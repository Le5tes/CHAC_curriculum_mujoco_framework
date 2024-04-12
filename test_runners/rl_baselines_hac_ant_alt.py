import pickle
from environment.GCB_wrapper import GCB_Wrapper
from config.env_config import DebugLogsConfig, GCBMujocoConfig
from environment.mujoco_env import MujocoEnvironment
from goal_conditioned_baselines import logger
from goal_conditioned_baselines.chac import config
from goal_conditioned_baselines.chac.chac_policy import set_env_to_load
from goal_conditioned_baselines.chac.rollout import RolloutWorker

from robot.ant_robot import AntSmallF

num_episodes = 50

def get_env(robot, env_config):
    return GCB_Wrapper(MujocoEnvironment(robot, env_config, logger), env_config)

def run_test_hac_ant_mujoco(load_path, time_horizon = 27, max_ep_length=700, step_size=15):
    robot = AntSmallF

    env_config = GCBMujocoConfig({
        "dims":{
            'o': robot.get_state_length(),
            'u': len(robot.joints),
            'g': 2,
        },
        "debug_logs": DebugLogsConfig({"robot": False, "env": False, "sim": False}),
        "time_scales": f"{time_horizon},{time_horizon}",
        "bounded_terrain": True,
        "fall_on_turn_over": False,
        "render": True,
        "step_size": step_size,
        "increasing_difficulty": True,
        "start_intensity": 0.0,
        "include_larger_features": True,
        "num_successes_to_increment": 1,
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
    for _ in range(num_episodes):
        evaluator.generate_rollouts()
        env.wrapped_env.update_successes(bool(evaluator.success_history[-1]))
    
    print("final intensity", env.wrapped_env.intensity)