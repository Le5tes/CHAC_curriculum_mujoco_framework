import json
from environment.GCB_wrapper import GCB_Wrapper, env_factory
from config.env_config import DebugLogsConfig, GCBMujocoConfig
from environment.mujoco_env import MujocoEnvironment
from goal_conditioned_baselines import logger
from goal_conditioned_baselines.chac.chac_policy import CHACPolicy
from goal_conditioned_baselines.chac import config
from goal_conditioned_baselines.chac.rollout import RolloutWorker
import os
import numpy as np
from robot.ant_robot import AntSmallF
from pathlib import Path


def train(rollout_worker, evaluator,n_epochs, n_test_rollouts, n_episodes, n_train_batches, policy_save_interval, save_policies, savepath):
    latest_policy_path = os.path.join(savepath, 'policy_latest.pkl')
    best_policy_path = os.path.join(savepath, 'policy_best.pkl')
    periodic_policy_path = os.path.join(savepath, 'policy_{}.pkl')

    best_achievement = -np.inf

    success_rates = []
    ending_intensities = []

    for epoch in range(n_epochs):
        # train
        logger.info("Training epoch {}".format(epoch))
        rollout_worker.clear_history()
        policy, time_durations = rollout_worker.generate_rollouts_update(n_episodes, n_train_batches, epoch)
        logger.info('Time for epoch {}: {:.2f}. Rollout time: {:.2f}, Training time: {:.2f}'.format(epoch, time_durations[0], time_durations[1], time_durations[2]))

        # eval
        logger.info("Evaluating epoch {}".format(epoch))
        evaluator.clear_history()

        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, val)
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, val)
        for key, val in policy.logs('policy'):
            logger.record_tabular(key, val)

        success_rate = evaluator.current_success_rate()
        success_rates.append(success_rate)
        end_intensity = evaluator.env.wrapped_env.intensity
        ending_intensities.append(end_intensity)
        achievement = success_rate * end_intensity

        try:
            rollout_worker.policy.draw_hists(img_dir=logger.get_dir())
        except Exception as e:
            pass

        logger.info("Data_dir: {}".format(logger.get_dir()))
        logger.dump_tabular()

        # save latest policy
        evaluator.save_policy(latest_policy_path)

        if policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)
        
        if achievement >= best_achievement and save_policies:
            best_achievement = achievement
            logger.info(
                'New best acheivement: {}. Saving policy to {} ...'.format(best_achievement, best_policy_path))
            evaluator.save_policy(best_policy_path)
        
        if (epoch + 1) == n_epochs:
            logger.info('All epochs are finished. Stopping the training now.')
            with open(savepath + "results.json", "w") as file:
                      file.write(json.dumps({"successes":success_rates, "ending_intensities":ending_intensities}))
            break

def make_env(robot, env_config):
    return GCB_Wrapper(MujocoEnvironment(robot, env_config, logger), env_config)

def run_hac(savepath, num_epochs = 1000, starting_difficulty = 0.0, increasing_difficulty = False, time_horizon = 27, max_ep_length=700, step_size=15):
    # Make sure the savepath directory exists and make it if not! 
    Path(savepath).mkdir(parents=True, exist_ok=True)

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
        "start_intensity":float(starting_difficulty),
        "increasing_difficulty": increasing_difficulty,
        "max_episode_length": max_ep_length,
        "include_env_in_state": include_env_in_state,
        "include_larger_features": include_env_in_state
    })

    params = config.DEFAULT_PARAMS
    params['num_threads'] = 1
    params['T'] =  env_config.max_episode_length
    params['gamma'] = 1.0 - 1.0/params['T']
    params['chac_params'] = dict()
    params['env_name']="AntMujoco"
    # params['fw_hidden_size'] = '256,256,256'
    # params['q_hidden_size'] = 256
    # params['mu_hidden_size'] = 256

    env = make_env(robot, env_config)
    def get_env():
        return env # keep the same env because it keeps track of intensity

    params['make_env'] = get_env

    policy = config.configure_policy(env_config.dims, params, get_env())
    rollout_params = config.ROLLOUT_PARAMS
    eval_params = config.EVAL_PARAMS

    for name in config.ROLLOUT_PARAMS_LIST:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(get_env, policy, env_config.dims, logger, **rollout_params)

    eval_params['exploit'] = True
    eval_params['training_rollout_worker'] = rollout_worker
    evaluator = RolloutWorker(get_env, policy, env_config.dims, logger, **eval_params)

    train(rollout_worker, evaluator, num_epochs, 100,100,100,10, True, savepath)
    logger.info(f"Final intensity reached: {env.wrapped_env.intensity}")
