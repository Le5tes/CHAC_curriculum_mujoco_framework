import json
import multiprocessing as mp
import pickle
import sys

from tqdm import tqdm
from environment.GCB_wrapper import GCB_Wrapper
from config.env_config import DebugLogsConfig, GCBMujocoConfig

from goal_conditioned_baselines import logger
from goal_conditioned_baselines.chac import config
from goal_conditioned_baselines.chac.chac_policy import set_env_to_load
from goal_conditioned_baselines.chac.rollout import RolloutWorker
import os
from multiprocessing_infra.policy_runner import PolicyProcess
from multiprocessing_infra.queued_buffer import QueuedBuffer
from robot.ant_robot import AntSmallF
from pathlib import Path
import random
import string

from robot.anymal_robot import Anymal


alphabet = string.ascii_lowercase + string.digits
def generate_short_id():
    return ''.join(random.choices(alphabet, k=4))
    

def save_value(path,writetype, value):
    saved = False
    trys = 0
    while not saved and trys < 3:
        try:
            with open(path, writetype) as f:
                f.write(value)
            saved = True
        except IOError as exc:
            trys += 1
            if trys < 3:
                print(f"failed to save value to {path} - trying again!", exc)
            else:
                print(f"still failed to save value to {path}, giving up this time!")

def train(rollout_worker, evaluator,n_epochs, n_test_rollouts, n_episodes, n_train_batches, policy_save_interval, save_policies, savepath, starting_epoch, sub_processes, queued_buffer):
    latest_policy_path = os.path.join(savepath, 'policy_latest.pkl')
    best_policy_path = os.path.join(savepath, 'policy_best_{}.pkl')
    periodic_policy_path = os.path.join(savepath, 'policy_{}.pkl')

    best_achievement = -1

    success_rates = []
    ending_intensities = []
    epoch = starting_epoch

    if starting_epoch == 0:
        policy_path = periodic_policy_path.format(0)
        evaluator.save_policy(policy_path)

    while epoch < n_epochs:
        # train
        logger.info("Training epoch {}".format(epoch))
        rollout_worker.clear_history()
        
        policy, time_durations = rollout_worker.generate_parallel_rollouts_update(n_episodes, n_train_batches, sub_processes, queued_buffer) if sub_processes is not None else rollout_worker.generate_rollouts_update(n_episodes, n_train_batches, epoch)

        logger.info('Time for epoch {}: {:.2f}. Rollout time: {:.2f}, Training time: {:.2f}'.format(epoch, time_durations[0], time_durations[1], time_durations[2]))

        # eval
        logger.info("Evaluating epoch {}".format(epoch))
        evaluator.clear_history()

        for _ in tqdm(range(n_test_rollouts), file=sys.__stdout__, desc='Eval Rollout'):
            evaluator.generate_rollouts()

        logger.info("prepping logs")
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

        save_value(savepath + "/results.csv", "a", f"{success_rate},{end_intensity},{achievement}\n")

        try:
            rollout_worker.policy.draw_hists(img_dir=logger.get_dir())
        except Exception as e:
            pass

        logger.info("Data_dir: {}".format(logger.get_dir()))
        logger.dump_tabular()

        # save latest policy
        evaluator.save_policy(latest_policy_path)

        save_value(savepath + "/epochs_completed", "w", f"{epoch}")

        if policy_save_interval > 0 and (epoch + 1) % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch + 1)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)
        
        if achievement >= best_achievement and save_policies:
            best_achievement = achievement
            policy_path = best_policy_path.format(epoch)
            logger.info(
                'New best acheivement: {}. Saving policy to {} ...'.format(best_achievement, policy_path))
            evaluator.save_policy(policy_path)

        epoch += 1

    logger.info('All epochs are finished. Stopping the training now.')


def run_hac(savepath, num_epochs = 1000, starting_difficulty = 0.0, increasing_difficulty = False, time_horizon = (27,27), max_ep_length=700, step_size=15, num_cpu= 1, nn_size = 64, loadpath = None, epoch_num = 0, include_env_in_state = False, use_curiosity = True, robot_choice = "ant"):
    # Make sure the savepath directory exists and make it if not! 
    # savepath = savepath + "/" + generate_short_id()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("context previously set")
    n_train_batches =  32
    policy_save_interval = 5

    Path(savepath).mkdir(parents=True, exist_ok=True)
    if robot_choice == "ant":
        robot = AntSmallF
    elif robot_choice == "anymal":
        robot = Anymal

    env_config = GCBMujocoConfig({
        "dims":{
            'o': robot.get_state_length() + (include_env_in_state * 21*21),
            'u': len(robot.joints),
            'g': 2,
        },
        "debug_logs": DebugLogsConfig({"robot": False, "env": False, "sim": False}),
        "time_scales": ','.join(str(i) for i in time_horizon),
        "bounded_terrain": True,
        "render": False,
        "step_size": step_size,
        "start_intensity":float(starting_difficulty),
        "increasing_difficulty": increasing_difficulty,
        "max_episode_length": max_ep_length,
        "include_env_in_state": include_env_in_state,
        "include_larger_features": include_env_in_state,
        "larger_feature_difficulty_scaling": 2 if robot_choice == "anymal" else 1
    })

    params = config.DEFAULT_PARAMS
    params['num_threads'] = 1
    params['time_scales'] = env_config.time_scales
    params['T'] =  env_config.max_episode_length
    params['gamma'] = 1.0 - 1.0/params['T']
    params['chac_params'] = dict()
    params['env_name']="AntMujoco"
    params['n_levels']= len(time_horizon)
    params['fw'] = int(use_curiosity)
    params['fw_hidden_size'] = f'{nn_size},{nn_size},{nn_size}'
    params['q_hidden_size'] = nn_size
    params['mu_hidden_size'] = nn_size
    params['batch_size'] = 4096
    params['n_pre_episodes']=30
    params['q_lr']=0.0001
    params['mu_lr']=0.0001
    params['fw_lr']=0.0001

    processes = None
    queued_buffer = None
    logger.debug("### starting processes")
    if num_cpu > 1:
        queued_buffer = QueuedBuffer(1_000_000,num_cpu - 1)
        processes = [PolicyProcess(args=(env_config.dims, params,robot, env_config, n_train_batches, queued_buffer)) for _ in range(num_cpu - 1)]
        for p in processes:
            logger.debug("calling start")
            p.start()
            logger.debug("called start")
        for p in processes:
            assert p.wait_for_message() == "ready"
    logger.debug("### processes started")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from environment.mujoco_env import MujocoEnvironment # have to import this AFTER setting up multiprocessing (presumably because mujoco uses threading internally)
    def get_env():
        return GCB_Wrapper(MujocoEnvironment(robot, env_config, logger), env_config)

    params['make_env'] = get_env

    if loadpath is not None:
        set_env_to_load(get_env())

        f = open(loadpath, 'rb')
        f.close()

        with open(loadpath, 'rb') as policy_file:
            policy = pickle.load(policy_file)
        
    else:
        policy = config.configure_policy(env_config.dims, params, get_env())

        ## Set up file for storing results!
        with open(savepath + "/results.csv", "w") as file:
            file.write("success_rate,end_intensity,achievement\n")

    if queued_buffer is not None:
        buffers = [layer.replay_buffer for layer in policy.layers]
        queued_buffer.set_buffers(buffers)
    
    rollout_params = config.ROLLOUT_PARAMS
    eval_params = config.EVAL_PARAMS

    for name in config.ROLLOUT_PARAMS_LIST:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(get_env, policy, env_config.dims, logger, **rollout_params)

    eval_params['exploit'] = True
    eval_params['training_rollout_worker'] = rollout_worker
    evaluator = RolloutWorker(get_env, policy, env_config.dims, logger, **eval_params)
    logger.debug("### start train")
    train(rollout_worker, evaluator, num_epochs, 100,100,n_train_batches,policy_save_interval, True, savepath, epoch_num, processes, queued_buffer)
    logger.info(f"Final intensity reached: {policy.env.wrapped_env.intensity}")

    ## cleanup
    if processes is not None:
        for process in processes:
            process.stop()
    ## these are deleted so if we run it again it doesn't try to include them in the context for the new processes we spin up.
    del MujocoEnvironment
    del get_env
    params['make_env'] = None
