
from collections import OrderedDict
import numpy as np
import time
import sys
from goal_conditioned_baselines.utils import store_args
from goal_conditioned_baselines.rollout import Rollout
from goal_conditioned_baselines import logger
from tqdm import tqdm
from copy import deepcopy

from robot.ant_robot import AntSmallF

class RolloutWorker(Rollout):
    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
            exploit=False, history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size,
                history_len=history_len, render=render, **kwargs)

        self.env = self.policy.env
        self.env.visualize = render
        self.env.graph = kwargs['graph']
        self.time_scales = np.array([int(t) for t in kwargs['time_scales'].split(',')])
        self.eval_data = {}

    def train_policy(self, n_train_rollouts, n_train_batches):
        dur_train = 0
        dur_ro = 0

        for episode in tqdm(range(n_train_rollouts), file=sys.__stdout__, desc='Train Rollout'):
            ro_start = time.time()

            success, self.eval_data, train_duration = self.policy.train(episode, self.eval_data, n_train_batches)
            dur_train += train_duration
            self.env.wrapped_env.update_successes(success)
            self.success_history.append(1.0 if success else 0.0)
            self.n_episodes += 1
            dur_ro += time.time() - ro_start - train_duration

        return dur_train, dur_ro
    
    def train_policy_parallel(self, n_train_rollouts, n_train_batches, policy_processes, queued_buffer):

        def detach(item):
            return OrderedDict([(key,val.to('cpu')) for key,val in item.items()])

        def propagate_policies():
            data = [
                # For using 2 gpus, the state_dict here is on gpu one, we need to find a way to copy it in a way we can transfer to gpu 2
                [detach(deepcopy(layer.actor.state_dict())), detach(deepcopy(layer.critic.state_dict())), detach(deepcopy(layer.state_predictor.state_dict())) if layer.fw else None] 
                  for layer in self.policy.layers
            ]

            for p in policy_processes:
                p.update_nets(data)

            for p in policy_processes:
                assert p.wait_for_message() == "nets updated"

        def learn_parallel(episode, queued_buffer, consume=True):
            if episode >= self.policy.pre_episodes:
                self.policy.learn(n_train_batches * len(policy_processes))
            if consume:
                queued_buffer.consume_from_queue()

        dur_train = 0
        dur_ro = 0

        for episode in tqdm(range(n_train_rollouts), file=sys.__stdout__, desc='Train Rollout'):
            ro_start = time.time()

            propagate_policies()

            for p in policy_processes:
                p.train(episode)

            learn_parallel(episode, queued_buffer)

            results = []
            for p in policy_processes:
                results.append(p.wait_for_data())

            successes, evals, train_durations = tuple(zip(*results))

            max_train_duration = max(train_durations)

            dur_train += max_train_duration

            for eval in evals:
                self.add_eval_data(eval)
            for success in successes:
                self.success_history.append(1.0 if success else 0.0)
            self.n_episodes += 1
            dur_ro += time.time() - ro_start - max_train_duration

        learn_parallel(self.n_episodes, queued_buffer, consume=False)

        return dur_train, dur_ro

    def generate_rollouts_update(self, n_train_rollouts, n_train_batches, epoch_num = None):
        dur_start = time.time()
        self.policy.set_train_mode()
        dur_train, dur_ro = self.train_policy(n_train_rollouts, n_train_batches)
        dur_total = time.time() - dur_start
        time_durations = (dur_total, dur_ro, dur_train)
        updated_policy = self.policy
        return updated_policy, time_durations
    
    def generate_parallel_rollouts_update(self, n_train_rollouts, n_train_batches, policy_processes, queued_buffer):
        dur_start = time.time()
        self.policy.set_train_mode()
        dur_train, dur_ro = self.train_policy_parallel(n_train_rollouts, n_train_batches, policy_processes, queued_buffer)
        dur_total = time.time() - dur_start
        time_durations = (dur_total, dur_ro, dur_train)
        updated_policy = self.policy
        return updated_policy, time_durations

    def generate_rollouts(self, return_states=False):
        self.reset_all_rollouts()
        self.policy.set_test_mode()
        success, self.eval_data, _ = self.policy.train(self.n_episodes, self.eval_data, None)
        self.env.wrapped_env.update_successes(success)
        self.success_history.append(1.0 if success else 0.0)
        self.n_episodes += 1
        return self.eval_data

    def logs(self, prefix=''):
        eval_data = self.eval_data

        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('episodes', self.n_episodes)]

        # Get metrics for all layers of the hierarchy
        for i in range(self.policy.n_levels):
            layer_prefix = '{}_{}/'.format(prefix, i)

            subg_succ_prefix = '{}subgoal_succ'.format(layer_prefix)
            if subg_succ_prefix in eval_data.keys():
                if len(eval_data[subg_succ_prefix]) > 0:
                    logs += [(subg_succ_prefix + '_rate',
                              np.mean(eval_data[subg_succ_prefix]))]
                else:
                    logs += [(subg_succ_prefix + '_rate', 0.0)]

            for postfix in ["n_subgoals", "fw_loss", "fw_bonus", "reward", "q_loss", "q_grads",
                    "q_grads_std", "target_q", "next_q", "current_q", "mu_loss", "mu_grads",
                    "mu_grads_std", "reward_-0.0_frac", "reward_-1.0_frac",
                    "reward_-{}.0_frac".format(self.time_scales[i])]:
                metric_key = "{}{}".format(layer_prefix, postfix)
                if metric_key in eval_data.keys():
                    logs += [(metric_key, eval_data[metric_key])]

            q_prefix = "{}q".format(layer_prefix)
            if q_prefix in eval_data.keys():
                if len(eval_data[q_prefix]) > 0:
                    logs += [("{}avg_q".format(layer_prefix), np.mean(eval_data[q_prefix]))]
                else:
                    logs += [("{}avg_q".format(layer_prefix), 0.0)]

        if prefix != '' and not prefix.endswith('/'):
            new_logs = []
            for key, val in logs:
                if not key.startswith(prefix):
                    new_logs += [((prefix + '/' + key, val))]
                else:
                    new_logs += [(key, val)]
            logs = new_logs

        return logs
    
    def add_eval_data(self, eval):
        for key in eval:
            if key in self.eval_data:
                self.eval_data[key] += eval[key]
            else:
                self.eval_data[key] = eval[key]

    def clear_history(self):
        self.success_history.clear()
        self.custom_histories.clear()
        if hasattr(self, 'eval_data'):
            self.eval_data.clear()
