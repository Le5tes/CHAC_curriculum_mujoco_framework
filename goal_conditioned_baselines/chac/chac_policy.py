from environment.GCB_wrapper import GCB_Wrapper
from environment.mujoco_env import MujocoEnvironment
from goal_conditioned_baselines.utils import (store_args)
from goal_conditioned_baselines.policy import Policy
import numpy as np
from goal_conditioned_baselines.chac.layer import Layer
from goal_conditioned_baselines.chac.utils import prepare_env
from goal_conditioned_baselines import logger
import torch
import time

from robot.ant_robot import AntSmallF

env_to_load = {'env':None}
def set_env_to_load(value):
    env_to_load['env'] = value


class CHACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, T, rollout_batch_size, agent_params, env, verbose=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)
        
        # TODO: torch.cuda.is_available() returning true on laptop where cuda is not available - gives segfault!
        if torch.cuda.is_available():
            print('cuda +')
            self.device = torch.device('cuda')
            logger.info('Running on GPU: {} {}'.format(torch.cuda.current_device(),
                  torch.cuda.get_device_name(torch.cuda.current_device())))
        else:
            print('cuda -')
            self.device = torch.device('cpu')
            logger.info('Running on CPU ...')
        # self.device = torch.device("cpu")
        self.verbose = verbose
        self.n_levels = agent_params['n_levels']
        self.pre_episodes = agent_params['n_pre_episodes']
        self.env = env
        self.fw = agent_params['fw']
        self._create_networks(agent_params)

        # goal_array stores goal for each layer of agent.
        self.goal_array = [None] * self.n_levels
        self.current_state = None
        self.steps_taken = 0
        self.total_steps = 0
        self.env_config = None

    def check_goals(self, env):
        """Determine whether or not each layer's goal was achieved. Also, if applicable, return the highest level whose goal was achieved."""
        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False] * self.n_levels
        max_lay_achieved = None

        # Project current state onto relevant goal spaces
        proj_end_goal = env.project_state_to_end_goal(self.current_state)
        proj_subgoal = env.project_state_to_sub_goal(self.current_state)

        for i in range(self.n_levels):
            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.n_levels - 1:
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds), \
                        "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"
                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.sub_goal_thresholds), \
                        "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"
                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    # print('state', proj_subgoal, 'subgoal', self.goal_array[i])
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.sub_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    def set_train_mode(self):
        self.test_mode = False

    def set_test_mode(self):
        self.test_mode = True

    def _create_networks(self, agent_params):
        logger.info("Creating a CHAC agent")
        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]
        # Create agent with number of levels specified by user
        self.layers = [
            Layer(i, self.env, agent_params, self.device)
            for i in range(self.n_levels)
        ]

    def learn(self, num_updates):
        """Update actor and critic networks for each layer"""
        return [
            self.layers[i].learn(num_updates) for i in range(len(self.layers))
        ]
    
    def top_layer(self):
        return self.layers[self.n_levels -1]

    def train(self, episode_num, eval_data, num_updates, skip_learn = False, queued_buffer = None):
        """Train agent for an episode"""
        # env.set_view(epoch_num is not None and epoch_num % 10 == 0 and episode_num == 50)

        obs = self.env.reset()
        self.current_state = obs['observation']

        if self.verbose:
            print("Initial State: ", self.current_state[:3])

        self.goal_array[self.n_levels - 1] = obs['desired_goal']
        self.env.wrapped_env.final_goal = obs['desired_goal']

        if self.verbose:
            print("Next End Goal: ", self.env.wrapped_env.final_goal, self.env.wrapped_env.goal)

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, eval_data, max_lay_achieved = self.top_layer().\
            train(self, self.env, episode_num=episode_num, eval_data=eval_data, queued_buffer=queued_buffer)

        train_duration = 0
        # Update networks if not testing and enough episodes finished
        if not self.test_mode and episode_num > self.pre_episodes and not skip_learn:
            train_start = time.time()
            learn_summaries = self.learn(num_updates)

            for l in range(self.n_levels):
                learn_summary = learn_summaries[l]
                for k, v in learn_summary.items():
                    eval_data["train_{}/{}".format(l, k)] = v

            train_duration += time.time() - train_start

        self.total_steps += self.steps_taken
        # Return whether end goal was achieved
        return goal_status[self.n_levels - 1], eval_data, train_duration

    def logs(self, prefix=''):
        logs = []
        logs += [('steps', self.total_steps)]

        if prefix != '' and not prefix.endswith('/'):
            logs = [(prefix + '/' + key, val) for key, val in logs]

        return logs

    def __getstate__(self):
        excluded_subnames = [
            '_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats', 'main',
            'target', 'lock', 'env', 'sample_transitions', 'stage_shapes',
            'create_actor_critic', 'obs2preds_buffer', 'obs2preds_model',
            'eval_data', 'layers', 'goal_array', 'total_steps',
            'current_state', 'device'
        ]

        state = {
            k: v
            for k, v in self.__dict__.items()
            if all([not subname in k for subname in excluded_subnames])
        }
        state['torch'] = {}

        state['env_config'] = self.env_config

        # save pytoch model weights
        for layer in self.layers:
            l = str(layer.level)
            state['torch']['actor' + l] = layer.actor.cpu().state_dict()
            state['torch']['critic' + l] = layer.critic.cpu().state_dict()
            if hasattr(layer, 'state_predictor'):
                state['torch']['fw_model' + l] = layer.state_predictor.cpu().state_dict()
                state['fw_model' + l + 'num_errs'] = layer.state_predictor.num_errs
                state['fw_model' + l + 'min_err'] = layer.state_predictor.min_err
                state['fw_model' + l + 'max_err'] = layer.state_predictor.max_err

            # move back, just in case
            layer.actor.to(self.device)
            layer.critic.to(self.device)
            if hasattr(layer, 'state_predictor'):
                layer.state_predictor.to(self.device)

        return state

    def __setstate__(self, state):

        state['env']= env_to_load['env']

        self.__init__(**state)
        self.env.agent = self

        # load network states
        for layer in self.layers:
            l = str(layer.level)
            layer.actor.load_state_dict(state['torch']['actor' + l])
            layer.critic.load_state_dict(state['torch']['critic' + l])
            if hasattr(layer, 'state_predictor'):
                layer.state_predictor.load_state_dict(state['torch']['fw_model' + l])

                layer.state_predictor.num_errs = state['fw_model' + l + 'num_errs']
                layer.state_predictor.min_err = state['fw_model' + l + 'min_err']
                layer.state_predictor.max_err = state['fw_model' + l + 'max_err']
