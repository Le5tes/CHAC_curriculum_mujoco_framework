from goal_conditioned_baselines.chac.utils import EnvWrapper
import numpy as np


def env_factory(env_constructor, alg_config):
    def constructor(robot, policy, step_fn, config):
        env = env_constructor(robot, policy, step_fn, config)
        return GCB_Wrapper(env, config, alg_config)
    return constructor

class GCB_Wrapper(EnvWrapper):
    def __init__(self, env, config):
        self.config = config
        self.wrapped_env = env
        self.graph = self.visualize = self.wrapped_env.env.render
        self.state_dim = len(env.state_bounds)
        self.goal_bounds = env.state_bounds[config.state_goal_indices]
        self.end_goal_dim = len(self.goal_bounds)
        self.subgoal_bounds = env.state_bounds[config.state_subgoal_indices]
        self.subgoal_dim = len(self.subgoal_bounds)
        self.action_dim = len(env.robot.joints)
        self.action_bounds = env.action_bounds
        self.action_offset = env.robot.joints_mid
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]
        
        for key in self.config.data:
            setattr(self, key, self.config[key])

        time_scales = np.array([int(t) for t in config.time_scales.split(',')])
        self.max_actions = np.prod(time_scales)

    def execute_action(self, action):
        obs = self.wrapped_env.step(action)
        return obs["observation"]
    
    def set_view(self,render):
        self.wrapped_env.set_view(render)

    @property
    def step_ctr(self):
        return self.wrapped_env.step_ctr

    def start(self):
        return self.wrapped_env.start()

    def step(self):
        return self.wrapped_env.step()
    
    def reset(self):
        return self.wrapped_env.reset()
    
    def finish(self):
        self.wrapped_env.finish()
    
    def update_policy(self, policy):
        self.wrapped_env.update_policy(policy)
    
    def done(self, state, fallen):
        return self.wrapped_env.done(state, fallen)
    
    def reward(self, *args):
        return self.wrapped_env.reward(*args)
    
    def goal(self):
        return self.wrapped_env.goal()
    
    def project_state_to_end_goal(self, state):
        if state is None:
            return np.zeros(len(self.config.state_goal_indices))
        return np.array(state)[self.config.state_goal_indices]

    def project_state_to_sub_goal(self, state):
        if state is None:
            return np.zeros(len(self.config.state_subgoal_indices))
        return np.array(state)[self.config.state_subgoal_indices]
