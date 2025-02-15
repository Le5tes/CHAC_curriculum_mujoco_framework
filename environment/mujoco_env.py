from mujoco_sim.mujoco_simulation import MJSimulation
from collections import deque
import numpy as np

class MujocoEnvironment:
    def __init__(self, robot, config, logger):
        self.config = config
        self.env = MJSimulation(robot.name, config.render, config.include_env_in_state, config.include_larger_features, config.larger_feature_difficulty_scaling)
        self.robot = robot
        self.action_bounds = robot.joints_scale
        self.state_bounds = self.get_state_bounds(robot, config.include_env_in_state)
        self.intensity = self.config.start_intensity
        self.successes = deque([], self.config.num_successes_to_increment)
        self.step_ctr = 0
        self.logger = logger

    def start(self):
        return self.reset()

    def step(self, action):
        for _ in range(self.config.step_size):
            observation = self.env.step(action)
        self.step_ctr += 1
        return self.assemble_obs(observation)

    def reset(self):
        self.is_done = False
        self.update_intensity()
        observation = self.env.reset(self.intensity)
        self.terminated = self.truncated = False
        self.step_ctr = 0
        return self.assemble_obs(observation)

    def update_intensity(self):
        if self.config.increasing_difficulty and len(self.successes) == self.config.num_successes_to_increment:
            if all(self.successes):
                self.intensity += self.config.intensity_increment
                self.logger.info(f"increasing intensity to {self.intensity}")
            elif not any(self.successes) and self.intensity >= self.config.intensity_increment:
                self.intensity -= self.config.intensity_increment
                self.logger.info(f"decreasing intensity to: {self.intensity}")

    def finish(self):
        pass

    def set_view(self, render):
        self.env.render = render

    def update_policy(self, policy):
        self.policy = policy

    def done(self, state, fallen):
        return self.is_done
    
    def reward(self, *args):
        return 0
    
    def goal(self):
        return self.env.goal

    def update_successes(self,success):
        self.successes.append(success)

    def get_state_bounds(self, robot, include_env_in_state):
        arena_size_bound = (-self.config.arena_size, self.config.arena_size)
        area_bounds = np.array([arena_size_bound, arena_size_bound, arena_size_bound])

        # TODO these values are for Ant - found empirically 
        orientation_bounds = np.array([(-1,1),(-1,1),(-1,1),(-1,1)]) # These must be the same for all robots
        velocity_bounds = robot.velocity_bounds

        joint_bounds = np.concatenate([np.array([joint.move_range, (-joint.max_velocity, joint.max_velocity)]) for joint in robot.joints]) # mujoco doesn't return joint effort

        state_bounds_parts = [area_bounds, orientation_bounds, velocity_bounds, joint_bounds]

        if include_env_in_state:
            terrain_bounds = np.array([(-10, 10) for _ in range(21 * 21)])
            state_bounds_parts.append(terrain_bounds)

        return np.concatenate(state_bounds_parts)
    
    def assemble_obs(self, observation):
        return {
            "observation": observation,
            "achieved_goal": observation[:2],
            "desired_goal": self.goal()
        }