import random
import os
cwd = os.getcwd()
from collections import namedtuple
import numpy as np

class Robot:
    def __init__(self, name, joints, velocity_bounds, passive_joints = [],  use_forces=False, use_pose=True, include_effort=False):
        # Need to work out the state and action space from the urdf file
        self.name = name
        self.joints = joints
        self.velocity_bounds = velocity_bounds # how fast the robot can move in each axis
        self.passive_joints = passive_joints
        self.use_forces=use_forces 
        self.use_pose=use_pose

        self.state_per_joint = 3 if include_effort else 2 # position, velocity, +- effort for each joint

        self.joints_mid = np.array([0 if use_forces else (joint.move_range[0] + joint.move_range[1])/2 for joint in self.joints])
        self.joints_scale = np.array([joint.max_force if use_forces else (joint.move_range[1] - joint.move_range[0])/2 for joint in self.joints])
        self.joints_range = [[joint.move_range[0] for joint in self.joints], [joint.move_range[1] for joint in self.joints]]
        
    def get_state_length(self):
        pose_state_size = 13 # 3 positional, 4 orientation, 3 velocity and 3 angular velocity
        joints_state_size = self.state_per_joint  * (len(self.joints) + len(self.passive_joints))
        return pose_state_size + joints_state_size if self.use_pose else joints_state_size


    def rictus_policy(self):
        def policy(state, goal):
            return [0.0 for _ in self.joints]
        return policy

    def twitchy_policy(self):
        def policy(state, goal):
            if self.use_forces:
                return [random.uniform(-joint.max_force, joint.max_force) for joint in self.joints]
            else:
                return [random.uniform(joint.move_range[0], joint.move_range[1]) for joint in self.joints]
        return policy
    
    def max_twitch_policy(self):
        def policy(state, goal):
            return [random.choice([joint.move_range[0], joint.move_range[1]]) for joint in self.joints]
        return policy


Joint = namedtuple('Joint',['name', 'move_range', 'max_velocity', 'max_force'])