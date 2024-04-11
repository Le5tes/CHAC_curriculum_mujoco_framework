import mujoco
from training_ground.terrain_generator import Point
import training_ground.terrain_generators_heightmaps as hms
import numpy as np
import os
import sys
# viewer_installed = 'mujoco_viewer' in sys.modules
viewer_installed = False
if viewer_installed:
    from mujoco_viewer import MujocoViewer


class MJSimulation:
    def __init__(self, robot, render = False, include_env_in_state = False):
        if robot == 'ant':
            xml_file_name = "ant-environment.xml"
        else:
            return print("robot not implemented")
        
        xml_file_path = os.path.join(os.path.dirname(__file__), xml_file_name)

        with open(xml_file_path, 'r') as file:
            xml_string = file.read()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.action_shape = self.data.ctrl.shape
        
        self.step_ctr = 0
        if render:
            if viewer_installed:
                self.render = render
                self.viewer = MujocoViewer(self.model, self.data)
                return
            else:
                print("viewer not installed")
        self.viewer = None
        self.render = False
        self.include_env_in_state = include_env_in_state

    def reset(self, intensity = 0.00):
        # print("steps taken", self.step_ctr)

        if self.render and self.viewer is None:
            self.viewer = MujocoViewer(self.model, self.data)
        else:
            if self.viewer:
                self.viewer.close()
                self.viewer = None

        self.step_ctr = 0
        self.choose_heightmap(intensity)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:2] = self.start_pos
        self.data.mocap_pos[0][:2] = self.goal
        self.data.mocap_pos[1][:2] = self.start_pos
        return self.state()

    def step(self, action):
        self.step_ctr += 1
        self.data.ctrl = action

        mujoco.mj_step(self.model, self.data)

        if self.render:
            self.viewer.render()

        return self.state()        

    def choose_heightmap(self, intensity):
        self.generate_start_and_goal(intensity)
        self.heightmap = hms.jagged_terrain(20,max(intensity - 0.1, 0), self.start_pos, self.goal).ravel() 

        self.model.hfield_data = self.heightmap
        if self.render:
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, self.model.hfield_adr)
    
    def state(self):
        state_parts = [self.data.qpos.copy(),self.data.qvel.copy()]

        if self.include_env_in_state:
            state_parts.append(self.heightmap)
        
        return np.concatenate(state_parts)
    
    def generate_start_and_goal(self, intensity):

        # intensity 0 - 0.09 => distance between 1 and 10
        intensity_distance = min(1 + 100 * intensity, 10)
        minval = -intensity_distance
        maxval = intensity_distance
        start,goal = np.zeros([2,2])
        while np.linalg.norm(start - goal) < intensity_distance:
            start,goal = np.random.uniform(minval,maxval,[2,2])
        self.start_pos, self.goal = start, goal
