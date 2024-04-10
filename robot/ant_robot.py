from robot.robot import Joint, Robot
import numpy as np

# TODO I've just put max_velocity as 1000 for now! Find out what it should really be.
ant_joints = [
    Joint("hip_1", (-0.5236, 0.5236),1000, 50.0),
    Joint("ankle_1", (0.5236, 1.2217),1000, 50.0),
    Joint("hip_2", (-0.5236, 0.5236),1000, 50.0),
    Joint("ankle_2", (-1.2217,-0.5236),1000, 50.0),
    Joint("hip_3", (-0.5236, 0.5236),1000, 50.0),
    Joint("ankle_3", (-1.2217,-0.5236),1000, 50.0),
    Joint("hip_4", (-0.5236, 0.5236),1000, 50.0),
    Joint("ankle_4", (0.5236, 1.2217),1000, 50.0),
]

# Mujoco - joint ranges are in degrees rather than radians
ant_small_joints = [
    Joint("hip_1", (-30.0, 30.0),1000, 1.0),
    Joint("ankle_1", (30.0, 70.0),1000, 1.0),
    Joint("hip_2", (-30.0, 30.0),1000, 1.0),
    Joint("ankle_2", (-70.0, -30.0),1000, 1.0),
    Joint("hip_3", (-30.0, 30.0),1000, 1.0),
    Joint("ankle_3", (-70.0, -30.0),1000, 1.0),
    Joint("hip_4", (-30.0, 30.0),1000, 1.0),
    Joint("ankle_4", (30.0, 70.0),1000, 1.0),
]


ant_velocity_bounds = np.array([(-500,500),(-500,500),(-500,500),(0,0),(0,0),(0,0)])

ant_small_velocity_bounds = np.array([(-3,3),(-3,3),(-3,3),(0,0),(0,0),(0,0)])

Ant = Robot('ant', ant_joints, ant_velocity_bounds)
AntF = Robot('ant', ant_joints, ant_velocity_bounds, use_forces=True)
AntSmall = Robot('ant', ant_small_joints, ant_small_velocity_bounds)
AntSmallF = Robot('ant', ant_small_joints, ant_small_velocity_bounds, use_forces=True)