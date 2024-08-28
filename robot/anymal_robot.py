from robot.robot import Joint, Robot
import numpy as np

# joint ranges for anymal are in radians
anymal_joints = [
Joint("LF_HAA_joint",(-0.72,0.49),1000,2),
Joint("LF_HFE_joint",(-3.14,3.14),1000,2),
Joint("LF_KFE_joint",(-3.14,3.14),1000,2),
Joint("RF_HAA_joint",(-0.49, 0.72),1000,2),
Joint("RF_HFE_joint",(-3.14,3.14),1000,2),
Joint("RF_KFE_joint",(-3.14,3.14),1000,2),
Joint("LH_HAA_joint",(-0.72, 0.49),1000,2),
Joint("LH_HFE_joint",(-3.14,3.14),1000,2),
Joint("LH_KFE_joint",(-3.14,3.14),1000,2),
Joint("RH_HAA_joint",(-0.49, 0.72),1000,2),
Joint("RH_HFE_joint",(-3.14,3.14),1000,2),
Joint("RH_KFE_joint",(-3.14,3.14),1000,2)
]

anymal_velocity_bounds = np.array([(-3,3),(-3,3),(-3,3),(0,0),(0,0),(0,0)])

Anymal = Robot('anymal', anymal_joints, anymal_velocity_bounds)