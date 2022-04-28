#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:35:53 2022

@author: qiang
"""

from ament_index_python.packages import get_package_share_directory
import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils
import os
import numpy as np

angular_joint_positions = np.array([[0, 0.9,-1.7,0, 0.9, -1.7,0, 0.9, -1.7],
                                    [0, 1,-1,0, 0.2, -0.7,0, 0.9, -1.7]])
# angular_joint_positions = np.array([0, 0.9,-1.7,0, 0.9, -1.7,0, 0.9, -1.7])

def init_kinematics():
    """Initialise the kinematics calculator for TriFingerPro."""
    robot_properties_path = get_package_share_directory(
        "robot_properties_fingers"
    )
    urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf(
        "trifingerpro"
    )
    finger_urdf_path = os.path.join(robot_properties_path, "urdf", urdf_file)
    kinematics = trifinger_simulation.pinocchio_utils.Kinematics(
        finger_urdf_path,
        ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"],
    )
    return kinematics

kin = init_kinematics()


tip_poses = []
for angular_pos in angular_joint_positions:
    tip_poses.append(kin.forward_kinematics(angular_pos))
print(tip_poses)
#%%
import numpy as np
a = np.array([[[0.08600000000000001, 0.06105533099647374, 0.0790693315811472], [0.009875467679413746, -0.1050058502236986, 0.0790693315811472], [-0.09587546767941375, 0.0439505192272248, 0.0790693315811472]], 
     [[0.08600000000000001, 0.18513535756926347, 0.043551631061097607], [-0.03816843842614979, -0.0772676881007304, -0.007223862357058325], [-0.09587546767941375, 0.0439505192272248, 0.0790693315811472]],
     [[0.41200001, 0.14316926347, 0.04497607], [-0.116842614979, -0.3676881007304, -0.32357058325], [-0.41375, 0.0439504272248, 0.490693315811472]]])
print(a[...,0][...,0:3])