#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:41:05 2022

@author: qiang
"""
import pandas as pd
from ament_index_python.packages import get_package_share_directory
import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils
import numpy as np
import os
import torch

class CsvCreator():
    def __init__(self,
                 csv_type = 'pos_ori',
                 history = None,
                 ):
        if csv_type == 'pos_ori':
            self.title = ['epoch','eval_rate','eval_pos_rate','eval_ori_rate','explore_rate','explore_pos_rate','explore_ori_rate',\
                          'a_loss','q_loss','rrc','rrc_pos','rrc_ori','z_mean','xy', 'ori']
        self.data = []
        
    def update(self,log,path="log.csv"):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data,columns=self.title)
        dataframe.to_csv(path,index=False,sep=',')

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

def process_inputs(o, g, o_mean, o_std, g_mean, g_std):
    clip_obs = 200
    clip_range = 5
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs