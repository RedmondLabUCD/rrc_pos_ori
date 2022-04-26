#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:41:05 2022

@author: qiang
"""
import pandas as pd

class CsvCreator():
    def __init__(self,
                 csv_type = 'pos_ori',
                 history = None,
                 ):
        if csv_type == 'pos_ori':
            self.title = ['epoch','eval_rate','eval_pos_rate','eval_ori_rate','explore_rate','explore_pos_rate','explore_ori_rate',\
                          'a_loss','q_loss','rrc','rrc_pos','rrc_ori','z_mean','xy']
        self.data = []
        
    def update(self,log,path="log.csv"):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data,columns=self.title)
        dataframe.to_csv(path,index=False,sep=',')
