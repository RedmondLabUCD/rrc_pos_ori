# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 12:30:22 2022

@author: 14488
"""
import numpy as np
a = [[1],
     [2],
     [3]]

print(np.linalg.norm(a,ord=1, axis=-1))
#%%
print(a[3:])

#%%
a = np.array([1])
b = a + a
print(b)
#%%
a = []
#%%
import numpy as np
a = [1,2,3,4,5,6]
print(a[3:5])