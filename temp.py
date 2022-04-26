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
#%%
from scipy.spatial.transform import Rotation as R
import numpy as np
a = np.array([[0,0,0.447, 0.894],
              [0,0.5,0,1],
              [0.5,0,0,1]])

rotation_d = R.from_quat(a)
ori_g = rotation_d.as_euler('xyz', degrees=True)
print(ori_g.type())
#%%
import numpy as np
gg = np.array([0,3.1415926,0,0,0,0,0])
print(np.cos(gg))
#%%
import math
print(np.cos(math.radians(22)))
#%%
print(np.cos(math.radians(66 * (3/3))))

#%%
import numpy as np
print(np.random.uniform(0,1.0))
#%%
import numpy as np
q = 0
for _ in range(10000):
    radnum = np.random.uniform(0,1.0)
    if radnum >= 0.6:
        q += 1
print(q)