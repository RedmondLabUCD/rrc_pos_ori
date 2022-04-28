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
#%%
import math
print(math.radians(90))
#%%
import os
print(os.path.abspath(os.path.curdir))
#%%
from scipy.spatial.transform import Rotation as R
x = [0,0,0,1]
y = [-0.5,0,0.5,1]
#%%
from scipy.spatial.transform import Rotation as R
a = [[0,0,0,1],
     [0,0,0,1],
     [0,0,0,1]]

b = [[0.19134172, 0.46193977 ,0.19134172 ,0.8446232],
     [0.19134172, 0.46193977 ,0.19134172 ,0.8446232],
     [0.19134172, 0.46193977 ,0.19134172 ,0.8446232]
     ]

a = R.from_quat(a)
b = R.from_quat(b)
error_rot = a.inv() * b
print(error_rot)
orientation_error = error_rot.magnitude()
print(orientation_error)
#%%
x = R.from_quat(x)
y = R.from_quat(y)
error_rot = x.inv() * y
orientation_error = error_rot.magnitude()
print(orientation_error)
r1 = x.as_euler('xyz',degrees=True)
r2 = y.as_euler('xyz',degrees=True)
q1 = R.from_euler('xyz',r1,degrees=True).as_quat()
q2 = R.from_euler('xyz',r2,degrees=True).as_quat()
print(r1)
print(r2)
print(q1)
print(q2)
#%%
from scipy.spatial.transform import Rotation as R
r1 = [45,45,45]
r2 = [0,0,0]
q1 = R.from_euler('xyz',r1,degrees=True).as_quat()
q2 = R.from_euler('xyz',r2,degrees=True).as_quat()
print(q1)
print(q2)
print('--'*30)
x = R.from_quat(q1)
y = R.from_quat(q2)
error_rot = x.inv() * y
orientation_error = error_rot.magnitude()
print(error_rot.as_euler('xyz',degrees=True))
print(orientation_error)
print('--'*30)
r1 = x.as_euler('xyz',degrees=True)
r2 = y.as_euler('xyz',degrees=True)

q1 = R.from_euler('xyz',r1,degrees=True).as_quat()
q2 = R.from_euler('xyz',r2,degrees=True).as_quat()
print(r1)
print(r2)
print('--'*30)
print(q1)
print(q2)
#%%
a = [1,2,3,4,5,6,7]
print(a[3:])
#%%
print(0.9000000 == 0.9)
