'''TransformHelpers.py

   These are helper functions for rotation and transform matrices.

'''

import numpy as np

# Compose p, R, and T:
def p(x,y,z):
    return np.array([[x],[y],[z]])
def T(p,R):
    return np.vstack((np.hstack((R,p)),
                      np.array([0,0,0,1])))

def Rotx(alpha):
    return np.array([[1, 0            ,  0            ],
                     [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha),  np.cos(alpha)]])
def Roty(alpha):
    return np.array([[ np.cos(alpha), 0, np.sin(alpha)],
                     [ 0            , 1, 0            ],
                     [-np.sin(alpha), 0, np.cos(alpha)]])
def Rotz(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha),  np.cos(alpha), 0],
                     [0            ,  0            , 1]])
def Reye():
    return np.eye(3)

# Decompose T to p and R:
def pfromT(T):
    return T[0:3,3:4]
def RfromT(T):
    return T[0:3,0:3]
