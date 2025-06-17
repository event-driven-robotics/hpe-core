#functions for use with vicon processing

import numpy as np
import math

def makeT(Rot, Trans):

    Rot = np.array(Rot) * (math.pi / 180.0)

    yawMatrix = np.matrix([
    [math.cos(Rot[2]), -math.sin(Rot[2]), 0],
    [math.sin(Rot[2]), math.cos(Rot[2]), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [math.cos(Rot[1]), 0, math.sin(Rot[1])],
    [0, 1, 0],
    [-math.sin(Rot[1]), 0, math.cos(Rot[1])]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(Rot[0]), -math.sin(Rot[0])],
    [0, math.sin(Rot[0]), math.cos(Rot[0])]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    T = np.zeros((4,4), dtype = np.float64)
    T[0:3, 0:3] = R
    T[3,3] = 1.0
    Trans = (R @ Trans.transpose()).transpose()
    T[0,3] = -Trans[0]
    T[1,3] = -Trans[1]
    T[2,3] = -Trans[2]
    return T

def marker_p(c3d_labels, c3d_points, mark_name):
        
    i = 0
    for label in c3d_labels:
        k = label.find(':') + 1
        if(label[k:k+4] == mark_name):
            break
        i = i + 1
    #marker_p = lambda index: np.array([np.append(val[index][0:3], 1) for val in points_3d.values()])

    return np.array([np.append(val[i][0:3], 1) for val in c3d_points])

def calc_indices(e_ts, period, delay):

    time_tags = np.arange(e_ts[0], e_ts[-1], period)
    index_tags = np.empty(len(time_tags))
    tags = zip(time_tags, index_tags)
    i = 0
    for tag in tags:
        while e_ts[i] < tag[0]:
            i = i + 1
        print(e_ts[i])
        tag[1] = i

    return tags
        


