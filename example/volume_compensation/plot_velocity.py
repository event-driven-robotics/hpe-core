#BSD 3-Clause License
#Copyright (c) 2021, Event Driven Perception for Robotics
#All rights reserved.

import matplotlib.pyplot as plt
import numpy as np

#%%
input_file_path = "/home/aglover/imu_out_temp.txt"
data = np.loadtxt(input_file_path);
data[:, 0] = data[:, 0] - data[0, 0]
labels = ["X", "Y", "Z", "PITCH", "YAW", "ROLL"]

#%%
plt.rcParams.update({'font.size': 12})
#plt.figure(figsize=(12, 9))
fig,axs = plt.subplots(6)
#fig.set_figsize(12, 9)
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 0.75)
axs[2].set_ylabel('velocity')
axs[5].set_xlabel('time')
plt.draw()
plt.pause(0.01)

for i in range(0, 6):
    axs[i].plot(data[:, 0], data[:, i+1], label=labels[i])
    axs[i].legend(loc="upper right")

for ax in axs:
    ax.label_outer()

plt.show()

