# %%
import numpy as np
from bimvee.importIitYarp import importIitYarp
import cv2

# %%
ts_start = 17.0 #s
ts_duration = 0.02 #s
img_shape = (480, 640)

data_dict = importIitYarp(filePathOrName='/home/schiavazza/data/hpe/vicon_recordings/giovanna/1/')

# %%
dvs_data = data_dict['data']['left']['dvs']

id_start = np.searchsorted(dvs_data['ts'], ts_start)
id_end = np.searchsorted(dvs_data['ts'], ts_start + ts_duration)

img = np.zeros(img_shape)

img[
    dvs_data['y'][id_start:id_end],
    dvs_data['x'][id_start:id_end]
] = 255

cv2.imwrite('/home/schiavazza/code/hpe/vicon_recordings/data/frame.png', img)


# %%
import matplotlib.pyplot as plt
times = data_dict['data']['left']['dvs']['ts']
plt.hist(times, 100);
# %%
