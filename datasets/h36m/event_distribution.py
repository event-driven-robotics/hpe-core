import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import os
from collections import Counter
from bimvee.importIitYarp import importIitYarp

dvs_file_path = '/home/cpham-iit.local/data/h36m/EV2/cam2_S11_Waiting/ch0dvs'

dvs_data = importIitYarp(filePathOrName=dvs_file_path)
events = dvs_data['data']['left']['dvs']
# print(events['ts'][0:100])
# fig, ax = plt.subplots()
# ax.hist(events['ts'], range = (events['ts'].min(), events['ts'].max()), bins = )
# plt.show()
value_counts = Counter(events['ts'])

unique_values = list(value_counts.keys())
# print(unique_values[:100])
counts = list(value_counts.values())
event_rate = [x / 0.001 for x in counts]

plt.figure(figsize=(10,6))
plt.bar(unique_values, event_rate, color ='skyblue')
plt.title('Event distribution')
plt.xlabel('Time [s]')
plt.ylabel('Event rate [events/s]')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()
