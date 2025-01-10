import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import os
from collections import Counter
from bimvee.importIitYarp import importIitYarp
import seaborn as sns
import pandas as pd
import csv 
# dvs_file_path = '/home/cpham-iit.local/data/h36m/EV2/cam2_S11_Waiting/ch0dvs'

# dvs_data = importIitYarp(filePathOrName=dvs_file_path)
# events = dvs_data['data']['left']['dvs']
# # print(events['ts'][0:100])
# # fig, ax = plt.subplots()
# # ax.hist(events['ts'], range = (events['ts'].min(), events['ts'].max()), bins = )
# # plt.show()
# value_counts = Counter(events['ts'])

# unique_values = list(value_counts.keys())
# # print(unique_values[:100])
# counts = list(value_counts.values())
# event_rate = [x / 0.001 for x in counts]

# plt.figure(figsize=(10,6))
# plt.bar(unique_values, event_rate, color ='skyblue')
# plt.title('Event distribution')
# plt.xlabel('Time [s]')
# plt.ylabel('Event rate [events/s]')
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.grid(axis='y', alpha=0.75)
# plt.tight_layout()
# plt.show()
#Load csv file
data_path = '/home/cpham-iit.local/code/LEDGE/example/cam2_S9_Directions/ledge_2.csv'
# data_path = '/home/cpham-iit.local/data/h36m/ledge/raw/cam2_S9_Directions/ledge.csv'
try:
    lsg = np.loadtxt(data_path, dtype=float)
except ValueError:
    # with open(data_path) as f:
    #     lines = []
    #     for line in f:
    #         lines.append(line.split(","))
    with open(data_path, 'r') as f:
         #read file
         content = f.readlines()
         N = float(content[0].split(',')[1]) # num of blocks
         ts = [] # processing time 
        #Calculate the upper bound delay time in LEDGE
         for line in content:
            delay_time = float(line.split(',')[0])
            ns = (len(line.split(',')[2:])-1)/11 #number of segments
            t_d = (delay_time * N)/ns
            ts.append(t_d)
            #  print(float(line.split(',')[1]))
            #  print((len(line.split(',')[2:])-1)/11)
         print('averge tracking line segments:',  np.mean(ts))
         upper_bound_delay_time = max(ts)
         time = np.linspace(0, 53.96, len(ts))
         plt.figure(figsize=(10,6))

         plt.plot(time, ts, color ='skyblue', label = 'tracking of line segments time')
         plt.axhline(y = upper_bound_delay_time, color='red', linestyle = '--', label = 'upper bound time')
         plt.axhline(y = np.mean(ts), color='green', linestyle = '--', label = 'averge time')
         plt.title('LEDGE running time at a random sample')
         plt.ylabel('[s]')
         plt.xlabel(' Timestamp [s]')
         plt.legend(loc="upper right", bbox_to_anchor = (0, 0.85, 1, 0.075))
        #  plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        #  plt.grid(axis='y', alpha=0.75)
         plt.tight_layout()
         plt.savefig('/home/cpham-iit.local/code/LEDGE/example/cam2_S9_Directions/ledge_delay.png', bbox_inches='tight')
         plt.show()