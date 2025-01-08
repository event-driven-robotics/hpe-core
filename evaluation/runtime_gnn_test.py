#%% BAR GRAPH FULL
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import pandas as pd
from numpy import inf

runtime_graph_log = '/home/cpham-iit.local/code/wp5-gamer/hpe-gnn/run_time/total_time_making_graph.csv'
df_sample_graph = pd.read_csv(runtime_graph_log, sep='\s+')
# print(df_sample_graph)
df_no_graph = df_sample_graph.iloc[:,1].values
# df_no_graph = [float(x) for x in df_no_graph]
df_total_time = df_sample_graph.iloc[:,2].values
# print(type(df_no_graph))
time_per_graph = df_total_time/df_no_graph
time_per_graph[time_per_graph == inf] = 0
#Plot 
plt.figure(figsize=(10,6))
print('max time per graph:', np.max(time_per_graph))
plt.plot(time_per_graph, color ='skyblue', label = 'time per graph')
plt.axhline(y = np.mean(time_per_graph), color='red', linestyle = '--', label = 'averge time per graph')
plt.text(1, np.mean(time_per_graph), str(np.mean(time_per_graph)))
plt.title('Time to create a graph in all samples')
plt.ylabel('Time [s]')
plt.xlabel('Samples')
plt.legend(loc="upper right")
        #  plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        #  plt.grid(axis='y', alpha=0.75)
# plt.tight_layout()
plt.savefig('/home/cpham-iit.local/code/LEDGE/example/cam2_S9_Directions/time_per_graph.png', bbox_inches='tight')
plt.show()
