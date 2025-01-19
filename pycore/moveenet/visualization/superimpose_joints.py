import cv2
import pandas as pd
import numpy as np

# File paths
csv_file = '/home/cpham-iit.local/data/h36m/ledge/raw/cam2_S11_Directions_1/ledge.csv'
# video_file = '/home/cpham-iit.local/data/h36m/videos/GT_RGB_video/cam4_S9_Phoning.mp4'
# output_video = '/home/cpham-iit.local/data/h36m/videos/superimposed_video.mp4'

# # Read the CSV file
# data = pd.read_csv(csv_file)
# # print(data)
# # exit()
# # Ensure the data is sorted by timestamp
# data = data.sort_values(data.iloc[:,0]).reset_index(drop=True)

# # Extract joint coordinates and timestamps
# timestamps = data[data[:,0]].values
# joint_coordinates = data.iloc[:, 2:].values.reshape(-1, 13, 2)  # Reshape to (N, 13, 2)

# # Open the video file
# cap = cv2.VideoCapture(video_file)
# if not cap.isOpened():
#     raise Exception("Error: Cannot open the video file.")

# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # Define the codec and create VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# # Iterate through video frames
# frame_idx = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Get the corresponding timestamp
#     video_time = frame_idx / fps * 1000  # Video time in milliseconds

#     # Find the closest timestamp in the data
#     closest_idx = np.argmin(np.abs(timestamps - video_time))
#     joints = joint_coordinates[closest_idx]

#     # Draw the joints on the frame
#     for joint in joints:
#         x, y = int(joint[0]), int(joint[1])
#         cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for joints

#     # Write the frame to the output video
#     out.write(frame)

#     # Increment frame index
#     frame_idx += 1

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Superimposed video saved at: {output_video}")
df_ledge_input = pd.read_csv(csv_file)
df_ledge_input = df_ledge_input.iloc[0:1].values
df_ledge_input = [x for x in df_ledge_input]

print(df_ledge_input)

# try:
#     predictions_old = np.loadtxt(csv_file, dtype=float)
# except ValueError:
#     with open(csv_file) as f:
#         content = f.readlines()
#     number_of_columns = len(content[0].split(','))
    
#     # Initialize predictions_old with zeros
#     predictions_old = np.zeros((len(content), number_of_columns))
#     for l, line in enumerate(content):
#         predictions_old[l,:] = np.asarray(line.split(','))
# predictions_old = predictions_old[predictions_old[:, 0].argsort()]