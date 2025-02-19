import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import pandas as pd
import numpy as np
import re
import cv2

from evaluation.utils.visualization import add_skeleton

def viz_prediction_all_joints(algo_names, skeletons_predictions, skeletons_gt, ts, output_folder_path, ds_name):
     
     ds_name_video = re.sub(r'_ch0$', '', ds_name)
     print(ds_name_video)
     video_file= f'/home/cpham-iit.local/data/h36m/videos/GT_RGB_video/{ds_name_video}.mp4'
    #  output_video = '/home/cpham-iit.local/data/h36m/videos/superimposed_video.mp4'

     res = [480, 640]
     image = np.zeros(res, np.uint8)
     thickness = 2
     count = 0
     # Open the video file
     cap = cv2.VideoCapture(video_file)
     if not cap.isOpened():
        raise Exception("Error: Cannot open the video file.")
     # Get video properties
     fps = int(cap.get(cv2.CAP_PROP_FPS))
     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #  file_path = save_video
     file_path = output_folder_path / f'{ds_name}.mp4'
    #  frame_width = 640
    #  frame_height = 480
    #  fps = 30
     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     output = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))
     print('saving video')

     #Iterate through video frame
     frame_idx = 0
     timestamps = []
     while cap.isOpened():
         ret, frame = cap.read()
         if ret:
             timestamps.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
         else:
             break
         
         #Get cooresponding timestamp
         video_time = frame_idx / fps #Video in seconds
         #Find the closest timestamps in the data
         closest_idx = np.argmin(np.abs(ts - video_time))
         #switch button or sth to choose which one to visualize
         joints_openposeRGB = skeletons_predictions[0][closest_idx,:].flatten()
         joints_moveenet = skeletons_predictions[1][closest_idx, :].flatten()
         joints_hpegnn = skeletons_predictions[2][closest_idx, :].flatten()
        #  joints_singleweight = skeletons_predictions[3][closest_idx, :].flatten()
         joints_GT = skeletons_gt[closest_idx,:].flatten()
         #Draw joints on the frame
         frame = add_skeleton(frame, joints_openposeRGB, (255,0,0), lines = True, normalised= False)
         frame = add_skeleton(frame, joints_moveenet, (0,0,255), lines = True, normalised= False)
         frame = add_skeleton(frame, joints_hpegnn, (255,255,0), lines = True, normalised= False)
        #  frame = add_skeleton(frame, joints_singleweight, (0,255,255), lines = True, normalised= False)
         frame = add_skeleton(frame,joints_GT, (0,255,0), lines=True, normalised=False)
         cv2.line(frame, (500, 440),(520,440), (255,0,0), thickness) #Openpose
         cv2.putText(frame, 'OpenposeRGB', (540, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), int(thickness/2), cv2.LINE_AA)
         cv2.line(frame, (500, 460),(520,460), (0,0,255), thickness) #MoveENet
         cv2.putText(frame, 'MoveEnet', (540, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), int(thickness/2), cv2.LINE_AA)
         cv2.line(frame, (500, 380),(520,380), (255,255,0), thickness) #Ledgestable
         cv2.putText(frame, 'HPE-GNN', (540, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), int(thickness/2), cv2.LINE_AA)
        #  cv2.line(frame, (500, 400),(520,400), (0,255,255), thickness) #Ledgesingleweight
        #  cv2.putText(frame, 'GNN shared weight', (540, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), int(thickness/2), cv2.LINE_AA)
         cv2.line(frame, (500, 420),(520,420), (0,255,0), thickness) #GT
         cv2.putText(frame, 'GT', (540, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), int(thickness/2), cv2.LINE_AA)
         #Write the frame to the output video
         output.write(frame)

         #Increment frame idx 
         frame_idx += 1
    #  print(len(timestamps))
     cap.release()
     output.release()
     cv2.destroyAllWindows()
     
     return None

def viz_prediction_all_joints_one_algo(algo_names, skeletons_predictions, skeletons_gt, ts, output_folder_path, ds_name):
     
     ds_name_video = re.sub(r'_ch0$', '', ds_name)
     print(ds_name_video)
     video_file= f'/home/cpham-iit.local/data/h36m/videos/GT_RGB_video/{ds_name_video}.mp4'
    #  output_video = '/home/cpham-iit.local/data/h36m/videos/superimposed_video.mp4'

     res = [480, 640]
     image = np.zeros(res, np.uint8)
     thickness = 2
     count = 0
     # Open the video file
     cap = cv2.VideoCapture(video_file)
     if not cap.isOpened():
        raise Exception("Error: Cannot open the video file.")
     # Get video properties
     fps = int(cap.get(cv2.CAP_PROP_FPS))
     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #  file_path = save_video
     file_path = output_folder_path / f'{ds_name}.mp4'
    #  frame_width = 640
    #  frame_height = 480
    #  fps = 30
     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     output = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))
     print('saving video')

     #Iterate through video frame
     frame_idx = 0
     timestamps = []
     while cap.isOpened():
         ret, frame = cap.read()
         if ret:
             timestamps.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
         else:
             break
         #Get cooresponding timestamp
         video_time = frame_idx / fps #Video in seconds
         #Find the closest timestamps in the data
         closest_idx = np.argmin(np.abs(ts - video_time))
         #switch button or sth to choose which one to visualize
        #  joints_openposeRGB = skeletons_predictions[0][closest_idx,:].flatten()
         joints_moveenet = skeletons_predictions[0][closest_idx, :].flatten()
        #  joints_hpegnn = skeletons_predictions[2][closest_idx, :].flatten()
        #  joints_singleweight = skeletons_predictions[3][closest_idx, :].flatten()
        #  joints_GT = skeletons_gt[closest_idx,:].flatten()
         #Draw joints on the frame
        #  frame = add_skeleton(frame, joints_openposeRGB, (255,0,0), lines = True, normalised= False)
         frame = add_skeleton(frame, joints_moveenet, (0,0,255), lines = True, normalised= False)
        #  frame = add_skeleton(frame, joints_hpegnn, (255,255,0), lines = True, normalised= False)
        #  frame = add_skeleton(frame, joints_singleweight, (0,255,255), lines = True, normalised= False)
        #  frame = add_skeleton(frame,joints_GT, (0,255,0), lines=True, normalised=False)

        #  cv2.line(frame, (500, 440),(520,440), (255,0,0), thickness) #Openpose
        #  cv2.putText(frame, 'OpenposeRGB', (540, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), int(thickness/2), cv2.LINE_AA)
        #  cv2.line(frame, (500, 460),(520,460), (0,0,255), thickness) #MoveENet
        #  cv2.putText(frame, 'MoveEnet', (540, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), int(thickness/2), cv2.LINE_AA)
        #  cv2.line(frame, (500, 380),(520,380), (255,255,0), thickness) #Ledgestable
        #  cv2.putText(frame, 'HPE-GNN', (540, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), int(thickness/2), cv2.LINE_AA)
        #  cv2.line(frame, (500, 400),(520,400), (0,255,255), thickness) #Ledgesingleweight
        #  cv2.putText(frame, 'GNN shared weight', (540, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), int(thickness/2), cv2.LINE_AA)
        #  cv2.line(frame, (500, 420),(520,420), (0,255,0), thickness) #GT
        #  cv2.putText(frame, 'GT', (540, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), int(thickness/2), cv2.LINE_AA)
         #Write the frame to the output video
        #  output.write(frame)
         cv2.imwrite('/home/cpham-iit.local/data/h36m_full/videos/cam4_S9_WalkDog_moveEnet/image_{}.jpg'.format(frame_idx), frame)
         #Increment frame idx 
         frame_idx += 1
    #  print(len(timestamps))
     cap.release()
     output.release()
     cv2.destroyAllWindows()
     
     return None

hpecore_kps_labels = {'Head': 0,
                      'Shoulder_right': 1,
                      'Shoulder_left': 2,
                      'Hip_left': 3,
                      'Hip_right': 4,
                      'Elbow_right': 5,
                      'Elbow_left': 6,
                      'Wrist_right': 7,
                      'Wrist_left': 8,
                      'Knee_right': 9,
                      'Knee_left': 10,
                      'Ankle_right': 11,
                      'Ankle_left': 12
                      }
hpe_kps_labels_cluster = {'Torso': 0,
                          'Arms': 1,
                          'Legs': 2

}

def plot_pck_individual_joints(output_folder_path, ds_name=None):
    if ds_name == None:
        pck_path = output_folder_path/f'pck_0.4.txt'
    else:
        pck_path = output_folder_path/f'pck_0.4_{ds_name}.txt'
    # Load global PCK@04 results
    df_PCK = pd.read_csv(pck_path, sep='\s+')
    # df_PCK = pd.read_csv("/home/cpham-iit.local/data/output_hpe_20per/pck_0.4.txt", sep='\s+')
    df_PCK_movenet_cam_24 = df_PCK["movenet_cam-24"]
    df_PCK_movenet_cam_24 = df_PCK_movenet_cam_24.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # print(df_PCK_movenet_cam_24)
    df_PCK_movenet_cam_24 = df_PCK_movenet_cam_24.iloc[1:14].values
    df_PCK_movenet_cam_24 = [float(x) for x in df_PCK_movenet_cam_24]
    df_PCK_openpose_rgb = df_PCK["openpose_rgb"]
    df_PCK_openpose_rgb = df_PCK_openpose_rgb.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_PCK_openpose_rgb = df_PCK_openpose_rgb.iloc[1:14].values
    df_PCK_openpose_rgb = [float(x) for x in df_PCK_openpose_rgb]

    # df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK["hpeGnn_splineConv"]
    # # print(df_PCK_ledge10_solo_weight_contrib_stepwise)
    # df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK_ledge10_solo_weight_contrib_stepwise.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # # print(df_PCK_ledge10_solo_weight_contrib_stepwise)
    # df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK_ledge10_solo_weight_contrib_stepwise.iloc[1:14].values
    # # print(df_PCK_ledge10_solo_weight_contrib_stepwise)
    # df_PCK_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_PCK_ledge10_solo_weight_contrib_stepwise]

    df_PCK_hpegnn = df_PCK["hpe-gnn_two_weight_cone_only_target_connectivity_15"]
    df_PCK_hpegnn = df_PCK_hpegnn.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_PCK_hpegnn = df_PCK_hpegnn.iloc[1:14].values
    df_PCK_hpegnn = [float(x) for x in df_PCK_hpegnn]

    #Plot pck
    width = 0.12
    my_dpi = 96
    fig, ax = plt.subplots(figsize = (2048/my_dpi, 1200/my_dpi), dpi = my_dpi)
    ax.grid(axis = 'y')
    # plt.bar(np.arange(len(df_PCK_ledge10_solo_weight_contrib_stepwise)),df_PCK_ledge10_solo_weight_contrib_stepwise , width = width, label = 'ledge_single_weight') #S_1_1 = hpe_gnn_spline_conv_gamer
    ax.bar(np.arange(len(df_PCK_hpegnn)) - width,df_PCK_hpegnn , width=width, label = 'GraphEnet')
    ax.bar(np.arange(len(df_PCK_movenet_cam_24)), df_PCK_movenet_cam_24, width = width, label = 'moveEnet')
    ax.bar(np.arange(len(df_PCK_openpose_rgb)) + width, df_PCK_openpose_rgb, width = width, label = 'openpose_rgb')

    locs, labels = plt.xticks()
    ax.set_xticks(np.arange(len(df_PCK_hpegnn)))
    ax.set_ylabel('PCK@0.4', fontsize=32, labelpad = 5)
    ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0], fontsize=24)
    # plt.ylabel('MPJPE [px]', fontsize=16, labelpad = 5)
    # ax = plt.gca()
    ax.tick_params(axis='x', labelrotation = 20)
    ax.legend(fontsize=32, loc='upper left', mode = "expand", ncol = 4, bbox_to_anchor = (0, 1.01, 1, 0.075))
    ax.set_xticklabels(hpecore_kps_labels.keys(), fontsize = 32)
    ax.set_ylim(0, 1)
    sns.despine(bottom=True)
   
    # plt.suptitle('Global MPJPE for 20per validation set', fontsize = 18, y = 0.92)
    # plt.show()

    # save plot
    if ds_name:
        # plt.suptitle(f'Global PCK@0.4 for {ds_name} sample', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'{ds_name}_pck.png'
    else:
        # plt.suptitle(f'Global PCK@0.4 for whole validation set', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'individual_pck.png'
    plt.savefig(str(fig_path.resolve()))
    # plt.close()
    return None

def joints_cluster_PCK(output_folder_path, ds_name=None):
    if ds_name == None:
        pck_path = output_folder_path/f'pck_0.4.txt'
    else:
        pck_path = output_folder_path/f'pck_0.4_{ds_name}.txt'
    # Load global PCK@04 results
    df_PCK = pd.read_csv(pck_path, sep='\s+')
    # df_PCK = pd.read_csv("/home/cpham-iit.local/data/output_hpe_20per/pck_0.4.txt", sep='\s+')
    df_PCK_movenet_cam_24 = df_PCK["movenet_cam-24"]
    df_PCK_movenet_cam_24 = df_PCK_movenet_cam_24.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # print(df_PCK_movenet_cam_24)
    df_PCK_movenet_cam_24 = df_PCK_movenet_cam_24.iloc[1:14].values
    df_PCK_movenet_cam_24 = [float(x) for x in df_PCK_movenet_cam_24]
    df_PCK_movenet_cam_24_cluster = [np.mean(df_PCK_movenet_cam_24[0:5]), np.mean(df_PCK_movenet_cam_24[5:9]), np.mean(df_PCK_movenet_cam_24[9:13])]

    df_PCK_openpose_rgb = df_PCK["openpose_rgb"]
    df_PCK_openpose_rgb = df_PCK_openpose_rgb.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_PCK_openpose_rgb = df_PCK_openpose_rgb.iloc[1:14].values
    df_PCK_openpose_rgb = [float(x) for x in df_PCK_openpose_rgb]
    df_PCK_openpose_rgb_cluster = [np.mean(df_PCK_openpose_rgb[0:5]), np.mean(df_PCK_openpose_rgb[5:9]), np.mean(df_PCK_openpose_rgb[9:13])]
    # df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK["hpeGnn_splineConv"]
    # # print(df_PCK_ledge10_solo_weight_contrib_stepwise)
    # df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK_ledge10_solo_weight_contrib_stepwise.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # # print(df_PCK_ledge10_solo_weight_contrib_stepwise)
    # df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK_ledge10_solo_weight_contrib_stepwise.iloc[1:14].values
    # # print(df_PCK_ledge10_solo_weight_contrib_stepwise)
    # df_PCK_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_PCK_ledge10_solo_weight_contrib_stepwise]

    df_PCK_hpegnn = df_PCK["hpe-gnn_two_weight_cone_only_target_connectivity_15"]
    df_PCK_hpegnn = df_PCK_hpegnn.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_PCK_hpegnn = df_PCK_hpegnn.iloc[1:14].values
    df_PCK_hpegnn = [float(x) for x in df_PCK_hpegnn]
    df_PCK_hpegnn_cluster = [np.mean(df_PCK_hpegnn[0:5]), np.mean(df_PCK_hpegnn[5:9]), np.mean(df_PCK_hpegnn[9:13])]
    #Plot pck
    width = 0.12
    my_dpi = 96
    fig, ax = plt.subplots(figsize = (2048/my_dpi, 1200/my_dpi), dpi = my_dpi)
    ax.grid(axis = 'y')
    # plt.bar(np.arange(len(df_PCK_ledge10_solo_weight_contrib_stepwise)),df_PCK_ledge10_solo_weight_contrib_stepwise , width = width, label = 'ledge_single_weight') #S_1_1 = hpe_gnn_spline_conv_gamer
    ax.bar(np.arange(len(df_PCK_hpegnn_cluster)) - width,df_PCK_hpegnn_cluster , width=width, label = 'GraphEnet')
    ax.bar(np.arange(len(df_PCK_movenet_cam_24_cluster)), df_PCK_movenet_cam_24_cluster, width = width, label = 'moveEnet')
    ax.bar(np.arange(len(df_PCK_openpose_rgb_cluster)) + width, df_PCK_openpose_rgb_cluster, width = width, label = 'openpose_rgb')

    # locs, labels = plt.xticks()
    ax.set_xticks(np.arange(len(df_PCK_hpegnn_cluster)))
    ax.set_ylabel('PCK@0.4', fontsize=32, labelpad = 5)
    ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0], fontsize=24)
    # plt.ylabel('MPJPE [px]', fontsize=16, labelpad = 5)
    # ax = plt.gca()
    ax.tick_params(axis='x', labelrotation = 0)
    ax.legend(fontsize=32, loc='upper left', mode = "expand", ncol = 4, bbox_to_anchor = (0, 1.01, 1, 0.075))
    ax.set_xticklabels(hpe_kps_labels_cluster.keys(), fontsize = 32)
    ax.set_ylim(0, 1)
    sns.despine(bottom=True)
   
    # plt.suptitle('Global MPJPE for 20per validation set', fontsize = 18, y = 0.92)
    # plt.show()

    # save plot
    if ds_name:
        # plt.suptitle(f'Global PCK@0.4 for {ds_name} sample', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'{ds_name}_pck.png'
    else:
        # plt.suptitle(f'Global PCK@0.4 clustering in body parts', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'individual_pck_cluster.png'
    plt.savefig(str(fig_path.resolve()))
    # plt.close()
    return None

def plot_mpjpe_individual_joints(output_folder_path, ds_name=None):
    if ds_name:
        mpjpe_path = output_folder_path/f'mpjpe_{ds_name}.txt'
    else:
        mpjpe_path = output_folder_path/f'mpjpe.txt'
    #Load MPJPE
    df_MPJPE = pd.read_csv(mpjpe_path, sep='\s+')
    # df_MPJPE = pd.read_csv("/home/cpham-iit.local/data/output_hpe_20per/mpjpe.txt", sep='\s+')
    df_MPJPE_movenet_cam_24 = df_MPJPE["movenet_cam-24"]
    df_MPJPE_movenet_cam_24 = df_MPJPE_movenet_cam_24.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # print(df_PCK_movenet_cam_24)
    df_MPJPE_movenet_cam_24 = df_MPJPE_movenet_cam_24.iloc[1:14].values
    df_MPJPE_movenet_cam_24 = [float(x) for x in df_MPJPE_movenet_cam_24]
    df_MPJPE_openpose_rgb = df_MPJPE["openpose_rgb"]
    df_MPJPE_openpose_rgb = df_MPJPE_openpose_rgb.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_MPJPE_openpose_rgb = df_MPJPE_openpose_rgb.iloc[1:14].values
    df_MPJPE_openpose_rgb = [float(x) for x in df_MPJPE_openpose_rgb]

    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE["hpeGnn_splineConv"]
    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE_ledge10_solo_weight_contrib_stepwise.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE_ledge10_solo_weight_contrib_stepwise.iloc[1:14].values
    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_MPJPE_ledge10_solo_weight_contrib_stepwise]

    df_MPJPE_hpegnn = df_MPJPE["hpe-gnn_two_weight_cone_only_target_connectivity_15"]
    df_MPJPE_hpegnn = df_MPJPE_hpegnn.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_MPJPE_hpegnn = df_MPJPE_hpegnn.iloc[1:14].values
    df_MPJPE_hpegnn = [float(x) for x in df_MPJPE_hpegnn]

    #Plot
    width = 0.1
    my_dpi = 96
    fig, ax = plt.subplots(figsize = (2048/my_dpi, 1200/my_dpi), dpi = my_dpi)
    ax.grid(axis = 'y')
    # plt.bar(np.arange(len(df_MPJPE_ledge10_solo_weight_contrib_stepwise)),df_MPJPE_ledge10_solo_weight_contrib_stepwise , width = width, label = 'ledge_single_weight') #S_1_1 = hpe_gnn_spline_conv_gamer
    ax.bar(np.arange(len(df_MPJPE_hpegnn)) - width,df_MPJPE_hpegnn , width = width, label = 'GraphEnet') 
    ax.bar(np.arange(len(df_MPJPE_movenet_cam_24)), df_MPJPE_movenet_cam_24, width = width, label = 'moveEnet')
    ax.bar(np.arange(len(df_MPJPE_openpose_rgb)) + width, df_MPJPE_openpose_rgb, width = width, label = 'openpose_rgb')

    locs, labels = plt.xticks()
    ax.set_xticks(np.arange(len(df_MPJPE_hpegnn)))
    # plt.ylabel('PCK@0.4 [%]', fontsize=16, labelpad = 5)
    ax.set_ylabel('MPJPE [px]', fontsize=32, labelpad = 5)
    ax.set_yticklabels([0,10,20,30,40,50,60], fontsize=24)
    # ax = plt.gca()
    ax.tick_params(axis='x', labelrotation = 20)
    ax.legend(fontsize = 32, loc='upper left', mode = "expand", ncol = 4, bbox_to_anchor = (0, 1.01, 1, 0.075))
    ax.set_xticklabels(hpecore_kps_labels.keys(), fontsize = 32)
    # ax.set_ylim(0, 100)
    sns.despine(bottom=True)
    # plt.suptitle('Global PCK@0.4 for 20per validation set', fontsize = 18, y = 0.92)
    
    # plt.show()
    # save plot
    if ds_name:
        # plt.suptitle(f'MPJPE for {ds_name} set', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'{ds_name}_mpjpe.png'
    else:
        # plt.suptitle(f'MPJPE for the whole validation set', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'individual_mpjpe.png'
    plt.savefig(str(fig_path.resolve()))
    # plt.close()
    return None

def joints_cluster_MPJPE(output_folder_path, ds_name=None):
    if ds_name:
        mpjpe_path = output_folder_path/f'mpjpe_{ds_name}.txt'
    else:
        mpjpe_path = output_folder_path/f'mpjpe.txt'
    #Load MPJPE
    df_MPJPE = pd.read_csv(mpjpe_path, sep='\s+')
    # df_MPJPE = pd.read_csv("/home/cpham-iit.local/data/output_hpe_20per/mpjpe.txt", sep='\s+')
    df_MPJPE_movenet_cam_24 = df_MPJPE["movenet_cam-24"]
    df_MPJPE_movenet_cam_24 = df_MPJPE_movenet_cam_24.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_MPJPE_movenet_cam_24 = df_MPJPE_movenet_cam_24.iloc[1:14].values
    df_MPJPE_movenet_cam_24 = [float(x) for x in df_MPJPE_movenet_cam_24]
    df_MPJPE_movenet_cam_24_cluster = [np.mean(df_MPJPE_movenet_cam_24[0:5]), np.mean(df_MPJPE_movenet_cam_24[5:9]), np.mean(df_MPJPE_movenet_cam_24[9:13])]

    df_MPJPE_openpose_rgb = df_MPJPE["openpose_rgb"]
    df_MPJPE_openpose_rgb = df_MPJPE_openpose_rgb.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_MPJPE_openpose_rgb = df_MPJPE_openpose_rgb.iloc[1:14].values
    df_MPJPE_openpose_rgb = [float(x) for x in df_MPJPE_openpose_rgb]
    df_MPJPE_openpose_rgb_cluster = [np.mean(df_MPJPE_openpose_rgb[0:5]), np.mean(df_MPJPE_openpose_rgb[5:9]), np.mean(df_MPJPE_openpose_rgb[9:13])]
    # print(df_MPJPE_openpose_rgb_cluster)
    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE["hpeGnn_splineConv"]
    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE_ledge10_solo_weight_contrib_stepwise.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE_ledge10_solo_weight_contrib_stepwise.iloc[1:14].values
    # df_MPJPE_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_MPJPE_ledge10_solo_weight_contrib_stepwise]

    df_MPJPE_hpegnn = df_MPJPE["hpe-gnn_two_weight_cone_only_target_connectivity_15"]
    df_MPJPE_hpegnn = df_MPJPE_hpegnn.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_MPJPE_hpegnn = df_MPJPE_hpegnn.iloc[1:14].values
    df_MPJPE_hpegnn = [float(x) for x in df_MPJPE_hpegnn]
    df_MPJPE_hpegnn_cluster = [np.mean(df_MPJPE_hpegnn[0:5]), np.mean(df_MPJPE_hpegnn[5:9]), np.mean(df_MPJPE_hpegnn[9:13])]

    #Plot
    width = 0.15
    my_dpi = 96
    fig, ax = plt.subplots(figsize = (2048/my_dpi, 1200/my_dpi), dpi = my_dpi)
    ax.grid(axis = 'y')
    # plt.bar(np.arange(len(df_MPJPE_ledge10_solo_weight_contrib_stepwise)),df_MPJPE_ledge10_solo_weight_contrib_stepwise , width = width, label = 'ledge_single_weight') #S_1_1 = hpe_gnn_spline_conv_gamer
    ax.bar(np.arange(len(df_MPJPE_hpegnn_cluster)) - width,df_MPJPE_hpegnn_cluster , width = width, label = 'GraphEnet') 
    ax.bar(np.arange(len(df_MPJPE_movenet_cam_24_cluster)), df_MPJPE_movenet_cam_24_cluster, width = width, label = 'moveEnet')
    ax.bar(np.arange(len(df_MPJPE_openpose_rgb_cluster)) + width, df_MPJPE_openpose_rgb_cluster, width = width, label = 'openpose_rgb')

    locs, labels = plt.xticks()
    ax.set_xticks(np.arange(len(df_MPJPE_hpegnn_cluster)))
    # plt.ylabel('PCK@0.4 [%]', fontsize=16, labelpad = 5)
    ax.set_ylabel('MPJPE [px]', fontsize=32, labelpad = 5)
    ax.set_yticklabels([0,10,20,30,40,50], fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation = 0)
    ax.legend(fontsize = 32, loc='upper left', mode = "expand", ncol = 4, bbox_to_anchor = (0, 1.01, 1, 0.075))
    ax.set_xticklabels(hpe_kps_labels_cluster.keys(), fontsize = 32)
    # ax.set_ylim(0, 100)
    sns.despine(bottom=True)
    # plt.suptitle('Global PCK@0.4 for 20per validation set', fontsize = 18, y = 0.92)
    
    # plt.show()
    # save plot
    if ds_name:
        # plt.suptitle(f'MPJPE for {ds_name} set', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'{ds_name}_mpjpe.png'
    else:
        # plt.suptitle(f'Clustering MPJPE in body parts', fontsize = 32, y = 0.98)
        fig_path = output_folder_path / f'individual_mpjpe_cluster.png'
    plt.savefig(str(fig_path.resolve()))
    # plt.close()
    return None
