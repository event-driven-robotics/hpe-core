import glob
import os
import sys

dev = True

video_path = '/home/ggoyal/code/h36m-fetch/archives/extracted/S1/Videos/Directions.55011271.mp4'
slomo_model_path = '/home/icub/data/v2e_data/slomo_model/SuperSloMo39.ckpt'
ts_res = str(.001)
auto_ts_res = str(False)
dvs_exposure_duration = str(0.005)
base_output_folder = '/home/icub/data/h36m/Slomo_sample/'
pos_thres=str(.15)
neg_thres=str(.15)
sigma_thres=str(0.03)
dvs_h5 = 'Directions.55011271'
output_width='346'
output_height='260'
cutoff_hz='15'
stop_time = 'None'
batch_size = '24'
dataset_path = '/home/icub/data/h36m/extracted/'
subs = ['S1','S5','S6','S7','S8','S9','S11']
all_cameras = {1:'54138969',2:'55011271',3:'58860488',4:'60457274'}
cam = 2 # And maybe later camera 4.


# Reading the files in the folders
all_data_paths = [os.path.join(dataset_path,sub,'Videos') for sub in subs]
print(all_data_paths)
print(all_cameras)

files = list([])
output_folders = list([])
for sub in subs:
    path = os.path.join(dataset_path,sub,'Videos')
    temp_name = path + '/*'+all_cameras[cam]+'.mp4'
    files_temp = glob.glob(temp_name)
    [files.append(file.replace(' ','\ ')) for file in files_temp]
    [output_folders.append(sub+'_'+os.path.basename(file).split('.')[0].replace(' ','_')) for file in files_temp]

# Check that everything went well
assert len(files) == len(output_folders)

# Call the v2e command for each file.
for i in range(len(files)):
# for i in range(len(files)):
    # print(output_folders[i]+ '>>>>' +files[i])
    print("\nFile {} of {} being processed.\n".format(i+1,len(files)))
    output_folder = os.path.join(base_output_folder, output_folders[i])
    if os.path.exists(output_folder):
        print("Folder already exists. File {} being skipped.".format(files[i]))
        continue
    command = "python v2e.py -i "+files[i]+" --overwrite --slomo_model "+slomo_model_path+ \
" --timestamp_resolution="+ts_res+" --auto_timestamp_resolution="+auto_ts_res+ \
" --dvs_exposure duration "+dvs_exposure_duration+" --output_folder="+output_folder+" --pos_thres="+pos_thres+ \
" --neg_thres="+neg_thres+" --sigma_thres="+sigma_thres+" --dvs_h5 "+dvs_h5+" --output_width="+output_width+ \
" --output_height="+output_height+" --cutoff_hz="+cutoff_hz+" --stop_time=5"
    if not dev:
        command = command + " --batch_size "+batch_size+" --skip_video_output --no_preview" 
    os.system(command)
    print("File {} processed.".format(files[i]))

print("Process completed.")
