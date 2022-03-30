import glob
import os
import sys
import cdflib
from tqdm import tqdm

dev = False

# video_path = '/home/ggoyal/code/h36m-fetch/archives/extracted/S1/Videos/Directions.55011271.mp4'
slomo_model_path = '/home/icub/data/v2e_data/slomo_model/SuperSloMo39.ckpt'
ts_res = str(.001)
auto_ts_res = str(False)
dvs_exposure_duration = str(0.005)
# base_output_folder = '/home/icub/data/h36m/events_fullHD/'
base_output_folder = '/home/icub/data/h36m/'
pos_thres = str(.15)
neg_thres = str(.15)
sigma_thres = str(0.03)
dvs_h5 = 'Directions.55011271'
cropping_log_file = base_output_folder+'cropping_data.txt'
output_width = 640
output_height = 480
buffer = 20
cutoff_hz = '15'
stop_time = 'None'
batch_size = '2'
dataset_path = '/home/icub/data/h36m/extracted/'
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
all_cameras = {1: '54138969', 2: '55011271', 3: '58860488', 4: '60457274'}
cams = [2, 4]  # And maybe later camera 4.

# Reading the files in the folders
# all_data_paths = [os.path.join(dataset_path, sub, 'Videos') for sub in subs]
# print(all_data_paths)
# print(all_cameras)

def get_cropping(anno_file):
    cdf_file = cdflib.CDF(anno_file)
    data = (cdf_file.varget("Pose")).squeeze()
    data2 = data.reshape([-1, 2], order='C')
    left = min(data2[:,0])
    right = max(data2[:,0])
    top = min(data2[:,1])
    bottom = max(data2[:,1])
    width = right-left +buffer
    height = bottom-top +buffer
    center = [int(left + width/2), int(top + height/2)]
    if width <output_width and height< output_height:
        crop_left = center[0] - int(output_width/2)
        crop_right = 1000 - int(center[0] + output_width/2)
        crop_top = center[1] - int(output_height/2)
        crop_bottom = 1000 - int(center[1] +output_height/2)
        crop = f'{crop_left},{crop_right},{crop_top},{crop_bottom}'
        assert (crop_left+crop_right == 360)
        assert (crop_top+crop_bottom == 520)
    else:
        if 0.75*width>height:
            height = 0.75*width
        elif 0.75 * width <= height:
            width = height / 0.75
        if width>1000:
            width=1000
            center[0] = 500
        if height>1000:
            height = 1000
            center[1] = 500


        crop_left = center[0] - int(width / 2)
        crop_right = 1000 - int(center[0] + width/2)
        crop_top = center[1] - int(height / 2)
        crop_bottom = 1000 - int(center[1] + height/2)

        if crop_left<0:
            crop_right = crop_right + crop_left
            crop_left = 0
        if crop_right < 0:
            crop_left = crop_right + crop_left
            crop_right = 0
        if crop_top < 0:
            crop_bottom = crop_bottom + crop_top
            crop_top = 0
        if crop_bottom < 0:
            crop_top = crop_bottom + crop_top
            crop_bottom = 0

        crop = f'{crop_left},{crop_right},{crop_top},{crop_bottom}'

    return crop

if dev:
    base_output_folder = os.path.join(base_output_folder,'tester')
else:
    base_output_folder = os.path.join(base_output_folder,'events_fullHD')

files = list([])
anno_files = list([])
output_folders = list([])
for sub in subs:
    for cam in cams:
        path = os.path.join(dataset_path, sub, 'Videos')
        temp_name = path + '/*' + all_cameras[cam] + '.mp4'
        files_temp = glob.glob(temp_name)
        for file in files_temp:
            if 'ALL' not in file:
                files.append(file.replace(' ', '\ '))
                anno_file = os.path.join(dataset_path, sub, "Poses_D2_Positions", os.path.basename(file).replace('mp4', 'cdf'))
                anno_files.append(anno_file)
                out_name = 'cam'+ str(cam)+ '_'+ sub + '_'+ os.path.basename(file).split('.')[0].replace(' ', '_')
                output_folders.append(out_name)

# Check that everything went well
assert len(files) == len(output_folders)

# Call the v2e command for each file.
n = 0
# for i in range(20):
# for i in range(1):
for i in tqdm(range(len(files))):
    crop = get_cropping(anno_files[i])
    # crop = '(0,0,0,0)' # (left, right, top, bottom)
    # print(output_folders[i], '>>>>' ,files[i],'>>>>',anno_files[i])

    print("\nFile {} of {} being processed.\n".format(i + 1, len(files)))
    output_folder = os.path.join(base_output_folder, output_folders[i])
    if os.path.exists(output_folder):
        print("Folder already exists. File {} being skipped.".format(files[i]))
        continue
    command = "python v2e.py -i " + files[i] + " --overwrite --slomo_model " + slomo_model_path + \
              " --timestamp_resolution=" + ts_res + " --auto_timestamp_resolution=" + auto_ts_res + \
              " --dvs_exposure duration " + dvs_exposure_duration + " --output_folder=" + output_folder + " --pos_thres=" + pos_thres + \
              " --neg_thres=" + neg_thres + " --sigma_thres=" + sigma_thres + " --dvs_h5 " + dvs_h5 + " --output_width=" + str(output_width) + \
              " --output_height=" + str(output_height) + " --cutoff_hz=" + cutoff_hz + " --dvs_aedat2 None --dvs640 " + \
              "--crop " + crop
    if not dev:
        command = command + " --batch_size " + batch_size + " --skip_video_output --no_preview stop_time=2"
    # print(command)
    os.system(command)
    print("File {} processed.".format(files[i]))
    with open(cropping_log_file, 'a') as f:
        f.write("%s %s" % (output_folders[i], crop))
        f.write("\n")
    if dev:
        exit()

print("Process completed.")
