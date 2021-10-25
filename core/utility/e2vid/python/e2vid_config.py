sensor_width = 346  # width of the sensor
sensor_height = 260  # height of the sensor
fixed_duration = False
window_size = 7500  # Size of each event window, in number of events. Ignored if fixed_duration = True
window_duration = 33.33  # Duration of each event window, in milliseconds. Ignored if fixed_duration = False
num_events_per_pixel = 0.33  # in case N (window size) is not specified, it will be automatically computed as N = width * height * num_events_per_pixel
skipevents = 0
suboffset = 0
compute_voxel_grid_on_cpu = False  # compute_voxel_grid_on_cpu

output_folder = None  # if None, will not write the images to disk
dataset_name = 'reconstruction'

use_gpu = True


###########
# Display #
###########

display = False
show_events = False

# Event display mode ('red-blue' or 'grayscale')
event_display_mode = 'red-blue'

# Number of bins of the voxel grid to show when displaying events (-1 means show all the bins)
num_bins_to_show = -1

# Remove the outer border of size display_border_crop before displaying image
display_border_crop = 0

# Time to wait after each call to cv2.imshow, in milliseconds (default: 1)
display_wait_time = 1


###############################
# Post-processing / filtering #
###############################

hot_pixels_file = None  # (optional) path to a text file containing the locations of hot pixels to ignore

# (optional) unsharp mask
unsharp_mask_amount = 0.3
unsharp_mask_sigma = 1.0

# (optional) bilateral filter
bilateral_filter_sigma = 0.0

# (optional) flip the event tensors vertically
flip = False


##########################################################
# Tone mapping (i.e. rescaling of the image intensities) #
##########################################################

# Min intensity for intensity rescaling (linear tone mapping)
Imin = 0.0

# Max intensity value for intensity rescaling (linear tone mapping)
Imax = 1.0

# If True, will compute Imin and Imax automatically
auto_hdr = False

# Size of the median filter window used to smooth temporally Imin and Imax
auto_hdr_median_filter_size = 10

# Perform color reconstruction? (only use this flag with the DAVIS346color
color = False


#######################
# Advanced parameters #
#######################

# disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results
no_normalize = False

# disable recurrent connection (will severely degrade the results; for testing purposes only
no_recurrent = False
