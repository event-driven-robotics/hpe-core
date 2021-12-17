# DHP19

The [Dynamic Vision Sensor 3D Human Pose (DHP19) dataset](https://sites.google.com/view/dhp19/home) is the first human 
pose dataset with data collected from DVS event cameras. It has recordings from 4 synchronized 346x260 pixel DVS cameras 
and marker positions in 3D space from Vicon motion capture system. The files have event streams and 3D positions recorded 
from 17 subjects each performing 33 movements.

In order to use this dataset in the context of our pipeline, data must be preprocessed.

## Pre-Processing

Pre-processing the raw data consists in filtering the events and separating them into four distinct camera channels. 
Code is written in *Matlab*. The script `preprocess.m` pre-processes a single recording file and requires the following
variables to be set:

* `subj`, `sess` and `mov` that define the selected recording to pre-process by choosing a subject, a session and a movement.
* `rootCodeFolder` is the folder where the *hpe-core* repo is located
* `rootDataFolder` points to the location where the raw dataset was downloaded
* `outFolder` is the destination folder where the output of the pre-processing will be stored

The script `preprocess_all.m` pre-processes all recording files and requires variables `rootCodeFolder`, `rootDataFolder`
and `outFolder` to be set.

The output of these scripts is a set of files named `S<subj>_<sess>_<mov>.mat` containing a dictionary with the following format

```
{
    'info': {},
    'data':
    {
        'ch0':
        {
            'dvs':
            {
                'x': numpy_array of uint16,
                'y': numpy_array of uint16,
                'ts': numpy_array of float64,
                'pol': numpy_array of bool,
            }
        'ch1': {...},
        'ch2': {...},
        'ch3': {...}
        }
```

where `ch<n>` specifies the camera id, `x` and `y` contain the events' coordinates and `pol` the events' polarity.


## Export to Yarp Format

Once data has been preprocessed, it can be exported to the Yarp format. Code for this step is written in Python and can 
be executed by calling

```shell
$ python3 export_to_yarp.py
      -e <path to .mat DVS file>
      -v <path to .mat Vicon file>
      -p <path to projection matrices folder>  # used for projecting 3D ground truth poses to the camera plane
      -w <number of window events>  # optional, used for computing an event frame
      -o <path to output folder>
```

The output of this script is a set of folders, one for each camera and data type (`dvs` for event data, `skeleton` for 
ground truth poses) containing [Bimvee](https://github.com/event-driven-robotics/bimvee) and [Mustard](https://github.com/event-driven-robotics/mustard) 
compatible `data.log` and `info.log` files, and an additional `play.xml` file for the `yarpmanager` app.


## Export to Numpy Format

It is also possible to export only the ground truth poses (arrays of skeletons) to numpy's file format by calling

```shell
$ python3 export_gt_to_numpy.py
      -e <path to .mat DVS file>
      -v <path to .mat Vicon file>
      -p <path to projection matrices folder>  # used for projecting 3D ground truth poses to the camera plane
      -w <number of window events>  # optional, used for computing an event frame
      -o <path to output folder>
      -td <flag specifying if poses must be projected to the camera space>
```

The output of this script is a set of `.npy` files containing numpy arrays of skeletons.
