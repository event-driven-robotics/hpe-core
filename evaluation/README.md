# Scripts for offline evaluation

## DHP19 dataset loading and visualization

The dataset is described [here](https://sites.google.com/view/dhp19/home), and it is available for download in [this link](https://link.resilio.com/#f=DHP19&sz=0&t=2&s=44DGCWEKSYCCO4BULYWCQ7DWGCR7OK4AG7IHD6R3JMGUERFTJD2Q&i=C6DGXPZQKSQ7TDMVCK5X346X6LIQTI7E7&v=2.6&a=2). And the code needed for visualization and evaluation is located in the folder named `convert_DHP19` of this repository. In order to use this dataset in the context of our pipeline, two steps are required. 

### Pre-processing

The first step consist of a pre-processing of the raw data using *Matlab* that performs a filtering of the event streams as well as the separation of the four different camera channels. In the file `preprocess.m` there are a couple of variables that must be set in order to obtain the desired outputs. These are:

* `subj`, `sess` and `mov` that define the selected recording to pre-process by choosing a subject, a session and a movement.
* `rootCodeFolder` is the folder where the *hpe-core* repo is located
* `rootDataFolder` points to the location where the raw dataset was downloaded
* `outFolder` is the destination folder where the output of the pre-processing will be stored

There is also the alternative to do the pre-processing for all the recordings in DHP19 using the file `preprocess_all.m` which requires the paths mentioned for the previous case.
### Data manipulation and visualization

The second stage consist of formatting the obtained output from the previous step such that is compatible with *YARP* and *Mustard*. This is done using *Python* code in the file `load_dhp19.py`. This file is divided into cells which are delimited by `# %% `, and are intended to be run as blocks. Here is a description of each block:

* *Preliminaries*: sets the `PYTHONPATH` so that *Bimvee* and *Mustard* are in it by loading environment  variables that must be first set in `.bashrc`. There is also the declaration of path variables pointing to the location of the dataset, as well as the choosing of the recording to be used (variables `subj`, `sess` and `mov`). The folder containing the dataset must have the following structure:
```
.../hpe-data
│
└───DVS
│   S1_1_1.mat
│   S1_1_2.mat
│   ...
│
└───P_Matrices
│   camera_positions.npy
│   P1.npy
│   P2.npy
│   P3.npy
│   P4.npy
│
└───Vicon
    S1_1_1.mat
    S1_1_2.mat
    ...
```
* `Load DVS data`: loads the information for the four different DVS cameras and structures it a way *bimvee* can process it and *Mustard* can play it, i.e. it builds up a container with four `dvs` channels.
*  `Load Vicon data`: loads ground-truth Vicon poses and builds the timestamp array that matches those poses and that is synced with the event streams.
* `Vicon 3D->2D`: converts 3D Vicon poses into 2D poses for each one of the four event camera channels using their respective camera positions and transformation matrices.
* `Build DVS+GT container`: creates a container that allows *Mustard* to visualize events and ground truth poses at the same time.
* `Plot joints vs t`: provides `x` and `y` ground-truth coordinates for each joints over time. This will then be extended to compare against tracking results.
* `Plot events (Roi) + GT`: provides `x` and `y` ground-truth coordinates for a single joint over time as well as events falling inside a given Region of Interest (that can be choose by setting the variable `roi`). 
* `Start mustard`: starts the visualizer in thread.
* `Visualize data`: sets the data to be visualized.
* `Export to yarp`: exports the four `dvs` channels in *Yarp* format that can be used in its environment, and ultimately will be used as input for the tracking algorithm.

There is also the file `exportYARP.py' which simply allows to take the output of the pre-processed files in the previous stage and export them in `yarpiit` format.
---

Python scripts to:

* load a dataset + results file + ground-truth
* visualise the output
* plot trajectories over time
* produce quantitative results

Common file format required of all results:

* TODO

Event loading should be using BIMVEE (extended for our data where required), visualization should be integrated with MUSTARD (extended for our data where required).
