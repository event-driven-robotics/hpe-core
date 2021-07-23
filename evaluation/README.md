# Scripts for offline evaluation

## DHP19 datasaet loading and visualization

The dataset is described [here](https://sites.google.com/view/dhp19/home), and it is availeable for dowload in [this link](https://link.resilio.com/#f=DHP19&sz=0&t=2&s=44DGCWEKSYCCO4BULYWCQ7DWGCR7OK4AG7IHD6R3JMGUERFTJD2Q&i=C6DGXPZQKSQ7TDMVCK5X346X6LIQTI7E7&v=2.6&a=2). And the code needed for visualization and evaluation is located in the folder named `convert_DHP19` of this repository. 

The first step consist of a pre-processing of the raw data using *Matlab* that performs a filtering of the event stremas as well as the separation of the four different camera channels. In the file `preprocess.m` there are a couple of variables that must be set in order to obtain the desired outputs. These are:

* `subj`, `sess` and `mov` that define the selected recording to preprocess by choosing a subject, a session and a movment.
* `rootCodeFolder` is the folder where the *hpe-core* repo is located
* `rootDataFolder` points to the locations where the raw dataset was downloaded
* `outFolder` is the destination folder where the output of the preprocessing will be stored




---

Python scripts to:

* load a dataset + results file + ground-truth
* visualise the output
* plot trajectories over time
* produce quantitative results

Common file format required of all results:

* TODO

Event loading should be using BIMVEE (extended for our data where required), visualisation should be integrated with MUSTARD (extened for our data where required).
