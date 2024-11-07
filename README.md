# Human Pose Estimation Core Library
_A library of functions for human pose estimation with event-driven cameras_

![demo](https://user-images.githubusercontent.com/9265237/216939617-703fc4ef-b4b9-4cbc-aab8-a87c04822be2.gif)

Please cite:
```
@InProceedings{Goyal_2023_CVPR,
    author    = {Goyal, Gaurvi and Di Pietro, Franco and Carissimi, Nicolo and Glover, Arren and Bartolozzi, Chiara},
    title     = {MoveEnet: Online High-Frequency Human Pose Estimation With an Event Camera},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {4023-4032}
}
```
also for the eH3.6m dataset:
```
https://zenodo.org/record/7842598
```
and checkout [other publications](#see-also)

Please contribute your event-driven HPE application and datasets to enable comparisons!

### [Core (C++)](https://github.com/event-driven-robotics/hpe-core/tree/main/core)

Compile and link the core C++ library in your application to use the event-based human pose estimation functions including:
* joint detectors: OpenPose built upon greyscales formed from events
* joint velocity estimation @>500Hz
* asynchronous pose fusion of joint velocity and detection
* event representation methods to be compatible with convolutional neural networks.

### [PyCore](https://github.com/event-driven-robotics/hpe-core/tree/main/pycore)

Importable python libraries for joint detection
* event-based movenet: MoveEnet built on PyTorch

### [Examples](https://github.com/event-driven-robotics/hpe-core/tree/main/example)

Some example applications are available giving ideas on how to use the HPE-core libraries
- MoveEnet inference code in corresponding example [here](https://github.com/event-driven-robotics/hpe-core/tree/main/example/movenet)

### [Evaluation](https://github.com/event-driven-robotics/hpe-core/tree/main/evaluation)

Python scripts can be used to compare different detectors and velocity estimation combinations

### [Datasets and Conversion](https://github.com/event-driven-robotics/hpe-core/tree/main/datasets)

Scripts to convert datasets into common formats to easily facilitate valid comparisons, and consistency

### Authors

> [@arrenglover](https://www.linkedin.com/in/arren-glover/)
> [@nicolocarissimi](https://www.linkedin.com/in/nicolocarissimi/)
> [@gaurvigoyal](https://www.linkedin.com/in/gaurvigoyal/)
> [@francodipietro](https://www.linkedin.com/in/francodipietrophd/)

[Event-driven Perception for Robotics](https://www.edpr.iit.it/research)

### See Also

```
@INPROCEEDINGS{9845526,
  author={Carissimi, Nicolò and Goyal, Gaurvi and Pietro, Franco Di and Bartolozzi, Chiara and Glover, Arren},
  booktitle={2022 8th International Conference on Event-Based Control, Communication, and Signal Processing (EBCCSP)}, 
  title={Unlocking Static Images for Training Event-driven Neural Networks}, 
  year={2022},
  pages={1-4},
  doi={10.1109/EBCCSP56922.2022.9845526}}
```
