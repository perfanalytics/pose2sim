---
title: 'Pose2Sim: An Open Source Python Package for multiview markerless kinematics'
tags:
  - python
  - markerless kinematics
  - motion capture
  - sports performance analysis
  - openpose
  - opensim
authors:
  - name: David Pagnon^[corresponding author] 
    orcid: 0000-0002-6891-8331
    affiliation: 1, 2
  - name: Mathieu Domalain
    orcid: 0000-0002-4518-8479
    affiliation: 2
  - name: Lionel Reveret
    orcid: 0000-0002-0810-5187
    affiliation: 1, 3
affiliations:
 - name: Laboratoire Jean Kuntzmann, Université Grenoble Alpes, 700 avenue Centrale, 38400 Saint Martin d’Hères, France
   index: 1
 - name: Institut Pprime, 2 Bd des Frères Lumière, 86360 Chasseneuil-du-Poitou, France
   index: 2
 - name: Inria Grenoble Rhône-Alpes, 38330 Montbonnot-Saint-Martin, France
   index: 3
date: January 24 2021
bibliography: paper.bib
---

# Summary
`Pose2Sim` provides a workflow for 3D markerless kinematics, as an alternative to the more usual marker-based motion capture methods.\
`Pose2Sim` stands for "OpenPose to OpenSim", as it uses OpenPose inputs (2D coordinates obtained from multiple videos) and leads to an OpenSim result (full-body 3D joint angles). 

The repository presents a framework for: \
• Detecting 2D joint coordinates from videos, e.g. via OpenPose [@Cao_2019], \
• Calibrating cameras, \
• Tracking the person viewed by the most cameras, \
• Triangulating 2D joint coordinates and storing them as 3D positions in a .trc file, \
• Filtering these calculated 3D positions, \
• Scaling and running inverse kinematics via OpenSim [@Delp_2007; @Seth_2018], in order to obtain full-body 3D joint angles.

Each task is easily customizable, and requires only moderate Python skills. Pose2Sim is accessible at [https://github.com/perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim). 

# Statement of need
For the last few decades, marker-based kinematics has been considered the best choice for the analysis of human movement, when regarding the trade-off between ease of use and accuracy. However, a marker-based system is hard to set up outdoors or in context, and it requires placing markers on the body, which can hinder natural movement [Colyer_2018]. 

The emergence of markerless kinematics opens up new possibilities. Indeed, the interest in deep-learning pose estimation neural networks has been growing fast since 2015 [@Zheng_2022], which makes it now possible to collect accurate and reliable kinematic data without the use of physical markers. OpenPose, for example, is a widespread open-source software which provides 2D joint coordinate estimations from videos. These coordinates can then be triangulated in order to produce 3D positions. Yet, when it comes to the biomechanical analysis of human motion, it is often more useful to obtain joint angles than joint center positions in space. Joint angles allow for better comparison among trials and individuals, and they represent the first step for other analysis such as inverse dynamics. 

OpenSim is another widespread open-source software which helps compute 3D joint angles, usually from marker coordinates. It lets scientists define a detailed musculoskeletal model, scale it to individual subjects, and perform inverse kinematics with customizable biomechanical constraints. It provides other features such as net calculation of joint moments or individual muscle forces resolution, although this is out of the scope of our contribution.

The goal of `Pose2Sim` is to build a bridge between the communities of computer vision and biomechanics, by providing a simple and open-source pipeline connecting the two aforementioned state-of-the-art tools: OpenPose and OpenSim. 
`Pose2Sim` has already been used and tested in a number of situations (walking, running, cycling, balancing, swimming, boxing), and published in peer-reviewed scientific publications [@Pagnon_2021; @Pagnon_2022] assessing its robustness and accuracy. The combination of its ease of use, customizable characteristics, and high robustness and accuracy makes it promising, especially for "in-the-wild" sports movement analysis.

So far, little work has been done towards obtaining 3D angles from multiple views [@Zheng_2022]. However, two software applications are worth mentioning. Anipose [@Karashchuk_2021] proposes a Python open-source framework which allows for joint angle estimation with spatio-temporal constraints, but it is primarily designed for animal motion analysis. Theia3D [@Kanko_2021] is a software application for human gait kinematics from videos. Although the GUI is more user friendly, it is not open-source nor easily customizable. Our results on inverse kinematics were deemed good when compared to marker-based ones. See [@Pagnon_2022] for more details on concurrent accuracy with other systems. 

# Features
## Pose2Sim workflow
`Pose2Sim` connects two of the most widely recognized (and open source) pieces of software of their respective fields:\
• OpenPose [@Cao_2019], a 2D human pose estimation neural network\
• OpenSim [@Delp_2007], a 3D biomechanics analysis software

![Pose2Sim full pipeline: (1) OpenPose 2D joint detection; (2i) Camera calibration; (2ii–iv) Tracking the person of interest, Triangulating his coordinates, and Filtering them; (3) Constraining the 3D coordinates to a physically consistent OpenSim skeletal model.\label{fig:pipeline}](Pipeline.png)

The workflow is organized as follows \autoref{fig:pipeline}: \
1. Preliminary OpenPose [@Cao_2019] 2D keypoints detection.\
2. Pose2Sim core, including 4 customizable steps:\
&nbsp;&nbsp;&nbsp;&nbsp;2.i. Camera calibration. \
&nbsp;&nbsp;&nbsp;&nbsp;2.ii. Tracking the person of interest.\
&nbsp;&nbsp;&nbsp;&nbsp;2.iii. 3D keypoints triangulation.\
&nbsp;&nbsp;&nbsp;&nbsp;2.iv. 3D coordinates filtering.\
3. A full-body OpenSim [@Delp_2007] skeletal model with OpenPose keypoints is provided, as well as scaling and inverse kinematics setup files. As the position of triangulated keypoints are not dependent on either the operator nor the subject, these setup files can be taken as is.

OpenPose, OpenSim, and the whole `Pose2Sim` workflow run from any video cameras, on any computer, equipped with any operating system. However, on Linux, OpenSim has to be compiled from source.


## Pose2Sim core
Pose2Sim is meant to be as fully and as easily configurable as possible, by editing the 'User/Config.toml' file. Among others, the following parameters can be adjusted.

### Project
User can change the project path and folder names, the video framerate, and the range of analyzed frames.

### Pose 2D
User can specify the 2D pose estimation model they use.\
The OpenPose BODY_25B experimental model is recommended, as it is as fast as the standard BODY_25 model while being more accurate [@Hidalgo_2019]. Non-OpenPose models can also be chosen, whether they are human such as the AlphaPose one [@Fang_2017], or animal such as any DeepLabCut model trained by the user [@Mathis_2018].

### Calibration
Whether cameras are going to be calibrated with a checkerboard, or simply going to be converted from a calibration file provided by a Qualisys system.\
If checkerboard calibration is chosen, corners are detected and refined with OpenCV. This detection can optionally be displayed for verification. Each camera is then calibrated using OpenCV with an algorithm based on [@Zhang_2000]. The user can choose which image should be used for extrinsic calibration (usually the first or the last one.)

### Tracking
Which body keypoint will be tracked in order to automatically single out the person of interest. We recommend the neck point or one of the hip points. Indeed, in most cases they are the least likely to move out of the camera views. \
This is important when other people are in the background of one or several cameras. This is done by trying out all available triangulations performed for a chosen keypoint in all detected persons. The triangulation with the smallest reprojection error is considered to correspond to the person of interest.

### Triangulation
It should be noted that OpenPose natively provides a module for reconstructing 3D keypoints coordinates [@Hidalgo_2021]. However, it is not developped nor supported anymore, and is acknowledged to be rudimentary. It also needs to be compiled from source, which can constitute an obstacle to non-programmer biomechanicians. On the other hand, triangulation is more robust in Pose2Sim. This is made possible largely because instead of using classic Direct Linear Transform (DLT) [@Hartley_1997], we propose a weighted DLT, i.e., a triangulation procedure where each 2D OpenPose coordinate is weighted with the confidence scores of each camera [@Pagnon_2021]. 
\
&nbsp;&nbsp;&nbsp;&nbsp;i. The minimum in likelihood below which a camera point will not be taken into account for triangulation.\
&nbsp;&nbsp;&nbsp;&nbsp;ii. The maximum in reprojection error above which triangulation results will not be accepted. This can happen if OpenPose provides a bad 2D keypoint estimation, or if the person of interest leaves the camera field. Triangulation will then be done again with one camera less.\
&nbsp;&nbsp;&nbsp;&nbsp;iii. The minimum amount of "good" cameras (remaining after the last two steps) required for triangulating a keypoint. If there is not enough, the 3D keypoint will be interpolated between other frames. The interpolation method can also be chosen.

### Filtering
The filter type and its parameters. Waveforms before and after filtering can be displayed and compared.

### OpenSim
The main contribution of this software is to build a bridge between OpenPose and OpenSim. The latter allows for much more accurate and robust results [@Pagnon_2022], which constrains kinematics to an individually scaled and physically accurate skeletal model. This model also takes into account systematic labelling errors in OpenPose [@Needham_2021]. Since these are considered similar regardless of the subject, neither the model nor the scaling or inverse kinematic files necessarily need to be modified when changing the operator or the participant.\
The OpenSim model, scaling setup file, and inverse kinematics setup files will not be edited or adjusted in the OpenSim GUI, rather than by using the `User\Config.toml` file. This can be done in the same way as one would do with a standard marker-based experiment.

## Pose2Sim utilities
A large part of Pose2Sim functions are also provided as standalone python scripts. Other tools are also provided for extending its usage, such as the ones presented below  \autoref{fig:utilities}.

### 2D pose
• `json_display_with_img.py`:  
Overlays 2D detected .json coordinates on original raw images. High confidence keypoints are green, low confidence ones are red.\
• `json_display_without_img.py`: 
Plots an animation of 2D detected .json coordinates.\
• `DLC_to_OpenPose.py`: 
Converts a DeepLabCut [@Mathis_2018] .h5 2D pose estimation file into OpenPose [@Cao_2019] .json files.

### 3D pose
• `trc_plot.py`: 
Displays X, Y, Z coordinates of a .trc file, each keypoint represented in its own tab.\
• `trc_desample.py`: 
Undersamples a .trc file.\
• `trc_Zup_to_Yup.py`: 
Changes Z-up system coordinates to Y-up system coordinates.\
• `trc_filter.py`: 
Filters trc files. Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median.\
• `trc_gaitevents.py`: 
Detects gait events from point coordinates according to [@Zeni_2008].

![Pose2Sim provides a few additional utilities to extend its capabilities.\label{fig:utilities}](Pose2Sim_workflow_utilities.jpg)

# Acknowledgements
We acknowledge the dedicated people involved in the many major software programs and packages used by Pose2Sim, such as Python, OpenPose, OpenSim, OpenCV [@Bradski_2000], among others. 

# References
