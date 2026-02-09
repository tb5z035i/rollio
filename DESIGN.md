This is an overall description of the data collection suite to implement:

A python package to collect timestamp-aligned robot episodes for the training of Vision-Language-Action models, and also for offline RL in embodied learning.

# Episodes

The basic unit for data is an episode.

The robot episodes should include:

1. RGB & Depth & other image based sensory data accompanied with timestamps

2. Joint states (positions, optional velocities, optional torques)

3. Meta information (signal frequencies, creation timestamp, description of tasks, frame resolution, codec, and pixel format for each channel, encoding quality (if any))

The episodes should be able to be supported in two formats: Lerobot 2.1 and Lerobot 3.0. The description of the data format is in LEROBOT_FORMAT.md

Also, the storage backend should support:

1. Save to local storage. The root for the storage is configurable with defaults

2. Upload to an HTTP endpoint. The upload process should happen at backstage non-blocking.

## Visual Data 

The image sensors and the corresponding channels that should support:

1. USB web camera (supported via v4l2, configurable transimission via YUYV or MJPG, also resolutions and fps via v4l2 ctrl)

2. Intel Realsense (supported via dependencies on pyrealsense, configurable channels and resolutions and fps)

3. Support possibility for other sensors (e.g., ZED X Mini, no need to implement now)

The visual data should be encoded to save space. These codec should be supported:

1. CPU-based codec, like libh264

2. Hardware-accelerated codec, like use of nvenc and nvdec on NVIDIA RTX 4090 and Jetson AGX Orin

## Proprioception Data

The recording of robot sensors should support:

1. AIRBOT (joint positions, velocities, efforts). See usage in external/kdl_demos

2. AgileX NERO. See https://github.com/agilexrobotics/pyAgxArm for the sources for SDK. Specifically https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/nero-API%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3.MD

Kinematic support should also be added. Give the descriptions (urdf files), it is possible to calculate the 6D poses for the end effector.

The recording of 6D pose should be implemented in 1. external sensor (like HTC vive, but don't need to implement now) 2. calculation via forward kinematics. See usage of kdl libraries in external/kdl_demos

## Control Input



## Workflow

### 1. Setup Wizard (Bootstraping)

The necessity of setup wizard is to scan the existance of supported hardware (cameras, robots), and prompt in TUI to assign the channel names for these hardware. Note that this part should also run properly even with no desktop environment (to identify robots, slowly oscillate the last joint of the arm in a small range; to identify cameras, render the image in real-time with ascii). After setup wizard, a configuration file should be generated to identify the combination of all meta information for the subsequent data collecting process. The other supported parameters, like the resolutions, fps, codec, should also be able to customize with default values. The default values can be stored in a separate template file.

The setup wizard should also be able to modify the configuration based on a given or default config, and save to a new config file. This should happen when the task or other meta information changes.

### 2. Formal Data Collection