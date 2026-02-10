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

When implementing, also provide with a python-based simple web server to save the file uploaded.

## Visual Data 

The image sensors and the corresponding channels that should support:

1. Pseudo camera: A fake camera that generate images with numbers increasing starting from 1 (as reference of frame number)

2. USB web camera (supported via v4l2, configurable transimission via YUYV or MJPG, also resolutions and fps via v4l2 ctrl)

3. Intel Realsense (supported via dependencies on pyrealsense, configurable channels and resolutions and fps)

4. Support possibility for other sensors (e.g., ZED X Mini, no need to implement now)

The visual data should be encoded to save space. These codec should be supported:

1. CPU-based codec, like libh264

2. Hardware-accelerated codec, like use of nvenc and nvdec on NVIDIA RTX 4090 and Jetson AGX Orin

## Proprioception Data

The recording of robot sensors should support:

1. Pseudo robot: A fake 6-dof robot that generate random walk action for emulating  

2. AIRBOT (joint positions, velocities, efforts). See usage in external/kdl_demos

3. AgileX NERO. See https://github.com/agilexrobotics/pyAgxArm for the sources for SDK. Specifically https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/nero-API%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3.MD

4. AIRBOT in DISCOVERSE simulator (to be updated)

Kinematic support should also be added. Give the descriptions (urdf files), it is possible to calculate the 6D poses for the end effector.

The recording of 6D pose should be implemented in 1. external sensor (like HTC vive, but don't need to implement now) 2. calculation via forward kinematics. See usage of kdl libraries in external/kdl_demos

## External Control Input

The program should accept input sources like (keyboard or mouse or other source of input) to control the data collecting process.

## Other information

Record timestamp, camera intrinsics (also give a tool to calibrate that), camera and robot extrinsics (also give a tool to calibrate that), compressed urdf file along with meshes

# Workflow

## 1. Setup Wizard (Bootstraping)

The necessity of setup wizard is to scan the existance of supported hardware (cameras, robots), and prompt in TUI to assign the channel names for these hardware. Note that this part should also run properly even with no desktop environment (to identify robots, slowly oscillate the last joint of the arm in a small range; to identify cameras, render the image in real-time with ascii). After setup wizard, a configuration file should be generated to identify the combination of all meta information for the subsequent data collecting process. The other supported parameters, like the resolutions, fps, codec, should also be able to customize with default values. The default values can be stored in a separate template file.

The setup wizard should also be able to modify the configuration based on a given or default config, and save to a new config file. This should happen when the task or other meta information changes.

## 2. Formal Data Collection

Currently two forms are supported for the data collection process: tele-operation and human-intervention. The mode to use should be configured in setup wizard

During the formal data collection, the TUI should contain these elements:

1. The live preview of cameras (rgb and depths, see image rendering demo in tui_webcam.py)

2. The live moving average of proprioception data (so that human operator would know the sensor is alive)

3. An information panel: like number of episodes that have been collected, and other information

The live preview of cameras are also broadcasted via rtsp.

The beginning and finishing of one episodes should be signaled by external control input. After finishing each episodes, one should be able to keep or discard the episodes that are just collected. The binding of "start", "stop", "keep", "discard" to the detailed action on external control input devices should be configurable in setup wizards with default values. 

### Tele-Operation Mode:

In tele-operation mode, the leading arms are in gravity compensation mode, in which the end effector of the leading arm can be manually dragged freely. And the following arms closely follow the leading arms to finish the task.

In this mode, it is possible to have 1 arm following 1 arm and 2 arms following 2 arms (left and right). The proprioceptive data from all 2 and 4 arms should be recorded accordingly. Which one is leading and which one is following should be specified in the configuring step of the setup wizard

### Human Intervention Mode:

In human intervention mode, the following arms actually accept external commands (maybe via websocket or mqtt or zmq or other publishing middlewares), and the leading arms are closely following the leading arms by default. The control implementation of human intervention is reserved for later, currently implement it as just closely following. In this mode, like the tele-operation mode, the proprioceptive data from all 2 and 4 arms should be recorded accordingly. Which one is leading and which one is following should be specified in the configuring step of the setup wizard

## 3. Episode Replay:

This workflow intends to replay a selected episode to validate. In this replay mode, the TUI layout should be the same as formal data collection, and when a "start" is triggered by external control input, the episode starts replaying (image live previews start, robot moving starts, etc.)
