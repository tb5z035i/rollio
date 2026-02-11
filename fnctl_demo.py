import fcntl
import os
import struct

# v4l2 constants (from linux/videodev2.h)
VIDIOC_S_FMT = 0xC0D05605
VIDIOC_S_PARM = 0xC0CC5616
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_PIX_FMT_MJPEG = 0x47504a4d  # 'MJPG'

fd = os.open("/dev/video0", os.O_RDWR)

# struct v4l2_format
fmt = struct.pack(
    "I4xI4x32s",
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    1280,  # width
    720,   # height
    struct.pack("I", V4L2_PIX_FMT_MJPEG),
)

fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)

# struct v4l2_streamparm
parm = struct.pack(
    "I4xIIII",
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    1,    # numerator
    30,   # denominator → 30 fps
    0, 0
)

fcntl.ioctl(fd, VIDIOC_S_PARM, parm)

os.close(fd)