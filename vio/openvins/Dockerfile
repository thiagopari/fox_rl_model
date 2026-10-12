# OpenVINS (ROS 2 Humble) for the synthetic Fox dataset.
#   docker build -t fox-openvins vio/openvins
FROM ros:humble-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends git libeigen3-dev libboost-all-dev libceres-dev \
      libopencv-dev ros-humble-cv-bridge ros-humble-image-transport ros-humble-tf2-geometry-msgs \
      ros-humble-rosbag2-storage-default-plugins python3-colcon-common-extensions && rm -rf /var/lib/apt/lists/*
ARG OV_REF=master
RUN git clone https://github.com/rpng/open_vins /ws/src/open_vins && git -C /ws/src/open_vins checkout ${OV_REF} \
    && git -C /ws/src/open_vins log -1 --format='open_vins %H %cs' > /ws/OPEN_VINS_VERSION
WORKDIR /ws
RUN . /opt/ros/humble/setup.sh && MAKEFLAGS=-j12 colcon build --event-handlers console_cohesion+ \
      --packages-up-to ov_msckf --cmake-args -DCMAKE_BUILD_TYPE=Release && rm -rf build log
