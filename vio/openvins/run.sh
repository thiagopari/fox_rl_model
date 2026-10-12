#!/usr/bin/env bash
# OpenVINS (mono + IMU) on DATASET/ros2bag in the fox-openvins image (vio/openvins/Dockerfile), real-time playback.
#   vio/openvins/prepare.py DATASET && vio/openvins/run.sh DATASET
# -> DATASET/openvins/{ov.log, ov_estimate.txt, ov_estimate.tum (IMU poses, dataset time)}
set -euo pipefail
DS=$(realpath "$1")
rm -f "$DS"/openvins/ov_estimate*.txt
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_LOCALHOST_ONLY=1 -v "$DS":/data fox-openvins bash -c '
  source /ws/install/setup.bash
  # ros2 run, not the launch file: the ROS 2 node reads filepath_est only from ROS params (empty dir otherwise -> crash)
  ros2 run ov_msckf run_subscribe_msckf --ros-args -r __ns:=/ov_msckf -p config_path:=/data/openvins/estimator_config.yaml \
    -p use_stereo:=false -p max_cameras:=1 -p save_total_state:=true -p filepath_est:=/data/openvins/ov_estimate.txt \
    -p filepath_std:=/data/openvins/ov_estimate_std.txt -p filepath_gt:=/data/openvins/ov_groundtruth.txt \
    > /data/openvins/ov.log 2>&1 &
  sleep 5
  ros2 bag play /data/ros2bag > /data/openvins/bag_play.log 2>&1
  sleep 2
  pkill -TERM -f run_subscribe_msckf; wait   # the node ignores SIGINT'
# save_total_state rows: t q_GtoI(JPL x y z w = Hamilton q of R_ItoG) p_IinG ...; bag stamps carry +100 s (prepare.py T0)
awk '!/^#/ && NF > 7 {printf "%.9f %s %s %s %s %s %s %s\n", $1 - 100, $6, $7, $8, $2, $3, $4, $5}' \
  "$DS/openvins/ov_estimate.txt" > "$DS/openvins/ov_estimate.tum"
wc -l "$DS/openvins/ov_estimate.tum"
