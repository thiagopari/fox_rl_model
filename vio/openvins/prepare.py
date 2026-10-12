#!/usr/bin/env python3
"""EuRoC-layout dataset -> ROS 2 bag (rosbags, sqlite3, /cam0/image_raw mono8 + /imu0) and OpenVINS Kalibr yamls.

    ~/.venvs/fox_vio/bin/python vio/openvins/prepare.py ~/data/fox_vio/sim_v1

writes DATASET/ros2bag/ and DATASET/openvins/{estimator_config.yaml, kalibr_imu_chain.yaml, kalibr_imucam_chain.yaml}.
Bag stamps = dataset stamps + T0 (some ROS tools treat stamp 0 as unset); subtract T0 from the estimate.
"""
import os
import shutil
import sys

import numpy as np
import yaml
from PIL import Image
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

T0 = 100 * 10**9


def mat(t):
    return "\n".join(f"    - [{', '.join(f'{v:.9f}' for v in row)}]" for row in t)


def main(ds):
    mav = os.path.join(ds, "mav0")
    cam = yaml.safe_load(open(os.path.join(ds, "camchain-imucam.yaml")))["cam0"]
    imu = yaml.safe_load(open(os.path.join(ds, "imu.yaml")))
    out = os.path.join(ds, "openvins")
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "estimator_config.yaml"), out)
    eye = mat(np.eye(4))
    with open(os.path.join(out, "kalibr_imu_chain.yaml"), "w") as f:
        f.write(f"%YAML:1.0\n\nimu0:\n  T_i_b:\n{eye}\n"
                + "".join(f"  {k}: {imu[k]:.6e}\n" for k in ("accelerometer_noise_density", "accelerometer_random_walk",
                                                             "gyroscope_noise_density", "gyroscope_random_walk"))
                + f"  rostopic: /imu0\n  time_offset: 0.0\n  update_rate: {imu['update_rate']}\n  model: \"kalibr\"\n"
                + "".join(f"  {k}:\n{mat(np.eye(3) if k != 'Tg' else np.zeros((3, 3)))}\n"
                          for k in ("Tw", "R_IMUtoGYRO", "Ta", "R_IMUtoACC", "Tg")))
    with open(os.path.join(out, "kalibr_imucam_chain.yaml"), "w") as f:
        f.write(f"%YAML:1.0\n\ncam0:\n  T_imu_cam:\n{mat(np.linalg.inv(np.array(cam['T_cam_imu'])))}\n"
                f"  cam_overlaps: []\n  camera_model: pinhole\n  distortion_coeffs: {cam['distortion_coeffs']}\n"
                f"  distortion_model: radtan\n  intrinsics: {cam['intrinsics']}\n  resolution: {cam['resolution']}\n"
                f"  rostopic: /cam0/image_raw\n")

    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    stamp = lambda ns: T["builtin_interfaces/msg/Time"](sec=int(ns // 10**9), nanosec=int(ns % 10**9))
    header = lambda ns, frame: T["std_msgs/msg/Header"](stamp=stamp(ns), frame_id=frame)
    vec = T["geometry_msgs/msg/Vector3"]
    bag = os.path.join(ds, "ros2bag")
    if os.path.exists(bag):
        shutil.rmtree(bag)
    msgs = []                                                               # (ns, topic, msg), written in time order
    for row in np.loadtxt(os.path.join(mav, "imu0", "data.csv"), delimiter=",", skiprows=1):
        ns = int(row[0]) + T0
        msgs.append((ns, "/imu0", T["sensor_msgs/msg/Imu"](
            header=header(ns, "imu0"), orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=np.r_[-1.0, np.zeros(8)], angular_velocity=vec(x=row[1], y=row[2], z=row[3]),
            angular_velocity_covariance=np.zeros(9), linear_acceleration=vec(x=row[4], y=row[5], z=row[6]),
            linear_acceleration_covariance=np.zeros(9))))
    for stamp_s, name in np.loadtxt(os.path.join(mav, "cam0", "data.csv"), delimiter=",", dtype=str, skiprows=1):
        msgs.append((int(stamp_s) + T0, "/cam0/image_raw", name))
    msgs.sort(key=lambda m: (m[0], m[1]))
    with Writer(bag, version=8) as w:                                       # v8: readable by Humble's rosbag2
        conns = {"/imu0": w.add_connection("/imu0", "sensor_msgs/msg/Imu", typestore=ts),
                 "/cam0/image_raw": w.add_connection("/cam0/image_raw", "sensor_msgs/msg/Image", typestore=ts)}
        for ns, topic, m in msgs:
            if topic == "/cam0/image_raw":
                img = np.asarray(Image.open(os.path.join(mav, "cam0", "data", m)).convert("L"))
                m = T["sensor_msgs/msg/Image"](header=header(ns, "cam0"), height=img.shape[0], width=img.shape[1],
                                               encoding="mono8", is_bigendian=0, step=img.shape[1], data=img.ravel())
            w.write(conns[topic], ns, ts.serialize_cdr(m, m.__msgtype__))
    print(f"{len(msgs)} messages -> {bag}; OpenVINS config -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
