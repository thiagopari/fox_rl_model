#!/usr/bin/env python3
"""BASELINE monocular visual odometry (NOT VIO) on a EuRoC-layout dataset, scored against ground truth with evo.

KLT tracks (forward-backward checked) + essential matrix (5-point RANSAC) between keyframes every STRIDE frames.
Monocular E gives rotation and the translation DIRECTION only, so every moving keyframe step has unit length (no scale
propagation) and evo's Sim(3) alignment supplies one global scale. That works here only because the synthetic robot
walks at a constant 0.2 m/s; metric, drift-free scale is what the IMU is for.

    ~/.venvs/fox_vio/bin/python vio/vo_baseline.py ~/data/fox_vio/sim_v1 [--plot vio/figures/vo_baseline.png]
    ~/.venvs/fox_vio/bin/python vio/vo_baseline.py DATASET --est traj.tum --frame imu --align se3   # score a VIO run
"""
import argparse
import os

import cv2
import matplotlib
import numpy as np
import yaml
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation as Rot, Slerp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STRIDE, MIN_FLOW, MIN_TRACKS = 5, 1.0, 400   # keyframe every 5 frames (6 Hz); px median flow below = standing still


def vo(ds, k):
    rows = np.loadtxt(os.path.join(ds, "mav0", "cam0", "data.csv"), delimiter=",", dtype=str, skiprows=1)
    ts, poses, prev, pts, key_pts = [], [], None, None, None
    pose = np.eye(4)                                                        # T_w_c, world = first camera
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for i, (stamp, name) in enumerate(rows):
        img = cv2.imread(os.path.join(ds, "mav0", "cam0", "data", name), cv2.IMREAD_GRAYSCALE)
        if prev is not None:                                               # track prev -> img, keep fwd-bwd consistent
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(img, prev, nxt, None, **lk)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 0.5)
            pts, key_pts = nxt[ok], key_pts[ok]
        if i % STRIDE == 0:
            if prev is not None and len(pts) >= 8 and np.median(np.linalg.norm(pts - key_pts, axis=2)) > MIN_FLOW:
                e, inl = cv2.findEssentialMat(key_pts, pts, k, cv2.RANSAC, 0.999, 0.5)
                _, r, t, _ = cv2.recoverPose(e, key_pts, pts, k, mask=inl)  # x_cur = R x_key + t
                step = np.eye(4)
                step[:3, :3], step[:3, 3] = r, t.ravel()
                pose = pose @ np.linalg.inv(step)
            ts.append(int(stamp) * 1e-9)
            poses.append(pose.copy())
            if pts is None or len(pts) < MIN_TRACKS:                       # top up with new corners away from tracks
                mask = np.full(img.shape, 255, np.uint8)
                for x, y in ([] if pts is None else pts.reshape(-1, 2)):
                    cv2.circle(mask, (int(x), int(y)), 8, 0, -1)
                new = cv2.goodFeaturesToTrack(img, 1000, 0.01, 8, mask=mask)
                if new is not None:
                    pts = new if pts is None else np.concatenate([pts, new.astype(np.float32)])
            key_pts = pts.copy()
        prev = img
    return np.array(ts), np.array(poses)


def traj(ts, poses):
    q = Rot.from_matrix(poses[:, :3, :3]).as_quat()
    return PoseTrajectory3D(positions_xyz=poses[:, :3, 3], orientations_quat_wxyz=np.roll(q, 1, axis=1), timestamps=ts)


def ground_truth(ds, ts, t_s_x):
    """GT pose of frame X (T_s_x: X in the IMU frame) at times ts, interpolated from the 200 Hz EuRoC ground truth."""
    gt = np.loadtxt(os.path.join(ds, "mav0", "state_groundtruth_estimate0", "data.csv"), delimiter=",", skiprows=1)
    tg = gt[:, 0] * 1e-9
    ts = ts[(ts >= tg[0]) & (ts <= tg[-1])]
    poses = np.repeat(np.eye(4)[None], len(ts), axis=0)
    poses[:, :3, 3] = np.stack([np.interp(ts, tg, gt[:, 1 + j]) for j in range(3)], axis=1)
    poses[:, :3, :3] = Slerp(tg, Rot.from_quat(gt[:, [5, 6, 7, 4]]))(ts).as_matrix()
    return traj(ts, poses @ t_s_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--est", help="TUM file to score instead of running the VO (e.g. a VIO run)")
    ap.add_argument("--frame", choices=["cam", "imu"], default="cam", help="which frame --est poses are")
    ap.add_argument("--align", choices=["sim3", "se3"], default="sim3", help="sim3 for monocular, se3 for VIO")
    ap.add_argument("--plot", default=None, help="PNG path for the top-down plot")
    ap.add_argument("--label", default="monocular VO baseline (KLT + E)")
    a = ap.parse_args()
    cam = yaml.safe_load(open(os.path.join(a.dataset, "camchain-imucam.yaml")))["cam0"]
    fu, fv, cu, cv = cam["intrinsics"]
    k = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    t_s_c = np.linalg.inv(np.array(cam["T_cam_imu"]))
    if a.est:
        d = np.loadtxt(a.est)
        est = PoseTrajectory3D(positions_xyz=d[:, 1:4], orientations_quat_wxyz=d[:, [7, 4, 5, 6]], timestamps=d[:, 0])
    else:
        ts, poses = vo(a.dataset, k)
        est = traj(ts, poses)
        out = os.path.join(a.dataset, "vo_baseline.tum")
        np.savetxt(out, np.c_[ts, est.positions_xyz, est.orientations_quat_wxyz[:, [1, 2, 3, 0]]], fmt="%.9f")
        print(f"VO: {len(ts)} keyframes -> {out}")
    ref = ground_truth(a.dataset, est.timestamps, t_s_c if a.frame == "cam" else np.eye(4))
    ref, est = sync.associate_trajectories(ref, est, max_diff=0.002)
    _, _, scale = est.align(ref, correct_scale=a.align == "sim3")
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    s = ape.get_all_statistics()
    length = ref.path_length
    print(f"ATE ({a.align} aligned, {len(ref.timestamps)} poses, GT path {length:.2f} m): rmse {s['rmse']:.3f} m "
          f"({100 * s['rmse'] / length:.1f} % of path), mean {s['mean']:.3f}, max {s['max']:.3f} m; scale {scale:.3f}")
    if a.plot:
        fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        ax.plot(*ref.positions_xyz[:, :2].T, color="#2a78d6", lw=2, label="ground truth")
        ax.plot(*est.positions_xyz[:, :2].T, color="#eb6834", lw=2, label=f"{a.label}, {a.align} aligned")
        ax.plot(*ref.positions_xyz[0, :2], "o", color="#52514e", ms=8)
        ax.annotate("start", ref.positions_xyz[0, :2], xytext=(8, -12), textcoords="offset points", color="#52514e")
        ax.set_aspect("equal")
        ax.grid(color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b2")
        ax.tick_params(colors="#52514e")
        ax.set_xlabel("x [m]", color="#52514e")
        ax.set_ylabel("y [m]", color="#52514e")
        ax.set_title(f"ATE rmse {s['rmse']:.3f} m over a {length:.1f} m path", color="#0b0b0b", loc="left")
        ax.legend(frameon=False, labelcolor="#0b0b0b", loc="best")
        fig.tight_layout()
        fig.savefig(a.plot)
        print(f"plot -> {a.plot}")


if __name__ == "__main__":
    main()
