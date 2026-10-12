#!/usr/bin/env python3
"""Degradation study on the sim_v1 render: where does mono VIO break on a small quadruped, and why calibrate?
OpenVINS runs on variants of the same images / IMU (no re-render), one seed per level, scored like vo_baseline.py.

  time_offset   truth: camera and IMU in sync; OpenVINS is told timeshift_cam_imu = level (online calibration off).
                Same error as an unmodelled IMU stamp shift of -level.
  extrinsic     T_imu_cam rotation perturbed by level deg about one fixed random axis (online calibration off).
  vibration     foot strikes: at every trot touchdown (3 Hz, scaled by speed like the bob) a damped 25 Hz ring
                (tau 20 ms) on accel and gyro along random axes; level = accel envelope m/s^2 (peak ~0.6 x),
                gyro envelope = 0.1 rad/s per m/s^2.
                IMU only (0.4 mm / 0.4 deg of real motion at 10 m/s^2, ignored in the images).
  bias_rw       true bias random walk = level x the assumed one, OpenVINS still told 1x.
  noise         shot + read noise: DN = g * Poisson(DN / g) + N(0, read^2); level = g (DN per e-), read = 2 g DN.
  blur          linear motion blur from the true camera rate at the image centre over an exposure of level ms
                (rotation only; roll and translation blur ignored). The VO baseline also runs on noise / blur.
Online-calibration runs (calib_cam_timeoffset / calib_cam_extrinsics on) at the worst level show what Kalibr-free
self-calibration recovers.

    MUJOCO_GL=egl ~/.venvs/fox_vio/bin/python vio/sweep.py ~/data/fox_vio/sim_v1 [--jobs 4] [--rate 2]
writes vio/results/sweep.csv and vio/figures/sweep.png; variant datasets go next to the source dataset.
"""
import argparse
import copy
import csv
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

import make_dataset as md
import vo_baseline as vb

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ([("baseline", 0, {})]
        + [("time_offset", ms, {"dt": ms / 1000}) for ms in (5, 10, 20, 40)]
        + [("time_offset+calib", 40, {"dt": 0.04, "calib_cam_timeoffset": "true"})]
        + [("extrinsic", deg, {"rot": deg}) for deg in (1, 2, 5)]
        + [("extrinsic+calib", 5, {"rot": 5, "calib_cam_extrinsics": "true"})]
        + [("vibration", a, {"data": f"vib{a}"}) for a in (2, 5, 10)]
        + [("bias_rw", 10, {"data": "rw10"})]
        + [("noise", g, {"data": f"noise{g}"}) for g in (0.25, 1, 4)]
        + [("blur", ms, {"data": f"blur{ms}"}) for ms in (5, 15, 33)])


def variant_imu(src, dst, kind, level, seed=1):
    rng = np.random.default_rng(seed)
    d = np.loadtxt(os.path.join(src, "mav0", "imu0", "data.csv"), delimiter=",", skiprows=1)
    t = d[:, 0] * 1e-9
    if kind == "vib":                                   # touchdowns at the bob minima, scaled by speed like the bob
        tt, p, _ = md.trajectory()
        k = np.interp(t, tt[1:], np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / (tt[1] - tt[0]) / md.V)
        for tk in np.arange(md.STAND + 0.75 / md.F_BOB, t[-1], 1 / md.F_BOB):
            on = (t >= tk) & (t < tk + 0.1)
            ring = np.exp(-(t[on] - tk) / 0.02) * np.sin(2 * np.pi * 25 * (t[on] - tk)) * np.interp(tk, t, k)
            u = rng.normal(size=(2, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            d[on, 4:7] += level * ring[:, None] * u[0]
            d[on, 1:4] += 0.1 * level * ring[:, None] * u[1]
    else:                                               # extra independent walk: total = level x the assumed one
        ds = 1 / md.IMU_HZ
        for cols, rw in ((slice(1, 4), md.GYR_RW), (slice(4, 7), md.ACC_RW)):
            d[:, cols] += np.cumsum(rng.normal(0, np.sqrt(level ** 2 - 1) * rw * np.sqrt(ds), (len(t), 3)), axis=0)
    os.makedirs(os.path.join(dst, "mav0", "imu0"), exist_ok=True)
    with open(os.path.join(src, "mav0", "imu0", "data.csv")) as f:
        header = f.readline().strip()
    np.savetxt(os.path.join(dst, "mav0", "imu0", "data.csv"), d, delimiter=",", fmt=["%d"] + ["%.6f"] * 6,
               header=header.lstrip("#"), comments="#")
    for sub in ("cam0", "state_groundtruth_estimate0"):
        os.symlink(os.path.join(src, "mav0", sub), os.path.join(dst, "mav0", sub))


def _degrade(job):
    src_png, dst_png, kind, level, kernel, seed = job
    img = cv2.imread(src_png, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        img = level * rng.poisson(img / level) + rng.normal(0, 2 * level, img.shape)
    elif kernel is not None:
        img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    cv2.imwrite(dst_png, np.clip(np.round(img), 0, 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def blur_kernel(flow_px):
    n = float(np.hypot(*flow_px))
    if n < 0.5:
        return None
    size = 2 * int(np.ceil(n / 2)) + 1
    k = np.zeros((size, size), np.float32)
    c, h = size // 2, np.asarray(flow_px) / 2
    cv2.line(k, (round((c - h[0]) * 16), round((c - h[1]) * 16)), (round((c + h[0]) * 16), round((c + h[1]) * 16)),
             1.0, 1, cv2.LINE_AA, shift=4)
    return k / k.sum()


def variant_img(src, dst, kind, level):
    rows = np.loadtxt(os.path.join(src, "mav0", "cam0", "data.csv"), delimiter=",", dtype=str, skiprows=1)
    kernels = [None] * len(rows)
    if kind == "blur":                                  # camera rate from the ground truth, OpenCV camera frame
        gt = np.loadtxt(os.path.join(src, "mav0", "state_groundtruth_estimate0", "data.csv"), delimiter=",", skiprows=1)
        tg, att = gt[:, 0] * 1e-9, Rot.from_quat(gt[:, [5, 6, 7, 4]])
        cam = yaml.safe_load(open(os.path.join(src, "camchain-imucam.yaml")))["cam0"]
        r_sc = Rot.from_matrix(np.linalg.inv(np.array(cam["T_cam_imu"]))[:3, :3])
        e = level / 2000                                # half the exposure, s
        tc = np.clip(rows[:, 0].astype(np.int64) * 1e-9, tg[0] + e, tg[-1] - e)
        s = Slerp(tg, att)
        w = ((s(tc - e) * r_sc).inv() * (s(tc + e) * r_sc)).as_rotvec() / (2 * e)
        kernels = [blur_kernel(md.F * np.array([-wi[1], wi[0]]) * level / 1000) for wi in w]
    os.makedirs(os.path.join(dst, "mav0", "cam0", "data"), exist_ok=True)
    shutil.copy(os.path.join(src, "mav0", "cam0", "data.csv"), os.path.join(dst, "mav0", "cam0"))
    jobs = [(os.path.join(src, "mav0", "cam0", "data", n), os.path.join(dst, "mav0", "cam0", "data", n), kind, level,
             kernels[i], i) for i, (_, n) in enumerate(rows)]
    with Pool(8) as pool:
        pool.map(_degrade, jobs, chunksize=32)
    for sub in ("imu0", "state_groundtruth_estimate0"):
        os.symlink(os.path.join(src, "mav0", sub), os.path.join(dst, "mav0", sub))


def make_variant(src, name):
    dst = os.path.join(os.path.dirname(src), "sweep", name)
    if os.path.exists(os.path.join(dst, "ros2bag")):
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    kind, level = re.match(r"([a-z]+)([\d.]+)", name).groups()
    if kind in ("vib", "rw"):
        variant_imu(src, dst, kind, float(level))
    else:
        variant_img(src, dst, kind, float(level))
    for f in ("camchain-imucam.yaml", "imu.yaml"):
        shutil.copy(os.path.join(src, f), dst)
    subprocess.run([sys.executable, os.path.join(HERE, "openvins", "prepare.py"), dst], check=True,
                   stdout=subprocess.DEVNULL)
    return dst


def config(ds, cfg, mods):
    """Copy DS/openvins to DS/cfg and apply the calibration errors / switches in mods."""
    out = os.path.join(ds, cfg)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(os.path.join(ds, "openvins"), out, ignore=shutil.ignore_patterns("ov_*", "*.log"))
    est = open(os.path.join(out, "estimator_config.yaml")).read()
    for key in ("calib_cam_timeoffset", "calib_cam_extrinsics"):
        if key in mods:
            est = re.sub(rf"^{key}: \w+", f"{key}: {mods[key]}", est, flags=re.M)
    open(os.path.join(out, "estimator_config.yaml"), "w").write(est)
    chain = open(os.path.join(out, "kalibr_imucam_chain.yaml")).read()
    if "rot" in mods:
        t = np.array(yaml.safe_load(chain.replace("%YAML:1.0", ""))["cam0"]["T_imu_cam"])
        axis = np.random.default_rng(0).normal(size=3)
        t[:3, :3] = Rot.from_rotvec(np.radians(mods["rot"]) * axis / np.linalg.norm(axis)).as_matrix() @ t[:3, :3]
        rows = "\n".join(f"    - [{', '.join(f'{v:.9f}' for v in r)}]" for r in t)
        chain = re.sub(r"  T_imu_cam:\n(    - \[.*\]\n){4}", f"  T_imu_cam:\n{rows}\n", chain)
    if "dt" in mods:
        chain += f"  timeshift_cam_imu: {mods['dt']}\n"
    open(os.path.join(out, "kalibr_imucam_chain.yaml"), "w").write(chain)
    return out


def score(ds, tum):
    d = np.loadtxt(tum, ndmin=2)
    if len(d) < 10:
        return dict(ate=np.nan, ate_max=np.nan, scale=np.nan, poses=len(d), first=np.nan)
    est = vb.PoseTrajectory3D(positions_xyz=d[:, 1:4], orientations_quat_wxyz=d[:, [7, 4, 5, 6]], timestamps=d[:, 0])
    ref = vb.ground_truth(ds, est.timestamps, np.eye(4))
    ref, est = vb.sync.associate_trajectories(ref, est, max_diff=0.002)
    e3 = copy.deepcopy(est)                             # for the Sim(3) scale
    est.align(ref)
    _, _, scale = e3.align(ref, correct_scale=True)
    ape = vb.metrics.APE(vb.metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    s = ape.get_all_statistics()
    return dict(ate=s["rmse"], ate_max=s["max"], scale=scale, poses=len(d), first=d[0, 0])


def run(args, panel, level, mods):
    ds = make_variant(args.dataset, mods["data"]) if "data" in mods else args.dataset
    cfg = f"sw_{panel}_{level}".replace("+", "_")
    tum = os.path.join(ds, cfg, "ov_estimate.tum")
    if not (args.resume and os.path.exists(tum) and os.path.getsize(tum) > 0):
        config(ds, cfg, mods)
        subprocess.run([os.path.join(HERE, "openvins", "run.sh"), ds, cfg, str(args.rate)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = dict(panel=panel, level=level, **score(ds, tum))
    if panel in ("noise", "blur"):                      # the VO baseline on the same images (camera frame, Sim(3))
        out = subprocess.run([sys.executable, os.path.join(HERE, "vo_baseline.py"), ds],
                             capture_output=True, text=True).stdout
        m = re.search(r"rmse ([\d.]+) m", out)
        r["vo_ate"] = float(m.group(1)) if m else np.nan
    print(r, flush=True)
    return r


PANELS = [("time_offset", "camera-IMU time offset not in the calibration [ms]"),
          ("extrinsic", "T_cam_imu rotation error [deg]"),
          ("vibration", "foot-strike vibration, accel envelope [m/s$^2$]"),
          ("bias_rw", "true bias random walk / assumed [x]"),
          ("noise", "shot noise gain g [DN/e-], read 2g DN"),
          ("blur", "exposure, motion blur [ms]")]


def plot(csv_path, png, vo_clean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(csv_path)))
    base = next(float(r["ate"]) for r in rows if r["panel"] == "baseline")
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (panel, xlabel) in zip(axs.ravel(), PANELS):
        pts = sorted([(float(r["level"]), float(r["ate"]), r.get("vo_ate")) for r in rows if r["panel"] == panel])
        x0 = 1.0 if panel == "bias_rw" else 0.0
        xs, ys = [x0] + [p[0] for p in pts], [base] + [p[1] for p in pts]
        ax.set_facecolor("#fcfcfb")
        ax.plot(xs, ys, "-o", color="#2a78d6", lw=2, ms=6, label="OpenVINS (mono + IMU), SE(3)")
        for x, y in zip(xs, ys):
            if y > 1:
                ax.annotate("diverged" if y > 10 else f"{y:.2f} m", (x, y), xytext=(4, 4), textcoords="offset points",
                            color=muted, fontsize=8)
        cal = [r for r in rows if r["panel"] == panel + "+calib"]
        if cal:
            ax.plot([float(r["level"]) for r in cal], [float(r["ate"]) for r in cal], "s", color="#1baf7a", ms=8,
                    label="same, online calibration on")
        if panel in ("noise", "blur"):
            vx = [0.0] + [p[0] for p in pts if p[2]]
            vy = [vo_clean] + [float(p[2]) for p in pts if p[2]]
            ax.plot(vx, vy, "-o", color="#eb6834", lw=2, ms=6, label="mono VO baseline, Sim(3)")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=muted, fontsize=9)
        ax.grid(color=grid, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
        ax.legend(frameon=False, fontsize=7, labelcolor=ink, loc="upper left")
    for ax in axs[:, 0]:
        ax.set_ylabel("ATE RMSE [m]", color=muted)
    fig.suptitle(f"Where mono VIO breaks on the synthetic Fox run (best case {base:.3f} m over 10.4 m; one seed per level)",
                 color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png)
    print("->", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--only", default=None, help="comma-separated panels")
    ap.add_argument("--resume", action="store_true", help="score existing estimates instead of re-running them")
    ap.add_argument("--plot", action="store_true", help="only redraw vio/figures/sweep.png from the csv")
    ap.add_argument("--vo-clean", type=float, default=0.027, help="VO baseline ATE on the clean images (vo_baseline.py)")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", "sweep.csv")
    if a.plot:
        return plot(out, os.path.join(HERE, "figures", "sweep.png"), a.vo_clean)
    a.dataset = os.path.realpath(a.dataset)
    runs = [r for r in RUNS if a.only is None or r[0] in a.only.split(",")]
    for name in sorted({m["data"] for _, _, m in runs if "data" in m}):   # variants first (CPU-heavy), then runs
        make_variant(a.dataset, name)
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda r: run(a, *r), runs))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    keys = ["panel", "level", "ate", "ate_max", "scale", "poses", "first", "vo_ate"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("->", out)
    plot(out, os.path.join(HERE, "figures", "sweep.png"), a.vo_clean)


if __name__ == "__main__":
    main()
