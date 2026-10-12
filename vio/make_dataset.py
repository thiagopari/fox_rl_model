#!/usr/bin/env python3
"""Synthetic Fox camera + IMU dataset (EuRoC / ASL layout), rendered in MuJoCo.

The base is KINEMATIC: every sample its pose is set from a scripted gait-like trajectory (no physics, legs held at
home). No Isaac Lab policy run is recorded on disk, so nothing is replayed; the trajectory is analytic (see PLAN).

    MUJOCO_GL=egl ~/.venvs/fox_vio/bin/python vio/make_dataset.py ~/data/fox_vio/sim_v1 [--seconds 10] [--check]

Writes OUT/mav0/{cam0,imu0,state_groundtruth_estimate0}/ (EuRoC) and OUT/{camchain-imucam.yaml, imu.yaml} (Kalibr).
--check renders one frame with a marker and asserts that the yaml's K and T_cam_imu project it to the right pixel.
"""
import argparse
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot, Slerp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 9.80665

# Camera: 640x480 @ 30 Hz pinhole, no distortion (an OV9281-like global-shutter module with an M12 lens; chosen, not
# a measured lens). Mounted on the nose (base front face at x = 0.122 m), centred, pitched 10 deg down.
W, H, FPS, F = 640, 480, 30, 400.0                   # F in px: HFOV 77.3 deg, VFOV 61.9 deg
CX, CY = (W - 1) / 2, (H - 1) / 2                    # OpenCV convention (pixel centres at integers)
P_BC, PITCH_DOWN = np.array([0.125, 0.0, 0.015]), np.radians(10.0)

# IMU = BNO055 frame: deploy/fox_pi.py IMU_TO_BODY (upside-down mount, 180 deg about x), at the MJCF 'imu' site (base
# origin). Noise: BNO055 datasheet BST-BNO055-DS000 rev 1.8, Table 0-2 (typ values). Random walks: NOT in the
# datasheet -> ASSUMED; replace with an Allan-variance fit of a long static log.
IMU_HZ = 200
R_BS = np.diag([1.0, -1.0, -1.0])
ACC_ND, GYR_ND = 150e-6 * G, np.radians(0.014)       # noise density: m/s^2/sqrt(Hz), rad/s/sqrt(Hz)
ACC_OFF, GYR_OFF = 0.080 * G, np.radians(1.0)        # typ zero-g / zero-rate offset: initial bias ~ U(-off, off)
ACC_RW, GYR_RW = 1e-3, 1e-4                          # ASSUMED bias random walk: m/s^3/sqrt(Hz), rad/s^2/sqrt(Hz)
ACC_LSB, GYR_LSB = 0.01, np.radians(1 / 16)          # BNO055 output resolution (m/s^2 and dps unit settings)

# Trajectory: stand, ramp to 0.2 m/s, then (duration s, yaw change deg) segments. Trot-like overlay scaled by speed:
# 6 mm peak-to-peak bob and 1 deg pitch at 3 Hz, 1.5 deg roll at 1.5 Hz (chosen, not measured from a policy run;
# the v7 policies walk straight only, so the turns are an assumption).
STAND, RAMP, V, Z0 = 2.0, 1.0, 0.2, 0.147            # Z0: Isaac Sim standing height of the base (README)
PLAN = [(12, 0), (4, 90), (10, 0), (4, 90), (12, 0), (4, 90), (5, 0)]
START = np.array([-1.3, -1.5])
BOB, F_BOB, PITCH_A, ROLL_A = 0.003, 3.0, np.radians(1.0), np.radians(1.5)
ROOM, WALL_H, MM_PX = 3.0, 2.5, 0.0025               # room half-size, wall height (m); texture resolution (m/px)


def trajectory(dt=1e-3):
    """Dense base trajectory: t, position (N,3), Rotation (N), all of it smooth enough to differentiate."""
    t = np.arange(0, STAND + RAMP + sum(d for d, _ in PLAN) + dt / 2, dt)
    u = np.clip((t - STAND) / RAMP, 0, 1)
    v = V * u * u * (3 - 2 * u)
    wz, t0 = np.zeros_like(t), STAND + RAMP
    for dur, deg in PLAN:
        u = (t - t0) / dur
        on = (u >= 0) & (u < 1)
        wz[on] = np.radians(deg) / dur * 2 * np.sin(np.pi * u[on]) ** 2      # integrates to deg over the segment
        t0 += dur
    yaw = np.cumsum(wz) * dt
    xy = START + np.cumsum(np.c_[v * np.cos(yaw), v * np.sin(yaw)], axis=0) * dt
    k, ph = v / V, 2 * np.pi * F_BOB * t
    z = Z0 + BOB * k * np.sin(ph)
    att = Rot.from_euler("ZYX", np.c_[yaw, PITCH_A * k * np.sin(ph + np.pi / 2), ROLL_A * k * np.sin(ph / 2)])
    return t, np.c_[xy, z], att


def imu(t, p, att, rng):
    """200 Hz BNO055-like samples + ground truth of the IMU frame S (EuRoC: R = world, S = sensor)."""
    dt = t[1] - t[0]
    vel = np.gradient(p, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    w_b = np.zeros_like(p)
    w_b[1:-1] = (att[:-2].inv() * att[2:]).as_rotvec() / (2 * dt)           # body rates, central difference
    w_b[0], w_b[-1] = w_b[1], w_b[-2]
    i = np.arange(0, len(t), round(1 / (IMU_HZ * dt)))
    r_ws = att[i] * Rot.from_matrix(R_BS)
    w_s = w_b[i] @ R_BS                                                      # R_BS^T w_b, row-wise
    f_s = r_ws.inv().apply(acc[i] + [0, 0, G])                               # specific force
    n, ds = len(i), 1 / IMU_HZ
    bg = rng.uniform(-GYR_OFF, GYR_OFF, 3) + np.cumsum(rng.normal(0, GYR_RW * np.sqrt(ds), (n, 3)), axis=0)
    ba = rng.uniform(-ACC_OFF, ACC_OFF, 3) + np.cumsum(rng.normal(0, ACC_RW * np.sqrt(ds), (n, 3)), axis=0)
    gyro = np.round((w_s + bg + rng.normal(0, GYR_ND / np.sqrt(ds), (n, 3))) / GYR_LSB) * GYR_LSB
    accel = np.round((f_s + ba + rng.normal(0, ACC_ND / np.sqrt(ds), (n, 3))) / ACC_LSB) * ACC_LSB
    # self-check: the noise-free specific force, rotated back and integrated, must reproduce the true velocity
    a_w = r_ws.apply(f_s) - [0, 0, G]
    v_int = vel[i[0]] + np.r_[np.zeros((1, 3)), np.cumsum((a_w[1:] + a_w[:-1]) / 2 * ds, axis=0)]
    assert np.abs(v_int - vel[i]).max() < 2e-3, np.abs(v_int - vel[i]).max()
    return t[i], gyro, accel, p[i], r_ws, vel[i], bg, ba


def texture(rng, w, h):
    """Non-repeating, corner-rich texture: smooth colour noise under a few thousand random shapes of many sizes."""
    small = rng.integers(0, 256, (h // 48 + 2, w // 48 + 2, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    dr = ImageDraw.Draw(img)
    for _ in range(w * h // 1500):
        x, y, s = rng.uniform(0, w), rng.uniform(0, h), np.exp(rng.uniform(np.log(5), np.log(110)))
        c, kind = tuple(int(v) for v in rng.integers(0, 256, 3)), rng.integers(3)
        if kind == 0:
            dr.rectangle([x, y, x + s * rng.uniform(0.3, 2), y + s * rng.uniform(0.3, 2)], fill=c)
        elif kind == 1:
            dr.ellipse([x, y, x + s, y + s * rng.uniform(0.3, 1.5)], fill=c)
        else:
            dr.line([x, y, x + rng.normal(0, 2 * s), y + rng.normal(0, 2 * s)], fill=c, width=int(rng.integers(1, 6)))
    return np.asarray(img)


def build(rng, marker=None):
    """v7 Fox + nose camera 'cam0' in a 6 x 6 m textured room with clutter boxes along the walls."""
    spec = mujoco.MjSpec.from_file(os.path.join(ROOT, "v7", "mjcf", "scene_reduced.xml"))
    for el in [spec.geom("floor"), *spec.lights]:                           # its floor and shadow-casting light
        spec.delete(el)
    spec.visual.headlight.diffuse, spec.visual.headlight.specular, spec.visual.headlight.ambient = [0] * 3, [0] * 3, [0.35] * 3
    for pos in ([2, 1, 3], [-2, -1, 3]):
        spec.worldbody.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=pos, dir=-np.array(pos, float),
                                 diffuse=[0.45] * 3, specular=[0] * 3, castshadow=False)

    def surface(name, size, mm_px=MM_PX, **geom):
        """A plane (half-sizes size[0] x size[1]) or a box (cube texture, 6 different faces), ~mm_px per pixel."""
        tw, th = (max(64, int(2 * s / mm_px)) for s in size[:2])
        kind = mujoco.mjtTexture.mjTEXTURE_2D
        if geom["type"] == mujoco.mjtGeom.mjGEOM_BOX:
            tw = max(64, int(2 * max(size) / mm_px))
            th, kind = 6 * tw, mujoco.mjtTexture.mjTEXTURE_CUBE
        tex = spec.add_texture(name=name, type=kind, width=tw, height=th, nchannel=3)
        tex.data = texture(rng, tw, th).tobytes()
        mat = spec.add_material(name=name, specular=0, shininess=0, reflectance=0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        spec.worldbody.add_geom(name=name, size=size, material=name, contype=0, conaffinity=0, **geom)

    plane = mujoco.mjtGeom.mjGEOM_PLANE
    surface("floor", [ROOM, ROOM, 0.1], MM_PX / 2, type=plane)              # seen closest: twice as fine
    for k, yaw in enumerate((0, 90, 180, 270)):                              # walls face the room centre
        q = Rot.from_euler("ZY", [yaw, -90], degrees=True)                    # plane normal (local z) -> inward
        surface(f"wall{k}", [WALL_H / 2, ROOM, 0.1], type=plane, quat=np.roll(q.as_quat(), 1),
                pos=[ROOM * np.cos(np.radians(yaw)), ROOM * np.sin(np.radians(yaw)), WALL_H / 2])
    for k in range(16):                                                      # boxes 0.25-0.75 m in from a wall
        side, along, inset = rng.integers(4), rng.uniform(-2.6, 2.6), rng.uniform(0.25, 0.75)
        hs = rng.uniform(0.08, 0.3, 3)
        c, s = np.cos(np.pi / 2 * side), np.sin(np.pi / 2 * side)
        xy = (ROOM - inset) * np.array([c, s]) + along * np.array([-s, c])
        surface(f"box{k}", hs, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*xy, hs[2]],
                quat=np.roll(Rot.from_euler("z", rng.uniform(0, 90), degrees=True).as_quat(), 1))
    # MuJoCo cameras look along -z with y up; the camera frame in the body: x = -y_body, y = "up" pitched, z = -forward
    fwd = np.array([np.cos(PITCH_DOWN), 0, -np.sin(PITCH_DOWN)])
    r_b_cmj = np.c_[[0, -1, 0], [np.sin(PITCH_DOWN), 0, np.cos(PITCH_DOWN)], -fwd]
    spec.body("base").add_camera(name="cam0", pos=P_BC, quat=np.roll(Rot.from_matrix(r_b_cmj).as_quat(), 1),
                                 fovy=np.degrees(2 * np.arctan(H / 2 / F)))
    if marker is not None:
        spec.worldbody.add_geom(name="marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], pos=marker,
                                rgba=[1, 0, 0, 1], contype=0, conaffinity=0)
    m = spec.compile()
    m.vis.map.znear = 0.01 / m.stat.extent                                  # 1 cm near plane
    return m, r_b_cmj @ np.diag([1.0, -1.0, -1.0])                          # R_B_C, OpenCV camera frame


_r = {}


def shot(m, pos, rot):
    """RGB image of cam0 with the base at pos / rot (kinematic: qpos set, legs at home)."""
    if id(m) not in _r:
        _r[id(m)] = (mujoco.Renderer(m, H, W), mujoco.MjData(m))
    r, d = _r[id(m)]
    d.qpos[0:3], d.qpos[3:7] = pos, np.roll(rot.as_quat(), 1)
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera="cam0")
    return r.render()


def t_cam_imu(r_b_c):
    """Kalibr T_cam_imu = T_C_S: maps points from the IMU (BNO055) frame into the OpenCV camera frame."""
    t_s_c = np.eye(4)
    t_s_c[:3, :3], t_s_c[:3, 3] = R_BS.T @ r_b_c, R_BS.T @ P_BC              # IMU at the base origin
    return np.linalg.inv(t_s_c), t_s_c


def write_yamls(out, t_c_s, t_s_c):
    mat = lambda t: "\n".join(f"  - [{', '.join(f'{v:.9f}' for v in row)}]" for row in t)
    flat = lambda t: ", ".join(f"{v:.9f}" for v in t.ravel())
    with open(os.path.join(out, "camchain-imucam.yaml"), "w") as f:
        f.write(f"cam0:\n  T_cam_imu:\n{mat(t_c_s)}\n  camera_model: pinhole\n  intrinsics: [{F}, {F}, {CX}, {CY}]\n"
                f"  distortion_model: radtan\n  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n  resolution: [{W}, {H}]\n"
                f"  rostopic: /cam0/image_raw\n  timeshift_cam_imu: 0.0\n  cam_overlaps: []\n")
    with open(os.path.join(out, "imu.yaml"), "w") as f:
        f.write(f"# BNO055 datasheet noise densities (typ); random walks ASSUMED (not in the datasheet)\n"
                f"accelerometer_noise_density: {ACC_ND:.6e}\naccelerometer_random_walk: {ACC_RW:.6e}\n"
                f"gyroscope_noise_density: {GYR_ND:.6e}\ngyroscope_random_walk: {GYR_RW:.6e}\n"
                f"rostopic: /imu0\nupdate_rate: {IMU_HZ}.0\n")
    with open(os.path.join(out, "mav0", "cam0", "sensor.yaml"), "w") as f:
        f.write(f"sensor_type: camera\ncomment: synthetic Fox nose camera (MuJoCo)\nT_BS:\n  cols: 4\n  rows: 4\n"
                f"  data: [{flat(t_s_c)}]\nrate_hz: {FPS}\nresolution: [{W}, {H}]\ncamera_model: pinhole\n"
                f"intrinsics: [{F}, {F}, {CX}, {CY}]\ndistortion_model: radial-tangential\n"
                f"distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n")
    with open(os.path.join(out, "mav0", "imu0", "sensor.yaml"), "w") as f:
        f.write(f"sensor_type: imu\ncomment: synthetic BNO055 (upside-down mount)\nT_BS:\n  cols: 4\n  rows: 4\n"
                f"  data: [{flat(np.eye(4))}]\nrate_hz: {IMU_HZ}\n"
                f"gyroscope_noise_density: {GYR_ND:.6e}\ngyroscope_random_walk: {GYR_RW:.6e}\n"
                f"accelerometer_noise_density: {ACC_ND:.6e}\naccelerometer_random_walk: {ACC_RW:.6e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=None, help="cut the trajectory (default: all of PLAN)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="projection self-check only")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t, p, att = trajectory()
    if a.seconds:
        keep = t <= a.seconds
        t, p, att = t[keep], p[keep], att[keep]

    if a.check:                                   # marker 1.2 m ahead, 0.2 m left, 0.25 m up of the camera at t = 0
        m0, r_b_c = build(np.random.default_rng(a.seed))
        marker = p[0] + att[0].apply(P_BC + r_b_c @ np.array([-0.2, -0.25, 1.2]))
        m1, _ = build(np.random.default_rng(a.seed), marker)                 # same seed: same room
        diff = np.abs(shot(m1, p[0], att[0]).astype(int) - shot(m0, p[0], att[0])).max(-1) > 30
        ys, xs = np.nonzero(diff)
        t_c_s, _ = t_cam_imu(r_b_c)                                          # project through the yaml's chain
        pc = t_c_s[:3, :3] @ (att[0] * Rot.from_matrix(R_BS)).inv().apply(marker - p[0]) + t_c_s[:3, 3]
        uv = F * pc[:2] / pc[2] + [CX, CY]
        err = np.hypot(xs.mean() - uv[0], ys.mean() - uv[1])
        print(f"marker centroid ({xs.mean():.2f}, {ys.mean():.2f}) px, projected ({uv[0]:.2f}, {uv[1]:.2f}), "
              f"error {err:.3f} px over {diff.sum()} px")
        assert diff.sum() > 50 and err < 0.5
        return

    m, r_b_c = build(rng)
    ti, gyro, accel, p_s, r_ws, v_s, bg, ba = imu(t, p, att, rng)
    mav = os.path.join(a.out, "mav0")
    for sub in ("cam0/data", "imu0", "state_groundtruth_estimate0"):
        os.makedirs(os.path.join(mav, sub), exist_ok=True)
    ns = np.round(ti * 1e9).astype(np.int64)
    np.savetxt(os.path.join(mav, "imu0", "data.csv"), np.c_[ns, gyro, accel], delimiter=",",
               fmt=["%d"] + ["%.6f"] * 6, comments="",
               header="#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                      "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")
    q = r_ws.as_quat()                                                      # x y z w
    np.savetxt(os.path.join(mav, "state_groundtruth_estimate0", "data.csv"),
               np.c_[ns, p_s, q[:, 3], q[:, :3], v_s, bg, ba], delimiter=",", fmt=["%d"] + ["%.9f"] * 16, comments="",
               header="#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], "
                      "q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], "
                      "b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], "
                      "b_a_RS_S_z [m s^-2]")
    t_c_s, t_s_c = t_cam_imu(r_b_c)
    write_yamls(a.out, t_c_s, t_s_c)

    tc = np.arange(0, t[-1], 1 / FPS)
    cns = np.round(tc * 1e9).astype(np.int64)
    pc = np.stack([np.interp(tc, t, p[:, k]) for k in range(3)], axis=1)
    ac = Slerp(t, att)(tc)
    with open(os.path.join(mav, "cam0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],filename\n")
        for n_, pos, rot in zip(cns, pc, ac):
            gray = np.asarray(Image.fromarray(shot(m, pos, rot)).convert("L"))   # mono, like the OV9281
            Image.fromarray(gray).save(os.path.join(mav, "cam0", "data", f"{n_}.png"), compress_level=3)
            f.write(f"{n_},{n_}.png\n")
    print(f"{len(cns)} frames @ {FPS} Hz, {len(ns)} IMU samples @ {IMU_HZ} Hz, {t[-1]:.1f} s, "
          f"path {np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum():.2f} m -> {a.out}")


if __name__ == "__main__":
    main()
