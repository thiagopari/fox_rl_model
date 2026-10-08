#!/usr/bin/env python3
"""Drive a few poses through the 12 servos of the exact mechanism in MuJoCo and check stability; save a gallery.
Poses are given as hip / thigh / calf angles from the stance (mechanism.json; the CAD pose if it has none) and converted
with the servo map: pivot = thigh, gear = -N (thigh + calf)."""
import json
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from PIL import Image

ROOT = os.path.abspath(os.environ.get('FOX_MODEL', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # FOX_MODEL=v7: another model folder
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'mjcf', 'scene.xml'))
d = mujoco.MjData(m)
act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
mech = json.load(open(os.path.join(ROOT, 'mechanism.json')))
N = mech['servo_map'].get('pinion_per_gear', 1.0)
stance = mech.get('stance', {}).get('joints_rad', {})

def target(hip=0.0, thigh=0.0, calf=0.0, legs=('FL', 'FR', 'RL', 'RR'), base=None):
    u = np.zeros(m.nu) if base is None else base.copy()
    for l in legs:
        sgn = 1.0 if l[1] == 'L' else -1.0          # abduction outward on both sides
        th, ca = thigh + stance.get(l + '_thigh_joint', 0.0), calf + stance.get(l + '_calf_joint', 0.0)
        u[act[l + '_hip']] = sgn * hip
        u[act[l + '_pivot']] = th               # pivot servo turns the femur
        u[act[l + '_gear']] = -N * (th + ca)    # gear servo sets the shin angle through the N:1 gear + parallelogram
    return u

def settle(u, secs=1.5, ramp=0.5):
    """Ramp the servo targets from where they are to u over `ramp` s, then hold. (A step on all 12 servos saturates
    them together and the robot hops and can flip: the pivot and gear servos share the knee load in the mechanism.)"""
    u0 = d.ctrl.copy()
    for k in range(int(secs / m.opt.timestep)):
        d.ctrl[:] = u0 + (u - u0) * min(1.0, k * m.opt.timestep / ramp)
        mujoco.mj_step(m, d)

r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 0.55, -15, 120
shots, results = [], {}
poses = [('stand', target()),
         ('crouch thigh+0.35 calf-0.6', target(thigh=0.35, calf=-0.6)),
         ('abduct 0.25', target(hip=0.25)),
         ('stand again', target())]
for name, u in poses:
    settle(u)
    up = d.xmat[1].reshape(3, 3)[:, 2]
    results[name] = {'base_z': round(float(d.qpos[2]), 4), 'upright': round(float(up[2]), 4),
                     'track_err': round(float(np.abs(d.actuator_length - u).max()), 3)}
    cam.lookat[:] = d.qpos[:3]
    r.update_scene(d, cam)
    shots.append(r.render())
Image.fromarray(np.hstack([np.vstack(shots[:2]), np.vstack(shots[2:4])])).save(os.path.join(ROOT, 'validation', 'mujoco_poses.png'))
for k, v in results.items():
    print('%-28s %s' % (k, v))
crouch, stand = results['crouch thigh+0.35 calf-0.6'], results['stand']
print('crouch lowers the body:', crouch['base_z'] < stand['base_z'] - 0.005, '| stays upright:', all(v['upright'] > 0.9 for v in results.values()))
# single-leg retraction is checked with the body held (tools/mechanism_mujoco.py): on 3 feet this robot tips, its COM is on the support edge
