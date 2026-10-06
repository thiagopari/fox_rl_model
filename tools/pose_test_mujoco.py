#!/usr/bin/env python3
"""Drive a few poses through the position servos in MuJoCo and check sign conventions + stability; save a gallery."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'mjcf', 'scene.xml'))
d = mujoco.MjData(m)
act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
foot = {l: m.body(l + '_calf').id for l in ('FL', 'FR', 'RL', 'RR')}

def target(hip=0.0, thigh=0.0, calf=0.0, legs=('FL', 'FR', 'RL', 'RR'), base=None):
    u = np.zeros(m.nu) if base is None else base.copy()
    for l in legs:
        sgn = 1.0 if l[1] == 'L' else -1.0          # abduction outward on both sides
        u[act[l + '_hip']] = sgn * hip
        u[act[l + '_thigh']] = thigh
        u[act[l + '_calf']] = calf
    return u

def settle(u, secs=1.5):
    for _ in range(int(secs / m.opt.timestep)):
        d.ctrl[:] = u
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
                     'track_err': round(float(np.abs(d.qpos[7:] - u[[m.actuator_trnid[i, 0] - 1 for i in range(m.nu)]]).max()), 3)}
    cam.lookat[:] = d.qpos[:3]
    r.update_scene(d, cam)
    shots.append(r.render())
# lift FL foot: bend FL thigh/calf only and check that FL foot rises while the others stay down
base_u = target()
z_before = {l: float(d.xpos[b][2]) for l, b in foot.items()}
settle(target(thigh=0.5, calf=-0.9, legs=('FL',), base=base_u), 1.0)
z_after = {l: float(d.xpos[b][2]) for l, b in foot.items()}
results['lift FL'] = {'FL_calf_rise_m': round(z_after['FL'] - z_before['FL'], 4), 'upright': round(float(d.xmat[1].reshape(3, 3)[2, 2]), 4)}
cam.lookat[:] = d.qpos[:3]
r.update_scene(d, cam)
shots.append(r.render())
Image.fromarray(np.hstack([np.vstack(shots[:2]), np.vstack(shots[2:4])])).save(os.path.join(ROOT, 'validation', 'mujoco_poses.png'))
Image.fromarray(shots[4]).save(os.path.join(ROOT, 'validation', 'mujoco_lift_FL.png'))
for k, v in results.items():
    print('%-28s %s' % (k, v))
crouch, stand = results['crouch thigh+0.35 calf-0.6'], results['stand']
print('crouch lowers the body:', crouch['base_z'] < stand['base_z'] - 0.005, '| stays upright:', all(v['upright'] > 0.95 for v in results.values()))
