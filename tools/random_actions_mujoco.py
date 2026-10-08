#!/usr/bin/env python3
"""RL-style robustness check: 20 s of random joint targets (re-sampled every 0.1 s) must never blow up the simulation."""
import os
import numpy as np
import mujoco

ROOT = os.path.abspath(os.environ.get('FOX_MODEL', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # FOX_MODEL=v7: another model folder
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'mjcf', 'scene.xml'))
d = mujoco.MjData(m)
rng = np.random.default_rng(0)
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
worst_v, min_z, falls, steps = 0.0, 1.0, 0, int(20.0 / m.opt.timestep)
u = np.zeros(m.nu)
for k in range(steps):
    if k % 50 == 0:
        u = rng.uniform(lo, hi) * 0.7
    d.ctrl[:] = u
    mujoco.mj_step(m, d)
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        print('NON-FINITE STATE at t=%.3f' % d.time)
        raise SystemExit(1)
    worst_v = max(worst_v, float(np.abs(d.qvel[6:]).max()))
    min_z = min(min_z, float(d.qpos[2]))
    if d.xmat[1].reshape(3, 3)[2, 2] < 0.0:
        falls += 1
        mujoco.mj_resetData(m, d)
pen = min([d.contact[i].dist for i in range(d.ncon)] or [0.0])
print('20 s random actions: finite=True  max joint speed %.1f rad/s  min base z %.3f m  falls/resets %d  deepest contact %.4f m'
      % (worst_v, min_z, falls, pen))
ok = worst_v < 200 and min_z > -0.01 and pen > -0.01
print('PASS' if ok else 'FAIL')
raise SystemExit(0 if ok else 1)
