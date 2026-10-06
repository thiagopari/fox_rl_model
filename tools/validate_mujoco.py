#!/usr/bin/env python3
"""Validate the Fox MJCF in MuJoCo: structure, masses, foot contact at the zero pose, 4 s standing test, renders."""
import json, os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'mjcf', 'scene.xml'))
d = mujoco.MjData(m)
rep = {'nq': m.nq, 'nv': m.nv, 'nu': m.nu, 'njnt': m.njnt, 'nbody': m.nbody,
       'total_mass_kg': float(m.body_subtreemass[1]),
       'joints': [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]}
ok = {'structure (free base + 12 hinges, 12 actuators)': m.nq == 19 and m.nv == 18 and m.nu == 12}
mujoco.mj_forward(m, d)
# lowest collision point of each calf at the zero pose (feet should all touch the floor)
feet = {}
for g in range(m.ngeom):
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    if b and b.endswith('_calf') and m.geom_group[g] == 3:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
        feet[b] = float(w[:, 2].min())
rep['foot_heights_at_zero_pose_m'] = feet
ok['all four feet within 5 mm of the floor'] = max(feet.values()) - min(feet.values()) < 0.005 and min(feet.values()) > -0.001
# standing test: hold the zero pose with the position servos for 4 s
z0 = float(d.qpos[2])
trace = []
for k in range(int(4.0 / m.opt.timestep)):
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)
    if k % 100 == 0:
        up = d.xmat[1].reshape(3, 3)[:, 2]
        trace.append([round(d.time, 2), round(float(d.qpos[2]), 4), round(float(up[2]), 4)])
up = d.xmat[1].reshape(3, 3)[:, 2]
rep['standing'] = {'base_z_start_m': z0, 'base_z_end_m': float(d.qpos[2]), 'upright_cos': float(up[2]),
                   'max_joint_err_rad': float(np.abs(d.qpos[7:]).max()), 'contacts_end': int(d.ncon),
                   'finite': bool(np.isfinite(d.qpos).all()), 'trace': trace[::5]}
ok['stands for 4 s (upright, <15 mm sag, finite)'] = rep['standing']['finite'] and up[2] > 0.98 and z0 - d.qpos[2] < 0.015
ok['servo torque stays within 0.47 N m'] = bool(np.abs(d.actuator_force).max() <= 0.4701)
rep['actuator_force_end_Nm'] = np.round(d.actuator_force, 4).tolist()
# renders
try:
    r = mujoco.Renderer(m, 720, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.qpos[:3]
    cam.distance, cam.elevation = 0.55, -18
    for az, name in ((135, 'mujoco_front_left.png'), (-45, 'mujoco_rear_right.png')):
        cam.azimuth = az
        r.update_scene(d, cam)
        import PIL.Image as Image  # noqa
        Image.fromarray(r.render()).save(os.path.join(ROOT, 'validation', name))
    rep['renders'] = True
except Exception as e:
    rep['renders'] = 'failed: %s' % e
rep['checks'] = ok = {k: bool(v) for k, v in ok.items()}
os.makedirs(os.path.join(ROOT, 'validation'), exist_ok=True)
json.dump(rep, open(os.path.join(ROOT, 'validation', 'mujoco_report.json'), 'w'), indent=1)
for k, v in ok.items():
    print('%-50s %s' % (k, 'PASS' if v else 'FAIL'))
print('mass %.4f kg | base z %.4f -> %.4f | upright %.4f | feet %s' % (rep['total_mass_kg'], z0, d.qpos[2], up[2], {k: round(v, 4) for k, v in feet.items()}))
sys.exit(0 if all(ok.values()) else 1)
