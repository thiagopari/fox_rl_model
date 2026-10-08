#!/usr/bin/env python3
"""What each servo does, and the reduced tree vs the exact mechanism, in MuJoCo (body held in the air, no gravity).
  gear servo alone       crank turns 1/N the other way (N:1 gears), the shin keeps the crank's angle: shin swings
  pivot servo alone      femur turns, the shin keeps its angle: the knee bends, the leg extends / retracts
  pivot +a with gear -Na femur and crank turn together: the whole leg swings rigidly about the hip, like a pendulum
  equivalence            mjcf/fox_reduced.xml (tree + gear-servo tendon + four-bar equality) puts every foot where the
                         exact mechanism (mjcf/fox.xml) does, for random servo targets; both stand at the same height.
Writes validation/mechanism_report.json and validation/mujoco_servo_roles.png."""
import json, os
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.environ.get('FOX_MODEL', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # FOX_MODEL=v7: another model folder
N = json.load(open(os.path.join(ROOT, 'mechanism.json')))['servo_map'].get('pinion_per_gear', 1.0)
LEGS = ('FL', 'FR', 'RL', 'RR')


def load(name, held):
    """held: robot only (no floor), base welded to the world, gravity off."""
    path = os.path.join(ROOT, 'mjcf', name)
    if not held:
        return mujoco.MjModel.from_xml_path(path)
    tmp = os.path.join(ROOT, 'mjcf', '_held_' + name)
    xml = open(path).read().replace('<freejoint name="root"/>', '').replace('<option ', '<option gravity="0 0 0" ', 1)
    open(tmp, 'w').write(xml)
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)


def foot_point(m, d, leg, vid=None):
    """World position of the lowest collision vertex of the leg's foot (picked once at the zero pose)."""
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(leg + '_foot').id and m.geom_group[i] == 3)
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
    vid = int(np.argmin(w[:, 2])) if vid is None else vid
    return w[vid], vid


def settle(m, d, u, secs=1.5):
    for _ in range(int(secs / m.opt.timestep)):
        d.ctrl[:] = u
        mujoco.mj_step(m, d)


def servo_u(m, cmd):
    """cmd: {'pivot': a, 'gear': b, 'hip': c} applied to every leg (hip mirrored so + is outward)."""
    u = np.zeros(m.nu)
    for leg in LEGS:
        for k, v in cmd.items():
            u[m.actuator(leg + '_' + k).id] = v * (-1 if k == 'hip' and leg[1] == 'R' else 1)
    return u


ex = load('fox.xml', True)
dx = mujoco.MjData(ex)
mujoco.mj_forward(ex, dx)
hip_axis = {leg: dx.xanchor[ex.joint(leg + '_femur_joint').id].copy() for leg in LEGS}
foot0, vid = {}, {}
for leg in LEGS:
    foot0[leg], vid[leg] = foot_point(ex, dx, leg)


def describe(cmd):
    """Foot motion per leg: swing = change of the hip->foot direction (rad, about +y), extension = change of its length (m)."""
    mujoco.mj_resetData(ex, dx)
    settle(ex, dx, servo_u(ex, cmd))
    out = {}
    for leg in LEGS:
        r0, r1 = (foot0[leg] - hip_axis[leg])[[0, 2]], (foot_point(ex, dx, leg, vid[leg])[0] - hip_axis[leg])[[0, 2]]
        q = lambda n: float(dx.qpos[ex.jnt_qposadr[ex.joint('%s_%s_joint' % (leg, n)).id]])
        out[leg] = {'swing_rad': round(float(np.arctan2(r0[1] * r1[0] - r0[0] * r1[1], r0 @ r1)), 4),
                    'extension_m': round(float(np.linalg.norm(r1) - np.linalg.norm(r0)), 4),
                    'femur_rad': round(q('femur'), 4), 'shin_rad': round(q('femur') + q('knee'), 4), 'knee_rad': round(q('knee'), 4),
                    'gear_rad': round(q('gear'), 4)}
    return out


rep = {'servo_roles': {}, 'checks': {}}
a = 0.3
roles = {'gear servo +0.3 (pivot holds)': {'gear': a}, 'pivot servo +0.3 (gear holds)': {'pivot': a},
         'pendulum: pivot +0.3, gear -0.3 N': {'pivot': a, 'gear': -N * a}}
for name, cmd in roles.items():
    rep['servo_roles'][name] = describe(cmd)
g, p, s = (rep['servo_roles'][k] for k in roles)
ok = rep['checks']
ok['gear servo: gear turns 1/N reversed, shin turns with the crank, femur stays'] = all(
    abs(g[l]['gear_rad'] + a / N) < 0.01 and abs(g[l]['shin_rad'] + a / N) < 0.01 and abs(g[l]['femur_rad']) < 0.01 for l in LEGS)
ok['pivot servo: femur turns, shin keeps its angle (knee bends)'] = all(
    abs(p[l]['femur_rad'] - a) < 0.01 and abs(p[l]['shin_rad']) < 0.01 and abs(p[l]['knee_rad'] + a) < 0.01 for l in LEGS)
ok['pivot +a with gear -a: rigid pendulum swing (knee fixed, hip-foot length kept)'] = all(
    abs(s[l]['knee_rad']) < 0.01 and abs(s[l]['extension_m']) < 5e-4 and abs(s[l]['swing_rad'] - a) < 0.01 for l in LEGS)

# equivalence: exact mechanism vs reduced tree, random servo targets (held body), then free standing height
rd = load('fox_reduced.xml', True)
dr = mujoco.MjData(rd)
mujoco.mj_forward(rd, dr)
rvid = {leg: foot_point(rd, dr, leg)[1] for leg in LEGS}
rng, worst = np.random.default_rng(1), 0.0
for trial in range(6):
    cmd = {}
    u_ex, u_rd = np.zeros(ex.nu), np.zeros(rd.nu)
    for leg in LEGS:
        for k, lim in (('hip', 0.3), ('pivot', 0.6), ('gear', 0.6)):
            v = rng.uniform(-lim, lim)
            u_ex[ex.actuator(leg + '_' + k).id] = v
            u_rd[rd.actuator(leg + '_' + k).id] = v
    mujoco.mj_resetData(ex, dx)
    mujoco.mj_resetData(rd, dr)
    settle(ex, dx, u_ex)
    settle(rd, dr, u_rd)
    worst = max(worst, max(float(np.linalg.norm(foot_point(ex, dx, l, vid[l])[0] - foot_point(rd, dr, l, rvid[l])[0])) for l in LEGS))
rep['reduced_vs_exact_max_foot_error_m'] = worst
stand = {}
for name in ('scene.xml', 'scene_reduced.xml'):
    m = load(name, False)
    d = mujoco.MjData(m)
    settle(m, d, np.zeros(m.nu), 3.0)
    stand[name] = round(float(d.qpos[2]), 5)
rep['standing_base_z_m'] = stand
ok['reduced tree matches the exact mechanism (feet < 1 mm, standing height < 1 mm)'] = worst < 1e-3 and abs(stand['scene.xml'] - stand['scene_reduced.xml']) < 1e-3

# picture: left side, rows = the three servo roles, columns = -0.4 / 0 / +0.4
r = mujoco.Renderer(ex, 420, 420)
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[2] = 1
cam = mujoco.MjvCamera()
cam.lookat[:] = (0.0, 0.03, 0.09)
cam.azimuth, cam.elevation, cam.distance = -90, -8, 0.42
rows = []
for name, cmd in roles.items():
    row = []
    for v in (-0.4, 0.0, 0.4):
        mujoco.mj_resetData(ex, dx)
        settle(ex, dx, servo_u(ex, {k: x / a * v for k, x in cmd.items()}), 1.0)
        r.update_scene(dx, cam, opt)
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text((8, 8), '%s   %+.1f rad' % (name.split(' (')[0].split(':')[0].replace(' +0.3', ''), v), fill=(255, 255, 255))
        row.append(np.asarray(im))
    rows.append(np.concatenate(row, 1))
Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(ROOT, 'validation', 'mujoco_servo_roles.png'))
json.dump(rep, open(os.path.join(ROOT, 'validation', 'mechanism_report.json'), 'w'), indent=1)
for name, res in rep['servo_roles'].items():
    print('%-34s FL %s' % (name, res['FL']))
    print('%-34s RL %s' % ('', res['RL']))
print('reduced vs exact: max foot error %.2e m | standing base z %s' % (worst, stand))
for k, v in ok.items():
    print('%-80s %s' % (k, 'PASS' if v else 'FAIL'))
raise SystemExit(0 if all(ok.values()) else 1)
