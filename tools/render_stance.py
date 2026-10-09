#!/usr/bin/env python3
"""Side views (robot's right side, front on the right) of the v7 model at home (the CAD pose) and at the policy's
starting pose (the v7 stance), feet on the floor, with leg angles and the servo degrees of a servo_calib.json.
    scp fox-wifi:fox/servo_calib.json /tmp/ && MUJOCO_GL=egl ~/.venvs/fox_rl/bin/python tools/render_stance.py . /tmp/servo_calib.json v7/validation/policy_start_pose.png
"""
import json, math, os, sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT, CALIB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'v7', 'mjcf', 'scene_reduced.xml'))
d = mujoco.MjData(m)
mech = json.load(open(os.path.join(ROOT, 'v7', 'mechanism.json')))
cal = json.load(open(CALIB))
adr = lambda j: m.jnt_qposadr[m.joint(j).id]                        # noqa: E731
LEGS = ('FL', 'FR', 'RL', 'RR')
feet = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.endswith('_foot') and m.geom_contype[g]]


def set_pose(q):
    d.qpos[:] = m.qpos0
    d.qpos[3:7] = [1, 0, 0, 0]
    for j, v in q.items():
        d.qpos[adr(j)] = v
    d.qpos[2] = 0.3
    mujoco.mj_forward(m, d)
    low = min((d.geom_xpos[g] + m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
               @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min() for g in feet)
    d.qpos[2] -= low                                                  # lowest foot point on the floor
    mujoco.mj_forward(m, d)


def angles(leg):                                                      # from straight down, + = toward the front
    H, K, A = (d.xanchor[m.joint('%s_%s_joint' % (leg, p)).id] for p in ('thigh', 'calf', 'foot'))
    f = math.degrees(math.atan2(K[0] - H[0], -(K[2] - H[2])))
    s = math.degrees(math.atan2(A[0] - K[0], -(A[2] - K[2])))
    return f, s


def servo_deg(n, deg_from_home):
    return cal['home_deg'][n] + cal['direction'][n] * deg_from_home


rend = mujoco.Renderer(m, 760, 1000)
cam = mujoco.MjvCamera()
cam.type, cam.azimuth, cam.elevation, cam.distance = mujoco.mjtCamera.mjCAMERA_FREE, 90.0, -8.0, 0.52
try:
    font, small = ImageFont.load_default(size=26), ImageFont.load_default(size=21)
except TypeError:
    font = small = ImageFont.load_default()
panels = []
for title, q, from_cad in (('HOME = CAD pose (servos at their homes)', {}, {}),
                           ('POLICY START = v7 stance', mech['stance']['joints_rad'], mech['stance']['servo_deg_from_cad'])):
    set_pose(q)
    cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.085]
    rend.update_scene(d, camera=cam)
    img = Image.fromarray(rend.render())
    canvas = Image.new('RGB', (img.width, img.height + 330), (250, 250, 250))
    canvas.paste(img, (0, 70))
    dr = ImageDraw.Draw(canvas)
    dr.text((18, 18), title, fill=(20, 20, 20), font=font)
    y = img.height + 85
    for leg in ('FR', 'RR'):
        f, s = angles(leg)
        dr.text((18, y), '%s leg: femur %+.1f deg (%s), shin %+.1f deg from straight down' % (
            'front' if leg[0] == 'F' else 'rear', f, 'knee forward' if f > 0 else 'knee back', s), fill=(20, 20, 20), font=small)
        y += 34
    y += 10
    dr.text((18, y), 'servo deg with your servo_calib.json (pivot / gear):', fill=(60, 60, 60), font=small)
    y += 32
    for pair in (('FL', 'FR'), ('RL', 'RR')):
        dr.text((18, y), '   '.join('%s %.1f / %.1f' % (leg, servo_deg(leg + '_pivot', from_cad.get(leg + '_pivot', 0.0)),
                                                         servo_deg(leg + '_gear', from_cad.get(leg + '_gear', 0.0))) for leg in pair),
                fill=(20, 20, 20), font=small)
        y += 32
    dr.text((18, y + 6), 'front is on the right; legs in vertical planes (hips at home)', fill=(110, 110, 110), font=small)
    panels.append(canvas)
out = Image.new('RGB', (sum(p.width for p in panels) + 20, panels[0].height), (200, 200, 200))
x = 0
for p in panels:
    out.paste(p, (x, 0))
    x += p.width + 20
out.save(OUT)
print('wrote', OUT)
for name, q in (('home', {}), ('stance', mech['stance']['joints_rad'])):
    set_pose(q)
    print(name, {leg: tuple(round(a, 1) for a in angles(leg)) for leg in LEGS}, 'base z %.3f' % d.qpos[2])
