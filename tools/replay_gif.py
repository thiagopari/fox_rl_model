#!/usr/bin/env python3
"""Replay a run recorded by tools/gait_report.py --record in the identical MuJoCo model (software rendering, so it works
without the GPU) as an animated GIF, a strip of frames, and a foot-contact timeline (a trot: diagonal pairs together).
    MUJOCO_GL=osmesa ~/.venvs/fox_rl/bin/python tools/replay_gif.py run.npz out_prefix [--strip-at 9.0] [--every 2]
writes out_prefix.gif, out_prefix_strip.png (12 frames, 0.04 s apart, from --strip-at s) and out_prefix_contacts.png.
"""
import argparse
import os

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--strip-at", type=float, default=None, help="start of the frame strip (s); default the middle")
ap.add_argument("--every", type=int, default=2, help="GIF: every n-th recorded step")
a = ap.parse_args()
rec = np.load(a.run)
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "v7", "mjcf", "scene_reduced.xml"))
d = mujoco.MjData(m)
adr = [m.jnt_qposadr[m.joint(str(n)).id] for n in rec["names"]]
r = mujoco.Renderer(m, 360, 560)
cam = mujoco.MjvCamera()
cam.type, cam.azimuth, cam.elevation, cam.distance = mujoco.mjtCamera.mjCAMERA_FREE, 90.0, -10.0, 0.62
try:
    font = ImageFont.load_default(size=20)
except TypeError:
    font = ImageFont.load_default()
pos, dt = rec["pos"], float(rec["dt"])


def pose(i):
    d.qpos[:] = m.qpos0
    d.qpos[0:3], d.qpos[3:7] = pos[i], np.roll(rec["quat"][i], 1)   # Isaac Lab 3 quaternions are (x, y, z, w), MuJoCo's (w, x, y, z)
    for k, q in zip(adr, rec["joints"][i]):
        d.qpos[k] = q
    mujoco.mj_forward(m, d)


def frame(i, label=True):
    pose(i)
    cam.lookat[:] = [pos[i][0], pos[i][1], 0.08]
    r.update_scene(d, camera=cam)
    img = Image.fromarray(r.render())
    if label:
        j0, j1 = max(i - 25, 0), min(i + 25, len(pos) - 1)
        v = (pos[j1][0] - pos[j0][0]) / (dt * max(j1 - j0, 1))
        cmd = float(rec["cmd"][i])
        what = "stand" if cmd == 0 else ("forward" if cmd > 0 else "backward") + " %.2f m/s" % abs(cmd)
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, 559, 34], fill=(245, 245, 245))
        dr.text((10, 7), "t %4.1f s   command: %-16s   actual %+.2f m/s" % (i * dt, what, v), fill=(20, 20, 20), font=font)
    return img


frames = [frame(i).convert("P", palette=Image.ADAPTIVE, colors=128) for i in range(0, len(pos), a.every)]
frames[0].save(a.out + ".gif", save_all=True, append_images=frames[1:], duration=int(1000 * dt * a.every), loop=0, optimize=True)
s0 = int((a.strip_at if a.strip_at is not None else len(pos) * dt / 2) / dt)
strip = [frame(min(s0 + 2 * k, len(pos) - 1), label=False).crop((110, 60, 450, 330)) for k in range(12)]
sheet = Image.new("RGB", (6 * 340, 2 * 270), (255, 255, 255))
for k, im in enumerate(strip):
    sheet.paste(im, ((k % 6) * 340, (k // 6) * 270))
    ImageDraw.Draw(sheet).text(((k % 6) * 340 + 6, (k // 6) * 270 + 4), "+%.2f s" % (k * 2 * dt), fill=(0, 0, 0), font=font)
sheet.save(a.out + "_strip.png")
# foot contacts over the whole run: Isaac Lab's contact sensor (gait_report.py records it)
order = [str(n) for n in rec["contact_names"]]
contact = rec["contact"].astype(bool)
W, rowh = 1200, 40
img = Image.new("RGB", (W + 60, 4 * rowh + 40), (255, 255, 255))
dr = ImageDraw.Draw(img)
for k, leg in enumerate(["FL", "RR", "FR", "RL"]):                   # diagonal pairs next to each other
    dr.text((6, 20 + k * rowh + 8), leg, fill=(0, 0, 0), font=font)
    col = order.index(leg)
    for i in range(len(pos)):
        if contact[i, col]:
            x = 60 + int(i * W / len(pos))
            dr.line([(x, 20 + k * rowh + 4), (x, 20 + k * rowh + rowh - 6)], fill=(40, 90, 200) if k < 2 else (220, 110, 30))
for i in range(1, len(pos)):
    if rec["cmd"][i] != rec["cmd"][i - 1]:
        x = 60 + int(i * W / len(pos))
        dr.line([(x, 10), (x, 20 + 4 * rowh)], fill=(150, 150, 150))
dr.text((60, 4 * rowh + 22), "foot on the ground (bars) over %.0f s; grey lines = command changes" % (len(pos) * dt), fill=(80, 80, 80), font=font)
img.save(a.out + "_contacts.png")
print("wrote %s.gif (%d frames, %.1f MB), %s_strip.png, %s_contacts.png" % (a.out, len(frames), os.path.getsize(a.out + ".gif") / 1e6, a.out, a.out))
