#!/usr/bin/env python3
"""Gait quality of a trained Fox policy in Isaac Lab, beyond tracking: is the walk natural? 16 robots per command, the
last 2 s of 4 s, on the task's -Play variant (realistic servos, no pushes). Per command it prints:
    track     achieved forward speed vs commanded
    rhythm    touchdowns per foot per second, swing and stance time, duty factor (a trot: 0.5-0.65)
    lift      mean peak foot rise per swing (mm; a dragging foot is < ~10 mm)
    slip      foot speed while in contact (mm/s; planted feet ~0)
    trot      contact correlation of the diagonal pairs FL-RR, FR-RL (+1) and of the front pair FL-FR (-1)
    symmetry  left vs right legs: difference of mean joint angle and of swing amplitude (deg; 0 = mirror images)
    body      pitch / roll std (deg), height mean and std (mm), smoothness (servo-target change, rad/step)
and saves robot 0's motion (--record npz) for tools/replay_gif.py.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/gait_report.py --task Fox-V7-Velocity-Flat-Blind-Home-Play \\
        --checkpoint policies/fox_v7_home_v1/model_1999.pt [--record /tmp/run.npz]
"""
import argparse
import importlib.metadata as metadata
import os
import sys

from isaaclab.app import AppLauncher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--record", default=None, help="npz of robot 0's motion over all commands")
p.add_argument("--commands", default="0,0.05,0.1,0.2,-0.1", help="forward speeds, m/s")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "isaaclab"))
import fox_tasks  # noqa: E402,F401

cfg = parse_env_cfg(args.task, device=args.device, num_envs=16)
c = cfg.commands.base_velocity
c.resampling_time_range, c.rel_standing_envs = (1.0e9, 1.0e9), 0.0
agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
robot, cs, term, act = u.scene["robot"], u.scene["contact_forces"], u.command_manager.get_term("base_velocity"), u.action_manager.get_term("joint_pos")
feet_s, _ = cs.find_bodies(".*_foot")
feet_b = robot.find_bodies(".*_foot")[0]
fnames = [robot.body_names[i][:2] for i in feet_b]
pitch_j = [robot.joint_names.index("%s_%s_joint" % (leg, j)) for j in ("thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
obs, rec = env.get_observations(), {"pos": [], "quat": [], "joints": [], "cmd": [], "contact": []}
idx = {n: fnames.index(n) for n in ("FL", "FR", "RL", "RR")}
print("%s  %s" % (args.task, args.checkpoint))
for vx in (float(v) for v in args.commands.split(",")):
    down = cs.data.current_contact_time.torch[:, feet_s] > 0
    C, Z, S, J, PR, H, DA, swings, stances, lifts = [], [], [], [], [], [], [], [], [], []
    peak = torch.zeros(u.num_envs, len(feet_b), device=u.device)
    prev_a = u.action_manager.action.clone()
    for k in range(200):
        with torch.inference_mode():
            term.vel_command_b[:] = torch.tensor((vx, 0.0, 0.0), device=u.device)
            obs, _, _, _ = env.step(policy(obs))
        now = cs.data.current_contact_time.torch[:, feet_s] > 0
        fz = robot.data.body_pos_w.torch[:, feet_b, 2]
        peak = torch.where(now, torch.zeros_like(peak), torch.maximum(peak, fz))
        if k >= 100:
            td, lo = now & ~down, ~now & down
            swings += cs.data.last_air_time.torch[:, feet_s][td].tolist()
            stances += cs.data.last_contact_time.torch[:, feet_s][lo].tolist()
            C.append(now.float().cpu().numpy())
            Z.append(robot.data.root_pos_w.torch[:, 2].cpu().numpy())
            v = robot.data.body_lin_vel_w.torch[:, feet_b, :2].norm(dim=-1)
            S.append(v[now].mean().item() if now.any() else 0.0)
            J.append(robot.data.joint_pos.torch[:, pitch_j].cpu().numpy())
            g = robot.data.projected_gravity_b.torch
            PR.append(np.degrees(np.stack([torch.asin(g[:, 0].clamp(-1, 1)).cpu().numpy(), torch.asin(g[:, 1].clamp(-1, 1)).cpu().numpy()], 1)))
            H.append(robot.data.root_lin_vel_b.torch[:, 0].cpu().numpy())
            DA.append(float(((u.action_manager.action - prev_a) * act._scale).abs().mean()))
            if td.any():
                lifts += (peak_prev[td] - fz[td]).tolist()
        prev_a = u.action_manager.action.clone()
        peak_prev = peak.clone()
        down = now
        rec["pos"].append(robot.data.root_pos_w.torch[0].cpu().numpy() - u.scene.env_origins[0].cpu().numpy())
        rec["quat"].append(robot.data.root_quat_w.torch[0].cpu().numpy())
        rec["joints"].append(robot.data.joint_pos.torch[0].cpu().numpy())
        rec["cmd"].append(vx)
        rec["contact"].append(now[0].cpu().numpy())                     # robot 0's feet, in fnames order
    C, J, PR = np.array(C), np.degrees(np.array(J)), np.array(PR)
    T = 2.0
    corr = lambda a, b: float(np.nanmean([np.corrcoef(C[:, e, idx[a]], C[:, e, idx[b]])[0, 1] for e in range(u.num_envs)  # noqa: E731
                                          if C[:, e, idx[a]].std() > 0 and C[:, e, idx[b]].std() > 0]) if C.std() > 0 else float("nan"))
    sw, st = (np.mean(x) if x else 0.0 for x in (swings, stances))
    sym = []
    for i in (0, 2):                                   # front pair FL/FR, rear pair RL/RR: thigh (0-3) and calf (4-7)
        for jo in (0, 4):
            a, b = J[:, :, jo + i], J[:, :, jo + i + 1]
            sym.append((abs(a.mean() - b.mean()), abs(np.ptp(a, 0).mean() - np.ptp(b, 0).mean())))
    print("cmd %+.2f | track %+.3f m/s | rhythm %.1f steps/foot/s, swing %.3f s, stance %.3f s, duty %.2f | lift %.0f mm | slip %.0f mm/s"
          % (vx, np.mean(H), len(swings) / (u.num_envs * len(feet_b) * T), sw, st, st / (sw + st) if sw + st > 0 else float("nan"),
             1000 * np.mean(lifts) if lifts else 0.0, 1000 * np.mean(S)))
    print("          | trot corr FL-RR %+.2f FR-RL %+.2f FL-FR %+.2f | symmetry mean / amplitude diff deg: front thigh %.0f/%.0f calf %.0f/%.0f"
          "  rear thigh %.0f/%.0f calf %.0f/%.0f" % (corr("FL", "RR"), corr("FR", "RL"), corr("FL", "FR"), *[x for s in sym for x in s]))
    print("          | body pitch std %.1f roll std %.1f deg, height %.0f +- %.1f mm | target change %.3f rad/step"
          % (PR[:, :, 0].std(0).mean(), PR[:, :, 1].std(0).mean(), 1000 * np.mean(Z), 1000 * np.std(Z, 0).mean(), np.mean(DA)))
if args.record:
    np.savez(args.record, pos=np.array(rec["pos"]), quat=np.array(rec["quat"]), joints=np.array(rec["joints"]),
             names=np.array(robot.joint_names), cmd=np.array(rec["cmd"]), dt=u.step_dt,
             contact=np.array(rec["contact"]), contact_names=np.array([robot.body_names[i][:2] for i in feet_b]))
    print("recorded robot 0 ->", args.record)
env.close()
app.close()
