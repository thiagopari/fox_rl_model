#!/usr/bin/env python3
"""Servo chatter check in Isaac Lab: 16 robots hold the stance (zero actions) for 2 s while every 200 Hz physics step's
calf torque (the gear servo, -N t_gear) is recorded. FoxServoActuator's damping is explicit, so it is stable only while
damping x dt / inertia stays small: a ratio change without its reflected inertia (fox_cfg.geared) made the 2:1 gear
servo chatter at 100 Hz (stall torque, sign flips on 25 % of steps). Passes when < 5 % of steps flip sign above
0.05 N m and the mean step-to-step change is < 0.1 N m.
    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/check_servo_chatter.py --task Fox-V7-Velocity-Flat-Blind-Straight-Play
"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "isaaclab")]
import fox_cfg  # noqa: E402
import fox_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

log = []
orig = fox_cfg.FoxServoActuator.compute


def compute(self, control_action, joint_pos, joint_vel):
    out = orig(self, control_action, joint_pos, joint_vel)
    log.append(out.joint_efforts[:, self._calf].clone().cpu().numpy())   # = -N t_gear per leg
    return out


fox_cfg.FoxServoActuator.compute = compute
cfg = parse_env_cfg(args.task, device=args.device, num_envs=16)
cfg.commands.base_velocity.rel_standing_envs = 1.0
env = gym.make(args.task, cfg=cfg)
env.reset()
u = env.unwrapped
zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for _ in range(100):
    with torch.inference_mode():
        env.step(zero)
e = np.array(log[len(log) // 2:])              # (steps, envs, legs): second half, settled
flip = (e[1:] * e[:-1] < 0) & (np.abs(e[1:]) > 0.05) & (np.abs(e[:-1]) > 0.05)
step = np.abs(np.diff(e, axis=0)).mean()
ok = flip.mean() < 0.05 and step < 0.1
print("CHECK %s: calf torque |mean| %.3f N m, mean step-to-step change %.3f N m, sign flips > 0.05 N m on %.1f %% of steps, "
      "max |torque| %.3f N m - %s" % (args.task, np.abs(e).mean(), step, 100 * flip.mean(), np.abs(e).max(), "PASS" if ok else "FAIL"), flush=True)
env.close()
app.close()
