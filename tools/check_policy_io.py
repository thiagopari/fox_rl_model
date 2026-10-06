#!/usr/bin/env python3
"""Check the code the Raspberry Pi runs (deploy/fox_pi.py: FoxPolicy + policy.onnx) against Isaac Lab, closed loop:
the sim supplies what the Pi would read (gyro, gravity direction, the remote's command), FoxPolicy builds the
observation and runs the ONNX file (onnx's reference evaluator here, onnxruntime on the Pi), and its outputs drive the
simulated robot. Passes when FoxPolicy's observation equals Isaac Lab's, its actions equal the PyTorch policy's, its
output order matches the env's servo joints, and the robot follows 0.2 m/s forward, then a 0.6 rad/s turn.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/check_policy_io.py policies/fox_flat_blind_v3/model_1999.pt
(add --device cpu when the GPU is busy: one robot runs fine on CPU PhysX)
"""
import argparse
import importlib.metadata as metadata
import os
import sys

from isaaclab.app import AppLauncher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "Fox-Velocity-Flat-Blind-Play"
COMMANDS, STEPS = ((0.2, 0.0, 0.0), (0.0, 0.0, 0.6)), 150   # 3 s each
p = argparse.ArgumentParser()
p.add_argument("checkpoint", help="model_*.pt; policy.onnx is read from the same folder")
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

sys.path[:0] = [os.path.join(ROOT, "isaaclab"), os.path.join(ROOT, "deploy")]
import fox_tasks  # noqa: E402,F401
from fox_pi import SERVOS, FoxPolicy, onnx_infer  # noqa: E402

env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
agent_cfg.device = args.device
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
pi = FoxPolicy(onnx_infer(os.path.join(os.path.dirname(args.checkpoint), "policy.onnx")))

u = env.unwrapped
robot, cmd_term, act_term = u.scene["robot"], u.command_manager.get_term("base_velocity"), u.action_manager.get_term("joint_pos")
joint_of = {"hip": "hip", "pivot": "thigh", "gear": "calf"}
order_ok = list(act_term._joint_names) == ["%s_%s_joint" % (n[:2], joint_of[n[3:]]) for n in SERVOS]


def readings():
    """What the Pi reads each step, here from the sim: gyro and gravity direction (body frame), the remote's command."""
    return [x[0].detach().cpu().numpy() for x in (robot.data.root_ang_vel_b.torch, robot.data.projected_gravity_b.torch,
                                                  cmd_term.vel_command_b)]


obs = env.get_observations()
pi.reset(*readings())
obs_err, act_err, vel = 0.0, 0.0, []
for k in range(STEPS * len(COMMANDS)):
    with torch.inference_mode():
        obs_err = max(obs_err, float(np.abs(obs["policy"][0].cpu().numpy() - pi.obs[0]).max()))
        act_err = max(act_err, float(np.abs(policy(obs)[0].cpu().numpy() - pi.action).max()))
        cmd_term.vel_command_b[:] = torch.tensor(COMMANDS[k // STEPS], device=u.device)          # the remote
        obs, _, _, _ = env.step(torch.tensor(pi.action[None], device=u.device))                 # the Pi code drives
    pi.step(*readings())
    if k % STEPS >= STEPS // 2:
        vel.append([float(robot.data.root_lin_vel_b.torch[0, 0]), float(robot.data.root_ang_vel_b.torch[0, 2])])
vel = np.array(vel).reshape(len(COMMANDS), -1, 2).mean(1)
walks = abs(vel[0, 0] - COMMANDS[0][0]) < 0.04 and abs(vel[1, 1] - COMMANDS[1][2]) < 0.12

print("CHECK observation: %d floats, max |Isaac Lab - FoxPolicy| = %.2e" % (pi.obs.size, obs_err))
print("CHECK FoxPolicy (ONNX) vs PyTorch policy: max |action difference| = %.2e" % act_err)
print("CHECK output order = env servo joints (%s ... %s): %s" % (SERVOS[0], SERVOS[-1], order_ok))
print("CHECK closed loop on FoxPolicy: forward %.3f m/s (cmd 0.2), turn %.3f rad/s (cmd 0.6)" % (vel[0, 0], vel[1, 1]))
ok = pi.obs.size == 105 and obs_err < 1e-5 and act_err < 1e-4 and order_ok and walks
print("CHECK", "PASS" if ok else "FAIL")
env.close()
app.close()
