#!/usr/bin/env python3
"""Check the code the Raspberry Pi runs (deploy/fox_pi.py: FoxPolicy + policy.onnx) against Isaac Lab, closed loop:
the sim supplies what the Pi would read (gyro, gravity direction, the remote's command), FoxPolicy builds the
observation and runs the ONNX file (onnx's reference evaluator here, onnxruntime on the Pi), and its outputs drive the
simulated robot. FoxPolicy takes its scale / offset from the policy.json next to policy.onnx, as on the Pi. Passes when
FoxPolicy's observation equals Isaac Lab's, its actions equal the PyTorch policy's, its servo targets equal the env's
processed actions, the contract's servo order and gear ratio match the env, and driven by it the robot walks forward
(0.2 m/s commanded) and turns (0.6 rad/s), or for a straight task walks backward (-0.2 m/s), at least half as fast as
commanded (how accurately is fox_play.py --eval's job).

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/check_policy_io.py policies/fox_flat_blind_v3/model_1999.pt
    ... --task Fox-V7-Velocity-Flat-Blind-Straight-Play policies/<v7 policy>/model_*.pt
(add --device cpu when the GPU is busy: one robot runs fine on CPU PhysX)
"""
import argparse
import importlib.metadata as metadata
import os
import sys

from isaaclab.app import AppLauncher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = 150                                                  # 3 s per command
p = argparse.ArgumentParser()
p.add_argument("checkpoint", help="model_*.pt; policy.onnx (+ policy.json) is read from the same folder")
p.add_argument("--task", default="Fox-Velocity-Flat-Blind-Play", help="the -Play task the policy was trained for")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
TASK = args.task
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
from fox_pi import SERVOS, FoxPolicy, hips_free, onnx_infer, policy_contract  # noqa: E402

env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cmd_cfg = env_cfg.commands.base_velocity
straight = cmd_cfg.ranges.ang_vel_z == (0.0, 0.0) and cmd_cfg.ranges.lin_vel_y == (0.0, 0.0)
COMMANDS = ((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0)) if straight else ((0.2, 0.0, 0.0), (0.0, 0.0, 0.6))
cmd_cfg.resampling_time_range, cmd_cfg.rel_standing_envs = (1.0e9, 1.0e9), 0.0
# start on COMMANDS[0], so FoxPolicy and the sim's FoxServoAction apply the hip rule to the same command from step 0. (A
# command changed between steps is gated again by the sim's action term; COMMANDS go straight -> turning, where that
# is a no-op. The Pi gates once, with the command it read.)
cmd_cfg.ranges.lin_vel_x, cmd_cfg.ranges.lin_vel_y, cmd_cfg.ranges.ang_vel_z = ((v, v) for v in COMMANDS[0])
agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
agent_cfg.device = args.device
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
onnx_path = os.path.join(os.path.dirname(args.checkpoint), "policy.onnx")
contract = policy_contract(onnx_path)
pi = FoxPolicy(onnx_infer(onnx_path), contract["scale"], contract["offset_rad"])

u = env.unwrapped
robot, cmd_term, act_term = u.scene["robot"], u.command_manager.get_term("base_velocity"), u.action_manager.get_term("joint_pos")
joint_of = {"hip": "hip", "pivot": "thigh", "gear": "calf"}
order_ok = list(act_term._joint_names) == ["%s_%s_joint" % (n[:2], joint_of[n[3:]]) for n in SERVOS]
contract_ok = list(contract["servos"]) == SERVOS and float(contract["gear_ratio"]) == float(env_cfg.scene.robot.actuators["servos"].gear_ratio)


def readings():
    """What the Pi reads each step, here from the sim: gyro and gravity direction (body frame), the remote's command."""
    return [x[0].detach().cpu().numpy() for x in (robot.data.root_ang_vel_b.torch, robot.data.projected_gravity_b.torch,
                                                  cmd_term.vel_command_b)]


obs = env.get_observations()
targets = pi.reset(*readings())
obs_err, act_err, tgt_err, vel = 0.0, 0.0, 0.0, []
for k in range(STEPS * len(COMMANDS)):
    with torch.inference_mode():
        obs_err = max(obs_err, float(np.abs(obs["policy"][0].cpu().numpy() - pi.obs[0]).max()))
        a_torch = policy(obs)[0].cpu().numpy()
        a_torch[:4] *= hips_free(cmd_term.vel_command_b[0].cpu().numpy())                       # hips held when straight
        act_err = max(act_err, float(np.abs(a_torch - pi.action).max()))
        cmd_term.vel_command_b[:] = torch.tensor(COMMANDS[k // STEPS], device=u.device)          # the remote
        obs, _, _, _ = env.step(torch.tensor(pi.action[None], device=u.device))                 # the Pi code drives
        tgt_err = max(tgt_err, float(np.abs(act_term.processed_actions[0].cpu().numpy() - targets).max()))   # = servo targets
    targets = pi.step(*readings())
    if k % STEPS >= STEPS // 2:
        vel.append([float(robot.data.root_lin_vel_b.torch[0, 0]), float(robot.data.root_ang_vel_b.torch[0, 2])])
vel = np.array(vel).reshape(len(COMMANDS), -1, 2).mean(1)
if straight:                                                                     # it walks; accuracy: fox_play --eval
    walks = vel[0, 0] > 0.5 * COMMANDS[0][0] and vel[1, 0] < 0.5 * COMMANDS[1][0]
else:
    walks = vel[0, 0] > 0.5 * COMMANDS[0][0] and vel[1, 1] > 0.5 * COMMANDS[1][2]

print("CHECK observation: %d floats, max |Isaac Lab - FoxPolicy| = %.2e" % (pi.obs.size, obs_err))
print("CHECK FoxPolicy (ONNX) vs PyTorch policy: max |action difference| = %.2e" % act_err)
print("CHECK FoxPolicy servo targets (policy.json scale / offset) vs the env's processed actions: max |difference| = %.2e" % tgt_err)
print("CHECK output order = env servo joints (%s ... %s): %s; policy.json servo order and gear ratio %g: %s"
      % (SERVOS[0], SERVOS[-1], order_ok, float(contract["gear_ratio"]), contract_ok))
if straight:
    print("CHECK closed loop on FoxPolicy: forward %.3f m/s (cmd 0.2), backward %.3f m/s (cmd -0.2)" % (vel[0, 0], vel[1, 0]))
else:
    print("CHECK closed loop on FoxPolicy: forward %.3f m/s (cmd 0.2), turn %.3f rad/s (cmd 0.6)" % (vel[0, 0], vel[1, 1]))
ok = pi.obs.size == 105 and obs_err < 1e-5 and act_err < 1e-4 and tgt_err < 1e-5 and order_ok and contract_ok and walks
print("CHECK", "PASS" if ok else "FAIL")
env.close()
app.close()
