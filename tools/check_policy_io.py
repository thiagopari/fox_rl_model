#!/usr/bin/env python3
"""Check a blind Fox policy's hardware interface, i.e. what the Raspberry Pi must do around policy.onnx:
  * rebuild the observation from raw signals the way the Pi has to and compare it with Isaac Lab's;
  * run the exported ONNX on that vector (onnx's reference evaluator, no onnxruntime needed) against the PyTorch policy;
  * print the observation layout and the action -> servo order.
Observation (105 floats, no normalization): per term the last 5 readings, oldest first, terms in this order
    [0:15]   gyro, body frame x fwd / y left / z up (rad/s)
    [15:30]  gravity direction in the body frame, unit vector ((0, 0, -1) standing level)
    [30:45]  command (vx m/s, vy m/s, yaw rate rad/s)
    [45:105] the policy's own previous output (12)
At start the history holds the first reading 5 times. Servo targets = 0.25 * output (rad from the CAD stance).

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/check_policy_io.py policies/fox_flat_blind_v2/model_1499.pt
"""
import argparse
import collections
import importlib.metadata as metadata
import os
import sys

from isaaclab.app import AppLauncher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK, H = "Fox-Velocity-Flat-Blind-Play", 5
p = argparse.ArgumentParser()
p.add_argument("checkpoint", help="model_*.pt; policy.onnx is read from the same folder")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "isaaclab"))
import fox_tasks  # noqa: E402,F401

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
model = onnx.load(os.path.join(os.path.dirname(args.checkpoint), "policy.onnx"))
onnx_policy, onnx_in = ReferenceEvaluator(model), model.graph.input[0].name

u = env.unwrapped
robot, cmd_term, act_term = u.scene["robot"], u.command_manager.get_term("base_velocity"), u.action_manager.get_term("joint_pos")
hist = collections.defaultdict(lambda: collections.deque(maxlen=H))


def raw():
    """What the Pi reads each step: IMU gyro + gravity (body frame), the remote's command, its own last output."""
    return [robot.data.root_ang_vel_b.torch[0], robot.data.projected_gravity_b.torch[0], cmd_term.vel_command_b[0],
            u.action_manager.action[0]]


def pi_observation(first=False):
    for k, x in enumerate(raw()):
        for _ in range(H if first else 1):
            hist[k].append(x.detach().cpu().numpy().astype(np.float32))
    return np.concatenate([np.concatenate(list(hist[k])) for k in range(4)])


obs = env.get_observations()
ours, obs_err, act_err = pi_observation(first=True), 0.0, 0.0
for k in range(150):
    with torch.inference_mode():
        obs_err = max(obs_err, float(np.abs(obs["policy"][0].cpu().numpy() - ours).max()))
        a = policy(obs)
        a_onnx = onnx_policy.run(None, {onnx_in: ours[None]})[0][0]
        act_err = max(act_err, float(np.abs(a[0].cpu().numpy() - a_onnx).max()))
        cmd_term.vel_command_b[:] = torch.tensor([0.2, 0.0, 0.4] if k < 75 else [0.0, 0.15, -0.5], device=u.device)  # remote
        obs, _, _, _ = env.step(a)
    ours = pi_observation()

print("CHECK observation: %d floats, max |Isaac Lab - rebuilt| = %.2e" % (ours.size, obs_err))
print("CHECK ONNX on the rebuilt observation vs PyTorch policy: max |action difference| = %.2e" % act_err)
servo = {"hip": "hip servo", "thigh": "pivot servo", "calf": "gear servo"}
print("CHECK action k -> servo (target = %.2f * action, rad from the CAD stance):" % act_term.cfg.scale)
print("      " + ", ".join("%d %s %s" % (k, n[:2], servo[n.split("_")[1]]) for k, n in enumerate(act_term._joint_names)))
ok = ours.size == 105 and obs_err < 1e-5 and act_err < 1e-4
print("CHECK", "PASS" if ok else "FAIL")
env.close()
app.close()
