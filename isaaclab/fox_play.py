"""Drive the trained Fox walking policy with the keyboard in Isaac Lab (Kit viewer).

    cd ~/Documents/fox_rl_model && ~/isaacenv/bin/python isaaclab/fox_play.py [--checkpoint path/to/model_999.pt]

Click into the viewport first (the window needs keyboard focus), then hold:
    W / S  or  Up / Down      forward / backward   (0.3 m/s)
    A / D  or  Left / Right   sideways             (0.2 m/s)
    Q / E  or  Z / X          turn left / right    (1 rad/s)
    L                         stop (zero command)
Keys add up while held. The camera follows robot 0; green/blue arrows show commanded vs actual velocity.
Check without a keyboard: add --eval --headless --num_envs 16 (scripted commands, prints achieved velocities).
Default checkpoint: the newest model_*.pt under logs/rsl_rl/fox_flat. Also writes exported/policy.onnx next to it
(the file the Raspberry Pi would run).
"""
import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Keyboard teleop of the Fox walking policy.")
parser.add_argument("--checkpoint", default=None, help="model_*.pt (default: newest in logs/rsl_rl/fox_flat)")
parser.add_argument("--num_envs", type=int, default=1, help="robots in the scene (robot 0 is the one the camera follows)")
parser.add_argument("--eval", action="store_true", help="no keyboard: run scripted commands and print commanded vs achieved velocity")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless and not args.visualizer:
    args.visualizer = ["kit"]                       # Isaac Lab 3.0 stays headless without a visualizer
if not args.kit_args:                               # small window: full screen on the 3200x2000 panel renders slowly
    args.kit_args = "--/app/window/width=1280 --/app/window/height=800"
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
import importlib.metadata as metadata  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, HERE)
import fox_tasks  # noqa: E402,F401  (registers the Fox tasks)

TASK = "Fox-Velocity-Flat-Play"


class FoxKeyboard(Se2Keyboard):
    """Isaac Lab's SE(2) keyboard (arrows, Z/X, L) plus W/A/S/D and Q/E."""

    def _create_key_bindings(self):
        super()._create_key_bindings()
        m = self._INPUT_KEY_MAPPING
        m.update({"W": m["UP"], "S": m["DOWN"], "A": m["LEFT"], "D": m["RIGHT"], "Q": m["Z"], "E": m["X"]})


def newest_checkpoint() -> str:
    runs = sorted(glob.glob(os.path.join(os.getcwd(), "logs", "rsl_rl", "fox_flat", "*", "model_*.pt")), key=os.path.getmtime)
    if not runs:
        raise SystemExit("No checkpoint found under logs/rsl_rl/fox_flat - train first (isaaclab/fox_train.py) or pass --checkpoint")
    return runs[-1]


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range, cmd.rel_standing_envs, cmd.debug_vis = (1.0e9, 1.0e9), 0.0, True   # the keyboard owns it
    env_cfg.viewer.origin_type, env_cfg.viewer.asset_name = "asset_root", "robot"
    env_cfg.viewer.eye, env_cfg.viewer.lookat = (-0.55, -0.55, 0.35), (0.0, 0.0, 0.0)
    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    path = args.checkpoint or newest_checkpoint()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        runner.export_policy_to_onnx(path=os.path.join(os.path.dirname(path), "exported"), filename="policy.onnx")
    except Exception as e:  # export is a convenience for the hardware step; driving does not need it
        print("[WARN] ONNX export skipped:", e)
    term = env.unwrapped.command_manager.get_term("base_velocity")
    if args.eval:
        return evaluate(env, policy, term)
    keyboard = FoxKeyboard(Se2KeyboardCfg(v_x_sensitivity=0.3, v_y_sensitivity=0.2, omega_z_sensitivity=1.0, sim_device=env.unwrapped.device))
    print("[INFO] Driving %s with %s. Click the viewport, then W/A/S/D, Q/E (L = stop)." % (TASK, path))
    obs = env.get_observations()
    last = None
    while simulation_app.is_running():
        with torch.inference_mode():
            command = keyboard.advance()
            term.vel_command_b[:] = command                 # read by the policy's velocity_commands observation
            obs, _, _, _ = env.step(policy(obs))
        now = tuple(np.round(command.cpu().numpy(), 2))
        if now != last:
            print("[INFO] command vx %.2f m/s  vy %.2f m/s  yaw %.2f rad/s" % now, flush=True)
            last = now
    env.close()


def evaluate(env, policy, term):
    """Hold each command 4 s; report the mean achieved base velocity (body frame) over the last 2 s, all robots."""
    robot, obs = env.unwrapped.scene["robot"], env.get_observations()
    for c in ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.8), (0.2, 0.0, 0.5)):
        acc, falls = [], 0
        for k in range(200):
            with torch.inference_mode():
                term.vel_command_b[:] = torch.tensor(c, device=env.unwrapped.device)
                obs, _, dones, _ = env.step(policy(obs))
            falls += int(dones.sum())
            if k >= 100:
                acc.append(torch.cat([robot.data.root_lin_vel_b.torch[:, :2], robot.data.root_ang_vel_b.torch[:, 2:]], 1).mean(0))
        got = torch.stack(acc).mean(0).cpu().numpy()
        print("[EVAL] command vx %+.2f vy %+.2f yaw %+.2f  ->  achieved vx %+.3f vy %+.3f yaw %+.3f  (falls %d)" % (*c, *got, falls), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
