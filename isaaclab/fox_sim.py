"""Launch the Fox quadruped in Isaac Lab (3.0-beta2, Isaac Sim 6.0.1) and drive its servos.

One robot per environment, each showing one servo motion (servo targets through FoxServoActuator, fox_cfg.py):
    env 0  crouch / rise          pivot + gear together (knee bends)
    env 1  sideways sway          hip servos
    env 2  pendulum swing         pivot +a with gear -a: legs swing rigidly, body moves fore/aft
    env 3  shin swing             gear servos alone
Run (GUI):       ~/isaacenv/bin/python ~/Documents/fox_rl_model/isaaclab/fox_sim.py --viz kit
Run (headless):  ~/isaacenv/bin/python ~/Documents/fox_rl_model/isaaclab/fox_sim.py --headless --duration 10
(Isaac Lab 3.0 runs headless unless a visualizer is requested with --viz.)
PhysX GPU buffers are sized for a few robots so it fits next to other GPU apps on an 8 GB card.
"""
import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fox quadruped servo demo in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of robots (motions repeat every 4).")
parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 = until the window is closed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.kit_args:   # full-screen Kit on this 3200x2000 panel renders ~1.7 s/frame; a 1280x800 window is fast
    args_cli.kit_args = "--/app/window/width=1280 --/app/window/height=800"
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fox_cfg import FOX_CFG, LEGS  # noqa: E402

DT = 1.0 / 200.0
RENDER_EVERY = 4
SMALL_GPU = PhysxCfg(  # defaults reserve ~1 GB (8.4 M contacts); a handful of 17-link robots needs far less
    gpu_max_rigid_contact_count=2**16, gpu_max_rigid_patch_count=2**14, gpu_found_lost_pairs_capacity=2**16,
    gpu_found_lost_aggregate_pairs_capacity=2**18, gpu_total_aggregate_pairs_capacity=2**16,
    gpu_collision_stack_size=2**22, gpu_heap_capacity=2**22, gpu_temp_buffer_capacity=2**20,
    gpu_max_soft_body_contacts=2**10, gpu_max_particle_contacts=2**10)


@configclass
class FoxSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)))
    robot: ArticulationCfg = FOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def servo_targets(t: float, n: int, device) -> torch.Tensor:
    """(n, 12) servo targets [hip x4, pivot x4, gear x4], one motion per env (ramped in over the first second)."""
    s = min(1.0, t) * math.sin(2 * math.pi * t / 3.0)        # 3 s period
    u = torch.zeros(n, 12, device=device)
    for e in range(n):
        motion = e % 4
        if motion == 0:      # crouch / rise: thigh +0.25, knee -0.45 at the bottom -> pivot 0.25, gear 0.20
            c = 0.5 * (1 - math.cos(2 * math.pi * t / 3.0)) * min(1.0, t)
            u[e, 4:8], u[e, 8:12] = 0.25 * c, 0.20 * c
        elif motion == 1:    # sway: every hip servo turns the same way
            u[e, 0:4] = 0.15 * s
        elif motion == 2:    # pendulum: femur and crank together
            u[e, 4:8], u[e, 8:12] = 0.15 * s, -0.15 * s
        else:                # gear servos alone: the shins swing
            u[e, 8:12] = 0.12 * s
    return u


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device, physics=SMALL_GPU))
    sim.set_camera_view([0.9, -1.1, 0.55], [0.0, 0.6, 0.05])
    scene = InteractiveScene(FoxSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.6))
    sim.reset()
    robot = scene["robot"]
    jid = {}
    for part in ("hip", "thigh", "calf"):        # targets on these joints = hip / pivot / gear servo angles
        ids, _ = robot.find_joints(["%s_%s_joint" % (leg, part) for leg in LEGS], preserve_order=True)
        jid[part] = ids
    cols = jid["hip"] + jid["thigh"] + jid["calf"]
    print("[INFO] Fox: %d envs, joints %s" % (scene.num_envs, robot.joint_names))
    t, k, n, wall0 = 0.0, 0, scene.num_envs, time.time()
    while simulation_app.is_running() and (args_cli.duration <= 0 or t < args_cli.duration):
        target = robot.data.default_joint_pos.torch.clone()
        target[:, cols] = servo_targets(t, n, sim.device)
        robot.set_joint_position_target_index(target=target)
        scene.write_data_to_sim()
        sim.step(render=k % RENDER_EVERY == 0)   # 200 Hz physics, 50 Hz picture
        scene.update(DT)
        t, k = t + DT, k + 1
        if k % 400 == 0:   # every 2 s: base height and uprightness per env
            z = robot.data.root_pos_w.torch[:, 2]
            q = robot.data.root_quat_w.torch                  # Isaac Lab 3.0: x, y, z, w
            up = 1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2)
            print("[INFO] t=%5.1f s (%.2fx real time)  base z %s  upright %s" % (t, t / (time.time() - wall0), [round(float(x), 3) for x in z],
                                                                       [round(float(x), 3) for x in up]), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
