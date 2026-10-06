#!/usr/bin/env python3
"""Headless Isaac Sim 6.0 stand test for usd/fox.usd: ground plane, base at z=0.155, 3 s at 200 Hz (GPU PhysX),
all joint position targets held at 0. Adds a "stand_test" section + overall checks to validation/isaacsim_report.json
and renders validation/isaacsim_stand.png.

Rerun (after tools/isaacsim_import.py):
    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python ~/Documents/fox_rl_model/tools/isaacsim_stand_test.py
"""
import json
import os
import sys
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
sys.excepthook = lambda *exc: (traceback.print_exception(*exc), app.close(exit_code=1))  # else Kit exits 0

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import UsdGeom, UsdLux  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "usd", "fox.usd")
REPORT = os.path.join(ROOT, "validation", "isaacsim_report.json")
PNG = os.path.join(ROOT, "validation", "isaacsim_stand.png")
DT, T, Z0 = 1.0 / 200.0, 3.0, 0.155  # feet are 0.1528 m below the base origin at the zero pose
EXPECTED = ["%s_%s_joint" % (leg, part) for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]

stage = stage_utils.create_new_stage()
stage_utils.set_stage_up_axis("Z")
stage_utils.set_stage_units(meters_per_unit=1.0)
GroundPlane("/World/ground")
UsdGeom.XformCommonAPI(stage_utils.add_reference_to_stage(USD, "/World/fox")).SetTranslate((0.0, 0.0, Z0))
SimulationManager.setup_simulation(dt=DT, device="cuda:0")  # GPU pipeline, like Isaac Lab
robot = Articulation("/World/fox")  # resolves to the ArticulationRootAPI prim (the base link)

omni.timeline.get_timeline_interface().play()
SimulationManager.initialize_physics()  # loads physics, takes one warm-up step, fires PHYSICS_READY
assert robot.is_physics_tensor_entity_valid(), "articulation physics view not created"

np1 = lambda a: a.numpy()[0]  # noqa: E731  (warp (1, n) array -> numpy row)
kp, kd = map(np1, robot.get_dof_gains())
lower, upper = map(np1, robot.get_dof_limits())
props = {
    "dof_count": robot.num_dofs,
    "dof_names": robot.dof_names,  # PhysX articulation (BFS) order, i.e. the order RL actions/obs use
    "link_count": robot.num_links,
    "link_names": robot.link_names,
    "collision_shapes": robot.num_shapes,
    "total_mass_kg": round(float(np1(robot.get_link_masses()).sum()), 6),
    "dof_stiffness_Nm_per_rad": np.round(kp, 4).tolist(),
    "dof_damping_Nms_per_rad": np.round(kd, 4).tolist(),
    "dof_max_effort_Nm": np.round(np1(robot.get_dof_max_efforts()), 4).tolist(),
    "dof_limits_rad": [np.round(lower, 4).tolist(), np.round(upper, 4).tolist()],
    "dof_armature": np.round(np1(robot.get_dof_armatures()), 6).tolist(),
    # what PhysX 110 parses (the Articulation getter only reads the deprecated physxArticulation:* attribute)
    "self_collisions_enabled": robot.prims[0].GetAttribute("newton:selfCollisionEnabled").Get(),
    "physics": {"engine": SimulationManager.get_active_physics_engine(), "device": str(SimulationManager.get_device()),
                "dt_s": SimulationManager.get_physics_dt(), "duration_s": T},
}

trace, max_q, min_z, min_up = [], 0.0, 1e9, 1e9
zeros = np.zeros((1, robot.num_dofs), dtype=np.float32)
steps = int(round(T / DT))
for k in range(steps + 1):
    pos, quat = robot.get_world_poses()  # world frame, quaternion wxyz
    z, (w, x, y, _) = float(np1(pos)[2]), np1(quat)
    up = float(1.0 - 2.0 * (x * x + y * y))  # world-z component of the base z axis
    q = np1(robot.get_dof_positions())
    max_q, min_z, min_up = max(max_q, float(np.abs(q).max())), min(min_z, z), min(min_up, up)
    if k % 20 == 0:  # every 0.1 s
        trace.append([round(k * DT, 3), round(z, 5), round(up, 5)])
    if k == steps:
        break
    robot.set_dof_position_targets(zeros)
    SimulationManager.step(update_fabric=True)  # fabric keeps the renderer in sync for the PNG

# q is frozen at rest, but with armature 0 PhysX reports a constant non-physical q_dot (up to ~0.75 rad/s),
# so the holding torque is taken from the position error only
qd = np1(robot.get_dof_velocities())
torque = kp * np.abs(q)  # static PD holding torque with zero targets
finite = bool(np.isfinite(trace).all() and np.isfinite(q).all())
stand = {"spawn_base_z_m": Z0, "base_z_trace_t_z_upcos": trace, "final_base_z_m": round(z, 5),
         "min_base_z_m": round(min_z, 5), "final_upright_cos": round(up, 6), "min_upright_cos": round(min_up, 6),
         "final_base_xy_m": np.round(np1(pos)[:2], 5).tolist(), "max_abs_joint_pos_rad": round(max_q, 5),
         "final_joint_pos_rad": dict(zip(robot.dof_names, np.round(q, 5).tolist())),
         "final_max_abs_joint_vel_rad_s": round(float(np.abs(qd).max()), 4),
         "static_holding_torque_max_Nm": round(float(torque.max()), 4), "finite": finite}

rep = {}
if os.path.exists(REPORT):
    with open(REPORT) as f:
        rep = json.load(f)
rep.setdefault("api", {})["articulation"] = ("isaacsim.core.experimental.prims.Articulation + "
                                             "isaacsim.core.simulation_manager.SimulationManager (isaacsim.core.prims is "
                                             "deprecated in 6.0)")
rep["articulation_properties"], rep["stand_test"] = props, stand
rep["checks"] = checks = {
    "a_import_succeeded": bool(rep.get("import", {}).get("ok")) and robot.is_physics_tensor_entity_valid(),
    "b_12_dofs_expected_names": props["dof_count"] == 12 and sorted(props["dof_names"]) == sorted(EXPECTED),
    "c_stands_3s_upright_cos_gt_0.95_and_base_z_gt_0.10": finite and min_up > 0.95 and min_z > 0.10,
}

try:  # optional render of the final pose
    import omni.replicator.core as rep_core
    from PIL import Image

    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(600.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.XformCommonAPI(sun).SetRotate((40.0, 0.0, 30.0))
    cam = rep_core.create.camera(position=(0.35, 0.30, 0.25), look_at=(0.0, 0.0, 0.10), clipping_range=(0.01, 1e3))
    rgb = rep_core.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rep_core.create.render_product(cam, (960, 720))])
    omni.timeline.get_timeline_interface().pause()  # freeze physics while rendering
    rep_core.orchestrator.step(rt_subframes=16)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(PNG)
    rep["render_png"] = PNG
except Exception as e:  # rendering is optional; physics results above stand on their own
    rep["render_png"] = "failed: %s: %s" % (type(e).__name__, e)

rep["pass"] = all(checks.values())
with open(REPORT, "w") as f:
    json.dump(rep, f, indent=1)
for name, val in checks.items():
    print("%-55s %s" % (name, "PASS" if val else "FAIL"))
print("dofs %d links %d mass %.4f kg | base z %.4f -> %.4f (min %.4f) | upright %.5f | max|q| %.4f rad | torque %.3f N m"
      % (props["dof_count"], props["link_count"], props["total_mass_kg"], Z0, z, min_z, up, max_q, torque.max()))
print("kp", props["dof_stiffness_Nm_per_rad"][:3], "kd", props["dof_damping_Nms_per_rad"][:3], "png:", rep["render_png"])
app.close(exit_code=0 if rep["pass"] else 1)
