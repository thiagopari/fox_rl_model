#!/usr/bin/env python3
"""Headless Isaac Sim 6.0 test of usd/fox.usd (reduced tree + foot mimics) driven like the real robot: one saturated
PD per servo (hip / pivot / gear, kp 2.0, kd 0.05, 0.47 N m), mapped onto the joints by the servo map
    thigh = pivot, calf = -gear - pivot  ->  tau_thigh = t_pivot - t_gear, tau_calf = -t_gear   (= isaaclab/fox_cfg.py)
Two robots, GPU PhysX at 200 Hz:
  /World/fox       on the ground (base at z=0.155), all servos at 0 for 3 s: must stand
  /World/fox_held  bolted to the world in the air: gear servos +0.3, then pivot servos +0.3, then pivot +0.3 with
                   gear -0.3 (pendulum); joint angles must follow the servo map and the feet must follow their mimic.
Adds the results to validation/isaacsim_report.json and renders validation/isaacsim_stand.png.

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
from pxr import Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "usd", "fox.usd")
REPORT = os.path.join(ROOT, "validation", "isaacsim_report.json")
PNG = os.path.join(ROOT, "validation", "isaacsim_stand.png")
MECH = json.load(open(os.path.join(ROOT, "mechanism.json")))
DT, Z0 = 1.0 / 200.0, 0.155  # feet are 0.1528 m below the base origin at the zero pose
KP, KD, TMAX, ARM = 2.0, 0.05, 0.47, 0.0005
LEGS = ("FL", "FR", "RL", "RR")
EXPECTED = ["%s_%s_joint" % (leg, part) for leg in LEGS for part in ("hip", "thigh", "calf", "foot")]

stage = stage_utils.create_new_stage()
stage_utils.set_stage_up_axis("Z")
stage_utils.set_stage_units(meters_per_unit=1.0)
GroundPlane("/World/ground")
UsdGeom.XformCommonAPI(stage_utils.add_reference_to_stage(USD, "/World/fox")).SetTranslate((0.0, 0.0, Z0))
UsdGeom.XformCommonAPI(stage_utils.add_reference_to_stage(USD, "/World/fox_held")).SetTranslate((0.0, 1.0, 0.4))
bolt = UsdPhysics.FixedJoint.Define(stage, "/World/fox_held_bolt")  # world -> base link: a fixed-base articulation
bolt.CreateBody1Rel().SetTargets([Sdf.Path("/World/fox_held/Geometry/base")])
bolt.CreateLocalPos0Attr().Set((0.0, 1.0, 0.4))
SimulationManager.setup_simulation(dt=DT, device="cuda:0")  # GPU pipeline, like Isaac Lab
robot, held = Articulation("/World/fox"), Articulation("/World/fox_held")

omni.timeline.get_timeline_interface().play()
SimulationManager.initialize_physics()  # loads physics, takes one warm-up step, fires PHYSICS_READY
assert robot.is_physics_tensor_entity_valid() and held.is_physics_tensor_entity_valid(), "articulation views not created"

np1 = lambda a: a.numpy()[0]  # noqa: E731  (warp (1, n) array -> numpy row)
names = list(robot.dof_names)
ix = {n: names.index(n) for n in names}
H, T, C, F = ([ix["%s_%s_joint" % (leg, p)] for leg in LEGS] for p in ("hip", "thigh", "calf", "foot"))
for a in (robot, held):  # explicit servo model: no joint-space drive, armature on the servo joints
    a.set_dof_gains(np.zeros((1, a.num_dofs), np.float32), np.zeros((1, a.num_dofs), np.float32))
    arm = np.zeros((1, a.num_dofs), np.float32)
    arm[0, H + T + C] = ARM
    a.set_dof_armatures(arm)


def efforts(a, servo_target):
    """servo_target: (12,) [hip x4, pivot x4, gear x4] -> joint efforts via the servo map."""
    q, qd = np1(a.get_dof_positions()), np1(a.get_dof_velocities())
    s = np.concatenate([q[H], q[T], -(q[T] + q[C])])
    v = np.concatenate([qd[H], qd[T], -(qd[T] + qd[C])])
    t = np.clip(KP * (servo_target - s) - KD * v, -TMAX, TMAX)
    e = np.zeros(a.num_dofs, np.float32)
    e[H], e[T], e[C] = t[:4], t[4:8] - t[8:], -t[8:]
    a.set_dof_efforts(e[None])
    return q


props = {"dof_count": robot.num_dofs, "dof_names": names, "link_count": robot.num_links, "link_names": robot.link_names,
         "collision_shapes": robot.num_shapes, "total_mass_kg": round(float(np1(robot.get_link_masses()).sum()), 6),
         "dof_limits_rad": [np.round(x, 4).tolist() for x in map(np1, robot.get_dof_limits())],
         "physics": {"engine": SimulationManager.get_active_physics_engine(), "device": str(SimulationManager.get_device()), "dt_s": DT}}
mult = np.array([MECH["legs"][leg]["foot_poly"][1] for leg in LEGS])
phases = [("hold", np.zeros(12)), ("gear servos +0.3", np.r_[np.zeros(8), 0.3 * np.ones(4)]),
          ("pivot servos +0.3", np.r_[np.zeros(4), 0.3 * np.ones(4), np.zeros(4)]),
          ("pendulum: pivot +0.3, gear -0.3", np.r_[np.zeros(4), 0.3 * np.ones(4), -0.3 * np.ones(4)])]
trace, held_res, mimic_err, min_z, min_up = [], {}, 0.0, 1e9, 1e9
for name, tgt in phases:
    for k in range(int(1.0 / DT)):
        qs = efforts(robot, np.zeros(12))
        qh = efforts(held, tgt)
        SimulationManager.step(update_fabric=True)
        pos, quat = robot.get_world_poses()
        z, (w, x, y, _) = float(np1(pos)[2]), np1(quat)
        up = float(1.0 - 2.0 * (x * x + y * y))
        min_z, min_up = min(min_z, z), min(min_up, up)
        if k > 50:  # after the mimic constraint settles
            mimic_err = max(mimic_err, float(np.abs(qh[F] - mult * qh[C]).max()), float(np.abs(qs[F] - mult * qs[C]).max()))
        if k % 20 == 0:
            trace.append([round(len(trace) * 20 * DT, 2), round(z, 5), round(up, 5)])
    held_res[name] = {"femur": np.round(qh[T], 4).tolist(), "shin": np.round(qh[T] + qh[C], 4).tolist(), "knee": np.round(qh[C], 4).tolist(),
                      "foot": np.round(qh[F], 4).tolist()}
g, p, s = (held_res[n] for n, _ in phases[1:])
roles_ok = (np.allclose(g["femur"], 0, atol=0.03) and np.allclose(g["shin"], -0.3, atol=0.03)
            and np.allclose(p["femur"], 0.3, atol=0.03) and np.allclose(p["shin"], 0, atol=0.03)
            and np.allclose(s["knee"], 0, atol=0.03) and np.allclose(s["femur"], 0.3, atol=0.03))
finite = bool(np.isfinite(trace).all())
stand = {"spawn_base_z_m": Z0, "base_z_trace_t_z_upcos": trace[::5], "final_base_z_m": round(z, 5), "min_base_z_m": round(min_z, 5),
         "final_upright_cos": round(up, 6), "min_upright_cos": round(min_up, 6), "finite": finite,
         "final_joint_pos_rad": dict(zip(names, np.round(qs, 4).tolist()))}

rep = {}
if os.path.exists(REPORT):
    with open(REPORT) as f:
        rep = json.load(f)
rep.setdefault("api", {})["articulation"] = ("isaacsim.core.experimental.prims.Articulation + SimulationManager; "
                                             "servo PD computed in this script, applied with set_dof_efforts")
rep["articulation_properties"], rep["stand_test"], rep["held_servo_test"] = props, stand, held_res
rep["foot_mimic_max_err_rad"] = mimic_err
rep["checks"] = checks = {
    "a_import_succeeded": bool(rep.get("import", {}).get("ok")) and robot.is_physics_tensor_entity_valid(),
    "b_16_dofs_expected_names": props["dof_count"] == 16 and sorted(names) == sorted(EXPECTED),
    "c_stands_4s_upright_cos_gt_0.95_and_base_z_gt_0.10": finite and min_up > 0.95 and min_z > 0.10,
    "d_servo_map: gear->shin, pivot->femur, pivot+gear(-)->pendulum (held robot, 0.03 rad)": bool(roles_ok),
    "e_foot_mimics_follow_the_knee_(< 0.01 rad)": mimic_err < 0.01,
}

try:  # optional render of the final pose
    import omni.replicator.core as rep_core
    from PIL import Image

    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(600.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.XformCommonAPI(sun).SetRotate((40.0, 0.0, 30.0))
    cam = rep_core.create.camera(position=(0.35, -0.30, 0.25), look_at=(0.0, 0.0, 0.10), clipping_range=(0.01, 1e3))
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
    print("%-80s %s" % (name, "PASS" if val else "FAIL"))
print("dofs %d links %d mass %.4f kg | base z %.4f -> %.4f (min %.4f) | upright %.5f | foot mimic err %.4f rad"
      % (props["dof_count"], props["link_count"], props["total_mass_kg"], Z0, z, min_z, up, mimic_err))
for name, r in held_res.items():
    print("  held %-34s femur %s shin %s" % (name, r["femur"], r["shin"]))
app.close(exit_code=0 if rep["pass"] else 1)
