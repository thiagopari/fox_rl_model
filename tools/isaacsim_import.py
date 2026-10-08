#!/usr/bin/env python3
"""Import urdf/fox.urdf into a single self-contained USD for Isaac Sim 6.0 / Isaac Lab: usd/fox.usd (FOX_MODEL=v7: v7/...).

Rerun (headless, ~20 s):
    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python ~/Documents/fox_rl_model/tools/isaacsim_import.py

Uses the Isaac Sim 6.0 URDF importer (isaacsim.asset.importer.urdf 3.x, urdf-usd-converter backend) with
Isaac Lab convert_urdf-style settings, flattens the importer's package (Physics variant "physx") into one
binary USD (meshes stay instanceable), re-applies the drive gains (see DRIVE FIX below) and flattens the body
hierarchy for Isaac Lab's contact sensors (see FLAT HIERARCHY).
Starts a fresh validation/isaacsim_report.json ("import" section); tools/isaacsim_stand_test.py adds the rest.
"""
import json
import math
import os
import shutil
import sys
import tempfile
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
sys.excepthook = lambda *exc: (traceback.print_exception(*exc), app.close(exit_code=1))  # else Kit exits 0

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = os.path.abspath(os.environ.get("FOX_MODEL", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # FOX_MODEL=v7: another model folder
URDF = os.path.join(ROOT, "urdf", "fox.urdf")
OUT = os.path.join(ROOT, "usd", "fox.usd")
REPORT = os.path.join(ROOT, "validation", "isaacsim_report.json")
KP, KD = 5.0, 0.08  # N m/rad, N m s/rad: same as the MuJoCo position servos (tools/build_robot.py KP, KV)
DEG = math.pi / 180.0  # USD angular drive gains are per degree

def flatten_bodies(layer, stage, root="/fox/Geometry"):
    """FLAT HIERARCHY: the importer nests links along the kinematic tree (base/FL_hip/FL_thigh/...); Isaac Lab's
    contact sensor only finds bodies that are direct children of one parent (it found just "base"). Move every rigid
    body to root/<name> keeping its world pose (joint frames are body-local, so they stay valid) and retarget all
    relationships (physics:body0/1, isaac:physics:robotLinks) to the new paths."""
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    world = {p.GetPath(): UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) for p in bodies}
    root_w = UsdGeom.Xformable(stage.GetPrimAtPath(root)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    moved = {}
    for old in sorted(world, key=lambda q: -q.pathElementCount):  # deepest first: children leave before parents move
        new = Sdf.Path(root).AppendChild(old.name)
        if old.GetParentPath() == Sdf.Path(root):
            continue
        Sdf.CopySpec(layer, old, layer, new)
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(old, Sdf.Path.emptyPath)
        assert layer.Apply(edit), "could not remove %s" % old
        local = Gf.Transform(world[old] * root_w.GetInverse())
        prim = stage.GetPrimAtPath(new)
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(local.GetTranslation()) if prim.GetAttribute("xformOp:translate").GetTypeName() == Sdf.ValueTypeNames.Double3 else Gf.Vec3f(local.GetTranslation()))
        q = local.GetRotation().GetQuat()
        orient = prim.GetAttribute("xformOp:orient")
        orient.Set(Gf.Quatd(q) if orient.GetTypeName() == Sdf.ValueTypeNames.Quatd else Gf.Quatf(q))
        moved[old] = new

    def remap(t):  # longest moved prefix wins (descendants of a moved body move with it)
        best = max((o for o in moved if t.HasPrefix(o)), key=lambda o: o.pathElementCount, default=None)
        return t.ReplacePrefix(best, moved[best]) if best is not None else t

    def fix(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            for r in spec.relationships:
                lo = r.targetPathList
                for name in ("explicitItems", "prependedItems", "appendedItems"):
                    items = list(getattr(lo, name))
                    if any(remap(t) != t for t in items):
                        setattr(lo, name, [remap(t) for t in items])
    layer.Traverse(Sdf.Path.absoluteRootPath, fix)
    return moved


cfg = URDFImporterConfig(
    urdf_path=URDF,
    fix_base=False,  # floating base
    merge_fixed_joints=False,
    allow_self_collision=False,
    collision_from_visuals=False,  # use the URDF <collision> meshes (the converter authors convexHull on them)
    collision_type="Convex Hull",  # only consulted when collision_from_visuals=True
    joint_drive_type="force",
    joint_target_type="position",
    override_joint_stiffness=KP,  # importer converts N m/rad -> N m/deg
    override_joint_damping=KD,
    robot_type="Quadruped",
)
error = None
tmp = cfg.usd_path = tempfile.mkdtemp(prefix="fox_urdf_import_")  # importer writes a package <tmp>/fox/fox.usda
try:
    layer = Usd.Stage.Open(URDFImporter(cfg).import_urdf()).Flatten()  # composes default Physics variant "physx"
    layer.documentation = "Fox quadruped from urdf/fox.urdf via tools/isaacsim_import.py (Isaac Sim 6.0.1 URDF importer)"
    # DRIVE FIX: the importer's URDF->PhysX pass runs after the gain overrides and writes <dynamics damping="0.01">
    # (N m s/rad) raw into physics:damping (N m s/deg), i.e. 0.57 N m s/rad. Re-apply the intended gains.
    flat = Usd.Stage.Open(layer)  # keep a reference: a temporary stage expires mid-Traverse()
    for j in flat.Traverse():
        if j.IsA(UsdPhysics.RevoluteJoint):
            foot = j.GetName().endswith("_foot_joint")  # passive: its <mimic> (NewtonMimicAPI) moves it with the knee
            UsdPhysics.DriveAPI(j, "angular").GetStiffnessAttr().Set(0.0 if foot else KP * DEG)
            UsdPhysics.DriveAPI(j, "angular").GetDampingAttr().Set(0.0 if foot else KD * DEG)
    flatten_bodies(layer, flat)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    layer.Export(OUT)
except Exception as e:
    error = "%s: %s" % (type(e).__name__, e)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# summarize what actually landed in the USD, re-opened from disk
summary = {}
if error is None:
    st = Usd.Stage.Open(OUT)
    prims = list(st.Traverse(Usd.TraverseInstanceProxies()))
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in prims if p.IsA(UsdPhysics.RevoluteJoint)]
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    d0 = UsdPhysics.DriveAPI(joints[0], "angular")
    summary = {
        "default_prim": str(st.GetDefaultPrim().GetPath()),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(st),
        "up_axis": str(UsdGeom.GetStageUpAxis(st)),
        "articulation_root": [str(p.GetPath()) for p in roots],
        "self_collision_enabled": [p.GetAttribute("newton:selfCollisionEnabled").Get() for p in roots],
        "rigid_bodies": len(bodies),
        "revolute_joints": [p.GetName() for p in joints],
        "colliders": len(colliders),
        "collider_approximations": sorted({str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()) for p in colliders}),
        "instanceable_prims": sum(p.IsInstance() for p in prims),
        "total_mass_kg": round(sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies), 6),
        "drive_usd_joint0": {"type": d0.GetTypeAttr().Get(), "stiffness_Nm_per_deg": d0.GetStiffnessAttr().Get(),
                             "damping_Nms_per_deg": d0.GetDampingAttr().Get(), "max_force_Nm": d0.GetMaxForceAttr().Get(),
                             "target_position_deg": d0.GetTargetPositionAttr().Get(),
                             "armature": joints[0].GetAttribute("physxJoint:armature").Get(),
                             "joint_friction": joints[0].GetAttribute("physxJoint:jointFriction").Get(),
                             "max_velocity_deg_s": joints[0].GetAttribute("physxJoint:maxJointVelocity").Get()},
        "file_size_MB": round(os.path.getsize(OUT) / 1e6, 2),
        "bodies_flat": all(p.GetParent().GetPath() == Sdf.Path("/fox/Geometry") for p in bodies),
        "foot_mimics": {p.GetName(): [p.GetAttribute("newton:mimicCoef1").Get(), [str(t) for t in p.GetRelationship("newton:mimicJoint").GetTargets()]]
                        for p in joints if p.HasAPI("NewtonMimicAPI")},
    }
ok = error is None and summary["rigid_bodies"] == 17 and len(summary["revolute_joints"]) == 16 \
    and summary["colliders"] == 17 and summary["collider_approximations"] == ["convexHull"] and len(summary["foot_mimics"]) == 4 \
    and summary["bodies_flat"]
config = {k: v for k, v in vars(cfg).items() if k != "usd_path"}
config.update(output=OUT, stiffness_Nm_per_rad=KP, damping_Nms_per_rad=KD,
              post_fix="drive stiffness/damping re-set to KP*pi/180, KD*pi/180 on the flattened layer",
              flattened="single binary USD, Physics variant 'physx', instanceable meshes kept")
rep = {"api": {"importer": "isaacsim.asset.importer.urdf.URDFImporter + URDFImporterConfig "
                           "(extension 3.11.2, urdf-usd-converter 0.1.3)", "isaac_sim": "6.0.1"},
       "import": {"config": config, "result": summary, "error": error, "ok": bool(ok)}}
os.makedirs(os.path.dirname(REPORT), exist_ok=True)
with open(REPORT, "w") as f:
    json.dump(rep, f, indent=1)
print(json.dumps(rep["import"], indent=1))
print("IMPORT", "PASS" if ok else "FAIL", OUT)
app.close(exit_code=0 if ok else 1)
