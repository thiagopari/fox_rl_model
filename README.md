# Fox quadruped — RL-ready model

Link-ready model of the Fox quadruped (Fusion design `Fox_Prototype_03_v6`) for reinforcement learning in
**Isaac Lab / Isaac Sim** (URDF → USD) and **MuJoCo** (MJCF). Generated on 2026-10-06 from the design copy
`Fox_Prototype_03_v6_RL` (v14). The original design was never modified.

## Contents
| Path | What |
|---|---|
| `urdf/fox.urdf` | URDF: 13 links, 12 revolute joints, inertials, visual + collision meshes |
| `mjcf/fox.xml`, `mjcf/scene.xml` | MuJoCo model (floating base, 12 position servos, IMU site/sensors) + scene with floor |
| `usd/fox.usd` | Isaac Sim 6.0.1 import of the URDF (see validation) |
| `isaaclab/fox_cfg.py` | Isaac Lab `ArticulationCfg` (drop-in analogue of `UNITREE_GO1_CFG`) |
| `meshes/visual`, `meshes/collision` | per-link STL in the link frame (decimated visuals, convex-hull collisions), metres |
| `raw/` | raw Fusion export (full-resolution STLs + `raw.json` with masses / inertia / joints) |
| `tools/` | `build_robot.py` (raw → URDF/MJCF/meshes), MuJoCo checks, Isaac Sim import, Fusion add-ins |
| `validation/` | reports + renders |

Fusion documents (cloud, work › Default Project):
* `Fox_Prototype_03_v6` — original, untouched.
* `Fox_Prototype_03_v6_RL` — design copy with the fixes (mirrored front leg, rear gear pair, servos, restored tibias, Component77).
* `Fox_Prototype_03_v6_RL_URDF_v2` — the **link + joint model**: 13 top-level components (one per link, no nesting),
  RL materials, 12 as-built revolute joints. Open it and drag a leg: every part screwed to a moving part moves with it.
* `Fox_Prototype_03_v6_RL_URDF` — first link model, **superseded** by `_v2` (brackets stayed on the base); safe to delete.

## Conventions
* Frame: **x forward** (IMU end), **y left**, **z up**; base origin midway between the four hips at hip-axis height.
* Zero joint angles = the CAD standing pose; the base origin is **0.1528 m** above the ground there.
* Joint names follow Unitree: `*_hip_joint` (abduction, ≈+x), `*_thigh_joint` (hip pitch, +y), `*_calf_joint` (knee, +y).
  Legs: FL, FR, RL, RR. The abduction axes are the real hip-servo spline axes, which the CAD tilts out of the
  horizontal: **front (0.985, 0, −0.174)** (10° nose-down), **rear (0.940, 0, 0.342)** (20° nose-up). Code that assumes a
  pure +x abduction axis (e.g. analytic IK) must use these.
* Feet are the `*_calf` links (tibia + foot + calf link). Total mass **542.6 g** (no battery yet).

## Kinematics and masses
| Joint | Parent → child | Axis | Origin in parent (m) | Limits (rad) |
|---|---|---|---|---|
| FL_hip_joint | base → FL_hip | (0.985, 0, -0.174) | 0.0717 0.0242 0.0074 | -0.50 … 0.50 |
| FL_thigh_joint | FL_hip → FL_thigh | +y | 0.0131 0.0186 -0.0026 | -1.20 … 1.20 |
| FL_calf_joint | FL_thigh → FL_calf | +y | -0.0263 0.0016 -0.0703 | -1.20 … 1.20 |
| FR_hip_joint | base → FR_hip | (0.985, 0, -0.174) | 0.0717 -0.0243 0.0074 | -0.50 … 0.50 |
| FR_thigh_joint | FR_hip → FR_thigh | +y | 0.0131 -0.0185 -0.0026 | -1.20 … 1.20 |
| FR_calf_joint | FR_thigh → FR_calf | +y | -0.0263 -0.0017 -0.0703 | -1.20 … 1.20 |
| RL_hip_joint | base → RL_hip | (0.940, 0, 0.342) | -0.0736 0.0242 -0.0003 | -0.50 … 0.50 |
| RL_thigh_joint | RL_hip → RL_thigh | +y | -0.0124 0.0186 -0.0044 | -1.20 … 1.20 |
| RL_calf_joint | RL_thigh → RL_calf | +y | 0.0381 0.0018 -0.0646 | -1.20 … 1.20 |
| RR_hip_joint | base → RR_hip | (0.940, 0, 0.342) | -0.0734 -0.0242 -0.0002 | -0.50 … 0.50 |
| RR_thigh_joint | RR_hip → RR_thigh | +y | -0.0126 -0.0185 -0.0046 | -1.20 … 1.20 |
| RR_calf_joint | RR_thigh → RR_calf | +y | 0.0381 -0.0017 -0.0646 | -1.20 … 1.20 |

| Link | Mass (g) | COM in link frame (m) |
|---|---|---|
| base | 330.4 | -0.0006 0.0009 0.0162 |
| FL_hip | 34.8 | 0.0198 0.0039 -0.0087 |
| FL_thigh | 7.1 | -0.0194 0.0000 -0.0312 |
| FL_calf | 10.4 | 0.0040 0.0000 -0.0350 |
| FR_hip | 34.8 | 0.0198 -0.0039 -0.0087 |
| FR_thigh | 7.1 | -0.0194 -0.0000 -0.0312 |
| FR_calf | 10.7 | 0.0039 0.0000 -0.0339 |
| RL_hip | 35.3 | -0.0180 0.0039 -0.0117 |
| RL_thigh | 7.4 | 0.0246 0.0000 -0.0283 |
| RL_calf | 10.4 | -0.0260 -0.0000 -0.0307 |
| RR_hip | 36.8 | -0.0176 -0.0036 -0.0124 |
| RR_thigh | 7.3 | 0.0247 -0.0000 -0.0284 |
| RR_calf | 10.0 | -0.0266 -0.0000 -0.0316 |

## How the CAD maps to links
Link membership follows the fasteners (`tools/screw_graph.py` on a scan of every hole, pin and spline in the design):
parts that share a screw line, or that sit on a servo's output spline, are rigidly joined and always land in the same
link. Pin joints of the leg linkages and the spline axes themselves are pivots, not fasteners. The result is
`analysis/link_map.json` (one line per body, plus the 4 abduction axes).
* **base**: both pelvis halves, Pi mount, Raspberry Pi 4, 12-ch PWM board, 2× XL4015, Arduino Nano, BNO055, the bodies
  of the 4 inboard ("hip") servos that drive abduction, and everything screwed to those.
* **XX_hip**: the hip bracket that rides on the inboard servo's spline (front: `Servo Pelv Upper/Lower(Mirror)`; rear
  left: `Servo Pelv Upper`; rear right: `Component40/41(Mirror) (1)`), the rear bracket caps (`Body28`/`Body29`), the
  two servos clamped in the bracket (leg-pivot servo + gear servo), the gear servo's pinion (`m` gear) and the 15-tooth
  crank gear.
* **XX_thigh**: femur + the leg-pivot servo's output spline it is mounted on (+ the horn screw `Body4` on the rear left)
  + Quad Link riding along as mass.
* **XX_calf**: tibia, foot, the foot-levelling link (Component77 front, Calf Link hind) and the loose knee pin (`Body2`).
* Changed vs the first model (`_RL_URDF`): the 4 hip brackets, 2 caps and 4 inboard-servo splines moved base → hip;
  the 4 leg-pivot splines and the horn screw hip → thigh; the rear-left knee pin base → calf; plus `Body27`, a 0.07 cm³
  hole-free block on the rear-right leg-pivot servo, assigned by contact (base → RR_hip). Hip links went 22 → 35 g each.
* Hidden CAD bodies are excluded. Materials: PLA 1.24 g/cm³; each DS-843MG servo 8.5 g; electronics at typical masses
  (Pi 46 g, PWM 9 g, XL4015 16 g, Nano 7 g, BNO055 3 g).

## Modelling decisions (made autonomously — review)
1. **Hip abduction axis** = the output-spline axis of each inboard servo (the bracket turns on it), measured from the CAD:
   tilted 10° (front) / 20° (rear) from +x as listed above.
2. **Knee four-bar → serial chain**: the gear servo drives the knee through a pinion, a crank on the 15-tooth gear and the
   Quad Link. In the model the knee is a direct revolute joint; on hardware map knee angle → servo angle with the
   four-bar relation. Model the loop only if sim-to-real needs it.
3. **Limits are placeholders**: abduction ±0.5 rad, thigh ±1.2 rad, calf ±1.2 rad around the standing pose. Measure the
   real servo travel (DS-843MG is often listed at only ~±20–60°) and mechanical stops, then edit `build_robot.py`.
4. **Actuators**: DS-843MG at 6 V — 0.47 N·m stall, 10.5 rad/s; Isaac Lab `DCMotorCfg` effort limit 0.30 N·m,
   stiffness 2.0, damping 0.05, armature 5e-4 kg·m² (estimate of reflected rotor inertia). Gear/linkage ratios not folded in.
5. **Battery**: the battery bay is hidden in the CAD and no battery is modelled. Add its mass to `base` when known
   (it will dominate the ~0.33 kg base).
6. **Joint damping / friction / armature are not in the URDF** (written as 0 on purpose): Isaac Sim's URDF importer writes URDF
   damping into the per-degree USD field (0.01 would become ~0.57 N·m·s/rad), and URDF cannot carry armature. They live in
   `isaaclab/fox_cfg.py` (DCMotorCfg) and in the MJCF `<default>`. Without armature, PhysX shows a spurious ~0.75 rad/s joint
   velocity at rest; with armature 5e-4 it matches MuJoCo.
7. **Rear brackets un-joined**: in the design copy, `Combine1`/`Combine2` of `Component41(Mirror) (1)` and `Combine2` of
   `Servo Pelv Upper` join both rear brackets (and caps) into the pelvis, which would weld the rear legs' abduction.
   The link-model builder suppresses those 3 features **in memory only** (the design copy is closed unsaved); the
   timeline stays healthy without them. If the joins are meant for printing, nothing changes on your side.

## Use
* MuJoCo viewer: `~/.venvs/fox_rl/bin/python -m mujoco.viewer --mjcf ~/Documents/fox_rl_model/mjcf/scene.xml`
* Isaac Sim (installed: `~/isaacenv`, 6.0.1): regenerate the USD and run the standing test with
  `OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/isaacsim_import.py` and `... tools/isaacsim_stand_test.py`
  (importer: `isaacsim.asset.importer.urdf.URDFImporter`, floating base, convex-hull collisions from the collision meshes,
  position drives kp 2.0 / kd 0.05 per rad, self-collision off; `usd/fox.usd` is self-contained, default prim `/fox`).
* Isaac Lab: convert with `scripts/tools/convert_urdf.py` (command in `isaaclab/fox_cfg.py`) or use `usd/fox.usd`, then
  clone the Go1 velocity task: base `base`, feet `.*_calf`, undesired contacts `.*_thigh`, much smaller commands
  (≈0.3 m/s) and a lower target height (~0.14 m) than Go1.

## Regenerate
Each `tools/fusion/*` folder is a one-shot add-in: copy it into Fusion's `API/AddIns`, start Fusion, wait for its result
in `C:\fusion_jobs` (`drive_c/fusion_jobs` in the Wine prefix), then remove the folder again (it runs at every start).
1. After editing the design copy: `FoxScrewScan` → copy `screw_scan.json` to `analysis/`, run
   `~/.venvs/fox_rl/bin/python tools/screw_graph.py` (fails on any fastener conflict) and copy `analysis/link_map.json`
   into `tools/fusion/FoxBuildLinkModel/`.
2. `FoxBuildLinkModel` writes a new `…_RL_URDF_v2` document (timestamped if the name exists); it refuses to run if any
   body is missing from `link_map.json`. Wait until the upload finishes (Fusion log: `Uploaded document`) before closing Fusion.
3. `FoxExportRaw` → copy `C:\fusion_jobs\export` to `raw/`.
4. `~/.venvs/fox_rl/bin/python tools/build_robot.py`, then the checks in `tools/`.

## Validation
All checks pass (reports in `validation/`):

| Check | Result |
|---|---|
| Fasteners: every part sharing a screw line or a spline is in the same link | 380 rigid joins, 0 conflicts, 12/12 splines matched |
| Fusion link model: 13 links, 12 joints, each joint rotates its child rigidly about the intended axis | error 0.0 cm, all healthy |
| Exact geometry copy (542 visible bodies) | 0.0 µm vertex deviation; volume preserved |
| Servo masses | 12 × 8.50 g |
| URDF parse (yourdfpy) + mesh paths | 13 links, 12 actuated joints, no missing meshes |
| URDF vs MJCF kinematics | identical (3e-17 m) |
| MuJoCo: structure | free base + 12 hinges + 12 position servos |
| MuJoCo: all four feet touch the floor at the zero pose | all at +2 mm |
| MuJoCo: stands 4 s holding the zero pose | base 0.1528 → 0.1498 m, upright 1.000, max servo torque 0.037 N·m (of 0.47) |
| MuJoCo: crouch / abduction / single-leg lift | body lowers 8 mm / 10 mm, FL foot lifts 19 mm, stays upright |
| MuJoCo: 20 s random joint targets (RL exploration) | always finite, contact penetration ≤ 0.6 mm |
| Isaac Sim 6.0.1 URDF import → `usd/fox.usd` | 12 DOF with the expected names, 13 links |
| Isaac Sim: stands 3 s (GPU PhysX, dt 1/200) | base 0.155 → 0.1496 m, upright 1.000, holding torque ≤ 0.092 N·m (matches MuJoCo) |

Renders: `validation/mujoco_hip_abduction.png` (hip links turning as one piece), `mujoco_front_left.png`, `mujoco_poses.png`,
`mujoco_lift_FL.png`, `isaacsim_stand.png`.
Re-run: `~/.venvs/fox_rl/bin/python tools/validate_mujoco.py`, `tools/pose_test_mujoco.py`, `tools/random_actions_mujoco.py`;
Isaac Sim: `OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/isaacsim_import.py` then `tools/isaacsim_stand_test.py`.
