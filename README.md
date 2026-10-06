# Fox quadruped — RL-ready model

Link-ready model of the Fox quadruped (Fusion design `Fox_Prototype_03_v6`) for reinforcement learning in **MuJoCo** and
**Isaac Sim / Isaac Lab**, with the real leg mechanism: three servos per leg, a 1:1 gear pair, a parallelogram linkage
and a foot four-bar. Generated 2026-10-06 from the design copy `Fox_Prototype_03_v6_RL` (v14). The original design was never
modified. Previous checkpoint (screw-consistent serial model): git tag `checkpoint-screw-links`.

## The leg mechanism (what each servo does)
Per leg, three DS-843MG servos sit in the hip bracket:
* **Hip servo** (inboard): turns the whole leg about its own spline axis = abduction. The axes are tilted in the CAD:
  front (0.985, 0, −0.174), rear (0.940, 0, 0.342).
* **Pivot servo** ("adjacent"): its spline carries the **femur** (hip axis H).
* **Gear servo** ("outer"): its spline carries a 12-tooth pinion that drives a 12-tooth gear on the same hip axis H
  (module 1.5 mm, centre distance 18.00 mm, so **1:1, opposite direction**; the CAD names it "15 teeth", the geometry has
  12). The gear's crank arm (H→Q1, 15 mm) moves the **Quad Link** (Q1→Q2, 75 mm) which moves the **tibia** at Q2.
  Hip–crank–Quad Link–knee is an exact **parallelogram** (femur 75.0 mm = Quad Link, crank 15.0 mm = tibia offset), so the
  shin always keeps the crank's angle.
* **Foot**: hangs from the ankle and is steered by a second loop. On the rear legs it is a parallelogram with the
  Calf Link, so the foot stays parallel to the femur. On the front legs, Component77 forms a non-parallel four-bar:
  foot angle relative to the tibia = f(knee) ≈ 0.359·knee − 0.155·knee².

Servo map (pitch angles about +y, zero = CAD stance): **thigh = pivot, knee = −gear − pivot, foot = f(knee)**. So:
* the **gear servo alone** swings the shin (and foot) about the knee, like a pendulum; with the pivot servo unpowered it
  swings the whole leg (that is what Fusion shows when you drag it);
* the **pivot servo alone** turns the femur while the shin keeps its angle: the knee bends, the leg extends/retracts;
* **pivot +a with gear −a** swings the whole leg rigidly about the hip.

| Command (body held, pivot / gear servo) | Femur | Shin | Knee | FL foot: swing / extension | RL foot: swing / extension |
|---|---|---|---|---|---|
| gear servo +0.3 (pivot holds) | -0.00 | -0.30 | -0.30 | -0.183 rad / -15.5 mm | -0.101 rad / +16.1 mm |
| pivot servo +0.3 (gear holds) | +0.30 | -0.00 | -0.30 | +0.117 rad / -15.5 mm | +0.199 rad / +16.1 mm |
| pendulum: pivot +0.3, gear -0.3 | +0.30 | +0.30 | -0.00 | +0.300 rad / -0.0 mm | +0.300 rad / +0.0 mm |

Knee range is limited where a loop would reach a dead point (pins in line, 10° margin): rear knee
[-1.2, 0.74] rad (the rear foot parallelogram folds at +0.92 rad), front ±1.2 rad (placeholder).

## Contents
| Path | What |
|---|---|
| `mjcf/fox.xml`, `mjcf/scene.xml` | **exact mechanism** for MuJoCo: 33 bodies, 32 hinges, the 8 loop pins as `<connect>`, the 4 gear pairs as joint equalities, 12 position servos (`XX_hip`, `XX_pivot`, `XX_gear`), IMU site/sensors; scene with floor |
| `mjcf/fox_reduced.xml`, `scene_reduced.xml` | the same robot as a tree (17 bodies): gear servo on a fixed tendon (−thigh − calf), foot via a four-bar joint equality; matches the exact model |
| `urdf/fox.urdf` | that tree for Isaac (URDF cannot hold loops): 17 links, 16 revolute joints; foot joints are `<mimic>` of the knee |
| `usd/fox.usd` | Isaac Sim 6.0.1 import of the URDF (foot drives zeroed, mimics kept as `NewtonMimicAPI`) |
| `isaaclab/fox_cfg.py` | Isaac Lab `ArticulationCfg` + `FoxServoActuator`: targets on hip/thigh/calf joints are the hip/pivot/gear **servo** angles; `FOX_SERVO_REAL_CFG` adds latency, backlash and the torque-speed line |
| `isaaclab/fox_sim.py` | launches the robots in Isaac Lab (GUI or headless) and drives the servos |
| `isaaclab/fox_tasks.py`, `fox_train.py`, `fox_play.py` | RL walking tasks (`Fox-Velocity-Flat`, hardware-ready `Fox-Velocity-Flat-Blind`), PPO training, keyboard driving + eval |
| `policies/` | trained policies (`fox_flat_v1`, `fox_flat_blind_v1`, `fox_flat_blind_v2`, `fox_flat_blind_v3` = current), each with its ONNX export and config |
| `mechanism.json` | servo map, four-bar fits, knee ranges |
| `meshes/visual`, `meshes/collision` | per-part (`XX_femur.stl`…) and per-tree-link (`XX_thigh_reduced.stl`…) meshes, metres |
| `raw/` | raw Fusion export (per-part STLs + `raw.json` with masses, inertia, 40 joints, motion links) |
| `analysis/` | hole/pin scan of the CAD (`screw_scan.json`) and the part map (`link_map.json`) |
| `tools/` | `screw_graph.py`, `build_robot.py`, MuJoCo + Isaac checks, `check_policy_io.py` (the Pi's policy interface), Fusion add-ins |
| `validation/` | reports + renders |

Fusion documents (cloud, work › Default Project):
* `Fox_Prototype_03_v6` — original, untouched.
* `Fox_Prototype_03_v6_RL` — design copy with the fixes (mirrored front leg, rear gear pair, servos, restored tibias, Component77).
* `Fox_Prototype_03_v6_RL_Mechanism` — **current model**: 33 components (base + per leg hip, pinion, gear, femur, quad,
  tibia, foot, link), 40 as-built revolute joints (per leg `hip_joint`, `femur_joint` = pivot servo, `pinion_joint` = gear
  servo + 7 free pins, two of them closing the loops) and 4 motion links (pinion → gear, 1:1 reversed). Drag `XX_pinion_joint`
  to swing the leg; to see the pivot servo extend the leg, **lock `XX_pinion_joint`** (right-click → Lock) and drag
  `XX_femur_joint` — otherwise Fusion moves both servos together and the leg just swings.
* `Fox_Prototype_03_v6_RL_URDF_v2` — checkpoint: screw-consistent serial model (no gear relation).
* `Fox_Prototype_03_v6_RL_URDF` — first model, superseded; safe to delete.

## Conventions
* Frame: **x forward** (IMU end), **y left**, **z up**; base origin midway between the four hips at hip-axis height.
* Zero = the CAD standing pose for every servo and joint; the base origin is **0.1528 m** above the ground there.
* Legs FL, FR, RL, RR. Total mass **542.6 g** (no battery yet).

## Reduced tree (URDF / Isaac): joints and masses
| Joint | Parent → child | Axis | Origin in parent (m) | Limits (rad) | Effort (N·m) |
|---|---|---|---|---|---|
| FL_hip_joint | base → FL_hip | (0.985, 0, -0.174) | 0.071686 0.024238 0.007390 | -0.50 … 0.50 | 0.470 |
| FL_thigh_joint | FL_hip → FL_thigh | +y | 0.013073 0.017273 -0.002610 | -1.20 … 1.20 | 0.940 |
| FL_calf_joint | FL_thigh → FL_calf | +y | -0.026269 0.000000 -0.070249 | -1.20 … 1.20 | 0.470 |
| FL_foot_joint | FL_calf → FL_foot | +y | 0.005365 0.002750 -0.067459 | -3.14 … 3.14 | 10.000 |
| FR_hip_joint | base → FR_hip | (0.985, 0, -0.174) | 0.071686 -0.024262 0.007390 | -0.50 … 0.50 | 0.470 |
| FR_thigh_joint | FR_hip → FR_thigh | +y | 0.013073 -0.017248 -0.002610 | -1.20 … 1.20 | 0.940 |
| FR_calf_joint | FR_thigh → FR_calf | +y | -0.026269 0.000000 -0.070249 | -1.20 … 1.20 | 0.470 |
| FR_foot_joint | FR_calf → FR_foot | +y | 0.005365 -0.002750 -0.067459 | -3.14 … 3.14 | 10.000 |
| RL_hip_joint | base → RL_hip | (0.940, 0, 0.342) | -0.073606 0.024238 -0.000351 | -0.50 … 0.50 | 0.470 |
| RL_thigh_joint | RL_hip → RL_thigh | +y | -0.012394 0.017273 -0.004429 | -1.20 … 1.20 | 0.940 |
| RL_calf_joint | RL_thigh → RL_calf | +y | 0.038069 0.001750 -0.064620 | -1.20 … 0.74 | 0.470 |
| RL_foot_joint | RL_calf → RL_foot | +y | -0.052566 0.001000 -0.053496 | -3.14 … 3.14 | 10.000 |
| RR_hip_joint | base → RR_hip | (0.940, 0, 0.342) | -0.073382 -0.024238 -0.000188 | -0.50 … 0.50 | 0.470 |
| RR_thigh_joint | RR_hip → RR_thigh | +y | -0.012618 -0.017522 -0.004593 | -1.20 … 1.20 | 0.940 |
| RR_calf_joint | RR_thigh → RR_calf | +y | 0.038069 -0.001500 -0.064620 | -1.20 … 0.74 | 0.470 |
| RR_foot_joint | RR_calf → RR_foot | +y | -0.052566 -0.001000 -0.053496 | -3.14 … 3.14 | 10.000 |

| Link (URDF) | Parts | Mass (g) |
|---|---|---|
| base | base | 330.4 |
| FL_hip | hip, pinion, gear | 34.4 |
| FL_thigh | femur, quad | 7.5 |
| FL_calf | tibia, link | 7.8 |
| FL_foot | foot | 2.5 |
| FR_hip | hip, pinion, gear | 34.4 |
| FR_thigh | femur, quad | 7.5 |
| FR_calf | tibia, link | 8.2 |
| FR_foot | foot | 2.5 |
| RL_hip | hip, pinion, gear | 34.9 |
| RL_thigh | femur, quad | 7.8 |
| RL_calf | tibia, link | 7.4 |
| RL_foot | foot | 3.1 |
| RR_hip | hip, pinion, gear | 36.4 |
| RR_thigh | femur, quad | 7.7 |
| RR_calf | tibia, link | 7.0 |
| RR_foot | foot | 3.0 |

Parts of one leg in the exact model (`mjcf/fox.xml`):

| Part (MJCF body) | Mass (g) |
|---|---|
| base | 330.45 |
| FL_hip | 29.64 |
| FL_pinion | 2.28 |
| FL_gear | 2.52 |
| FL_femur | 4.89 |
| FL_quad | 2.61 |
| FL_tibia | 5.49 |
| FL_foot | 2.52 |
| FL_link | 2.35 |

## How the CAD maps to parts
Membership follows the fasteners (`tools/screw_graph.py` on a scan of every hole, pin and spline): parts that share a
screw line, or sit on a servo's output spline, are one rigid part; leg-linkage pins and spline axes are pivots.
* **base**: both pelvis halves, Pi mount, Raspberry Pi 4, 12-ch PWM board, 2× XL4015, Arduino Nano, BNO055, the bodies of
  the 4 hip servos, and everything screwed to them.
* **XX_hip**: the bracket riding on the hip-servo spline (front `Servo Pelv Upper/Lower(Mirror)`, rear left `Servo Pelv
  Upper`, rear right `Component40/41(Mirror) (1)`), rear caps `Body28`/`Body29` (+ `Body27`, by contact), both servo cases.
* **XX_pinion**: gear-servo spline + `m` gear. **XX_gear**: the 12-tooth gear with its crank arm.
* **XX_femur**: femur + pivot-servo spline + the gear's spacer and washer (clamped by the horn screw `Body4`, rear left).
* **XX_quad**: Quad Link. **XX_tibia**: tibia + knee pin (`Body2` rear left). **XX_foot**: foot + pad.
  **XX_link**: Component77 (front) / Calf Link (rear).
* In the tree model: hip = hip + pinion + gear, thigh = femur + Quad Link, calf = tibia + link, foot = foot.

## Modelling decisions (review)
1. **Abduction axis** = the hip-servo spline axis measured from the CAD (tilted 10° front / 20° rear from +x).
2. **Rear brackets un-joined**: `Combine1`/`Combine2` of `Component41(Mirror) (1)` and `Combine2` of `Servo Pelv Upper` join
   both rear brackets into the pelvis in the design copy; the builders suppress them **in memory only**.
3. **Limits are placeholders**: servos ±1.2 rad (hip ±0.5), knee capped by the loop dead points (above). Measure the real
   servo travel (DS-843MG ~±60°) and mechanical stops.
4. **Actuators**: DS-843MG at 6 V, 0.47 N·m stall, 10.5 rad/s; PD kp 5.0 / kd 0.08 per servo (a hobby servo reaches
   stall within ~5° of error), armature 5e-4 (estimate). The first guess, kp 2.0, was too soft: randomized robots tipped
   over just standing. The robot sags 4.7 mm when standing. Replace with bench-test values (step response) before
   sim-to-real.
5. **Isaac foot mimic** is the linear part of the four-bar: exact on the rear; front error 0.015 rad within ±0.3 rad of
   knee motion, 0.06 rad at ±0.6, 0.27 rad at ±1.2 (the MuJoCo models use the full quartic).
6. **Battery**: hidden in the CAD, not modelled. Add its mass to `base` when known.
7. **No joint damping/friction/armature in the URDF** (Isaac's importer writes URDF damping into the per-degree USD
   field); they live in `isaaclab/fox_cfg.py` and the MJCF defaults. The URDF thigh effort limit is 0.94 N·m (= pivot −
   gear torque) so PhysX does not clip the mapped torque.
8. **Loops in MuJoCo** use near-hard constraints (`solimp 0.999`): with the default the 2–3 g linkage parts let the pins
   gap 2 mm under load.

## Use
* **MuJoCo** (exact): `~/.venvs/fox_rl/bin/python -m mujoco.viewer --mjcf ~/Documents/fox_rl_model/mjcf/scene.xml`.
  Actions = 12 servo targets (`XX_hip`, `XX_pivot`, `XX_gear`), i.e. what the real servos get. Ramp large changes: a
  step on all 12 servos saturates them together and the robot can hop and flip.
* **Isaac Sim** (`~/isaacenv`, 6.0.1): `OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/isaacsim_import.py`, then
  `tools/isaacsim_stand_test.py` (servo PD applied with `set_dof_efforts`, two robots: standing + bolted in the air).
* **Isaac Lab** (installed: `~/IsaacLab`, tag `v3.0.0-beta2.patch1` = the release for Isaac Sim 6.0.1, pip-installed
  into `~/isaacenv` with torch 2.11+cu130 kept): `~/isaacenv/bin/python isaaclab/fox_sim.py --viz kit` opens the viewer
  with 4 robots, one servo motion each (crouch, sway, pendulum swing, shin swing); `--headless --duration 10` for a quick
  check. In your own tasks: `from fox_cfg import FOX_CFG`; joint position actions on `.*_hip_joint`, `.*_thigh_joint`,
  `.*_calf_joint` are servo targets (hip, pivot, gear); `FoxServoActuator.servo_angles()` gives the servo angles from
  joint angles for observations. Velocity task: base `base`, feet `.*_foot`, undesired contacts `.*_thigh|.*_calf`,
  small commands (≈0.3 m/s), target height ~0.14 m. Notes for this laptop (8 GB GPU): Isaac Lab 3.0 runs headless
  unless `--viz kit` is given; keep the Kit window small (full screen at 3200x2000 renders ~1.7 s/frame — the script
  defaults to 1280x800); the first GUI launch compiles shaders for minutes; close other GPU apps (Fusion) if RTX runs
  out of memory; PhysX GPU buffers are shrunk in `fox_sim.py` (the defaults reserve ~1 GB); don't switch "Simulation
  Output" to USD in the GUI (deadlocks Isaac Sim 6.0.1).
* Hardware: send pivot = thigh, gear = −thigh − knee (radians from the stance, then your servo's zero/scale/direction).

## RL walking + keyboard teleop (Isaac Lab)
* Task `Fox-Velocity-Flat` (`isaaclab/fox_tasks.py`): Isaac Lab's quadruped velocity task (Go1 recipe) on flat ground,
  re-scaled for this robot: actions = the 12 servo targets (±0.25 rad around the stance), commands vx ±0.3 m/s,
  vy ±0.2 m/s, yaw ±1 rad/s, 50 Hz policy / 200 Hz physics, mass +0.2 kg / COM / friction / push randomization.
  Changes that made it learn: a fall penalty + small alive bonus, a 10× smaller joint-acceleration penalty (the 2–3 g
  links accelerate a lot), initial action noise 0.5 (1.0 knocks the robot over in 0.5 s), stiffer servos (above).
* Train (headless, ~15 min for 1000 iterations on this laptop, 2048 robots):
  `cd ~/Documents/fox_rl_model && ~/isaacenv/bin/python isaaclab/fox_train.py --task Fox-Velocity-Flat --headless`
  → `logs/rsl_rl/fox_flat/<date>/model_*.pt` (TensorBoard: `logs/rsl_rl/fox_flat`).
* Hardware-ready task `Fox-Velocity-Flat-Blind`: the policy sees only what the robot has (IMU gyro + gravity direction,
  the remote's command, its own last servo commands, 5-step history); the critic sees the full state (training only).
  Gait rewards: action-rate penalty, diagonal trot (Spot's `GaitReward`), Spot's `air_time_reward` (each foot phase
  ~0.15 s; zero command = all feet down, even when drifting). `fox_train.py --task Fox-Velocity-Flat-Blind --headless`, ~30 min.
* Since v3 the blind task (training, play and eval) uses realistic servos, `FOX_SERVO_REAL_CFG`, each drawn at random
  per robot and episode:
  * 5–25 ms command latency (Pi → PCA9685 at 50 Hz → servo, sensor lag included);
  * backlash up to 0.02 rad on the hip and pivot servos, and 0.04 rad on the gear servo (the printed gear pair adds play);
  * torque that falls linearly from the stall torque at rest to zero at the no-load speed (DS-843MG: 0.47 N·m, 10.5 rad/s);
  * servo gains off by up to ±20 % (kp) and ±30 % (kd).

  These are estimates: put bench-test values into `fox_cfg.py`. The ideal servo (`FOX_SERVO_CFG`) stays in `Fox-Velocity-Flat`.
* Drive: `~/isaacenv/bin/python isaaclab/fox_play.py --checkpoint policies/fox_flat_blind_v3/model_1999.pt` → click the
  viewport, hold W/S (forward/back), A/D (sideways), Q/E (turn), L = stop; the camera follows the robot. For v1 add
  `--task Fox-Velocity-Flat-Play`. `--eval --headless --num_envs 16` prints commanded vs achieved velocities instead.
  It also exports `exported/policy.onnx` next to the checkpoint (for the Pi).
* Policies (`policies/*`: `model_*.pt`, `policy.onnx` for the Pi, `params/` = exact env + agent config), `--eval` results
  (16 robots, 7 commands × 4 s, mean |error| over the last 2 s of each):

  | Policy | Policy sees | Eval servos | Error vx / vy / yaw | Servo-command change | Swing | At zero command |
  |---|---|---|---|---|---|---|
  | `fox_flat_v1` | joint encoders + body velocity | ideal | 0.008 / 0.008 m/s / 0.014 rad/s | 0.082 rad/step | 0.072 s | steps 3.4/s, holds position |
  | `fox_flat_blind_v1` | IMU only | ideal | 0.011 / 0.019 / 0.027 | 0.067 | 0.075 s | marches 7.4/s, drifts 8 cm/s |
  | `fox_flat_blind_v2` | IMU only | ideal | 0.006 / 0.007 / 0.037 (under-turns 12 %) | 0.059 | 0.107 s | stands (1.4 steps/s, 0.2 cm/s) |
  | `fox_flat_blind_v2` | IMU only | realistic | 0.016 / 0.010 / 0.042 (0.3 → 0.26 m/s, 0.8 → 0.68 rad/s) | 0.059 | 0.104 s | stands (2.3 steps/s) |
  | **`fox_flat_blind_v3`** | IMU only | realistic | 0.008 / 0.006 / 0.025 (0.3 → 0.27 m/s, 0.8 → 0.83 rad/s) | 0.042 | 0.132 s | stands (0.5 steps/s, 0.2 cm/s) |

  No policy fell in any eval. Why each version changed:
  * Blind v1 trotted at 7 Hz. The trot term's timing error is in seconds, so short phases score best, and its
    step-length term was mostly silent (bug below).
  * v2's first try drifted at 0.13 m/s at zero command, just above the 0.1 m/s "moving" threshold, to keep the
    walking rewards. The threshold is now 1.0 m/s.
  * v3 was trained on realistic servos with a tighter yaw term (weight 1.0, std 0.3), for 2000 iterations.
  * The top speed of about 0.27 m/s is the speed-limited servos: v2 loses 15 % of its speed on them.
* Running a blind policy on the Pi (`policy.onnx`, 105 inputs → 12 outputs, no input normalization):
  * Observation: for each term, its last 5 readings, oldest first, terms in this order:
    * `[0:15]` gyro (rad/s);
    * `[15:30]` gravity direction as a unit vector, (0, 0, −1) standing level. An accelerometer or the BNO055 gravity
      output reads +z when level, so negate and normalise it;
    * `[30:45]` command (vx, vy m/s, yaw rate rad/s);
    * `[45:105]` the policy's own previous 12 outputs.
  * Frames are body frame: x forward, y left, z up (rotate the IMU's axes into it). At start, fill the history with
    the first reading 5 times.
  * Outputs: servo target = 0.25 × output (rad from the CAD stance), in this order: hip FL, FR, RL, RR, then pivot
    FL…RR, then gear FL…RR. Then apply each servo's zero, direction and µs/rad.
  * Check: `tools/check_policy_io.py <model.pt>` rebuilds this vector from raw signals, and runs the ONNX file
    against the PyTorch policy (v3: identical observation, actions within 4e-7).
* Isaac Lab 3.0-beta2 bug: `ContactSensor.compute_first_contact` never fires 2–16 s into an episode (float32 sensor time
  vs a 1e-8 tolerance), so the stock `feet_air_time` reward (still in `Fox-Velocity-Flat`) is silent most of each
  episode. The blind task uses `air_time_reward` instead, and `--eval` counts touchdowns from contact-state changes.
* Body names: feet `.*_foot`, undesired contacts `.*_thigh|.*_calf`; the USD's bodies are flattened (Isaac Lab's
  contact sensor only sees direct children of one parent; the Isaac Sim 6 importer nests them).

## Regenerate
Each `tools/fusion/*` folder is a one-shot add-in: copy it into Fusion's `API/AddIns`, start Fusion, wait for its result
in `C:\fusion_jobs` (`drive_c/fusion_jobs` in the Wine prefix), then remove the folder again (it runs at every start).
1. After editing the design copy: `FoxScrewScan` → copy `screw_scan.json` to `analysis/`, run
   `~/.venvs/fox_rl/bin/python tools/screw_graph.py` (fails on a fastener conflict or a non-parallelogram leg) and copy
   `analysis/link_map.json` into `tools/fusion/FoxBuildLinkModel/`.
2. `FoxBuildLinkModel` writes a new `…_RL_Mechanism` document (timestamped if the name exists) and saves it only if the
   drive tests pass. Wait for `Uploaded document` in the Fusion log before closing Fusion.
3. `FoxExportRaw` → copy `C:\fusion_jobs\export` to `raw/`.
4. `~/.venvs/fox_rl/bin/python tools/build_robot.py`, then the checks below.

## Validation
All checks pass (reports in `validation/`):

| Check | Result |
|---|---|
| Fasteners: every part sharing a screw line or a spline is in the same part | 380 rigid joins, 0 conflicts, 12/12 splines matched |
| Fusion mechanism: 40 joints + 4 motion links healthy; each hip joint turns its leg rigidly | rigid-rotation error 0.0 cm |
| Fusion drive tests (gear servo, held femur, held pinion, knee ±0.3): gear = −pinion, shin ∥ crank, Quad Link ∥ femur, rear foot ∥ femur | worst 7 µrad |
| Four-bar model (`build_robot.py`) vs Fusion's solver, front foot | 0.093 / −0.078 rad at knee +0.3 / −0.2 in both |
| MuJoCo exact model: structure, feet touch at the zero pose | 39 qpos, 12 servos, 12 equalities; all feet at +2 mm |
| MuJoCo exact model: stands 4 s | base 0.1528 → 0.1450 m, upright 0.999, loop gap ≤ 0.03 mm, gear error ≤ 0.0003 rad |
| MuJoCo: servo roles (table above) | gear → shin, pivot → femur/knee, pivot + gear(−) → rigid pendulum |
| MuJoCo: reduced tree vs exact mechanism (6 random servo poses, held) | feet within 0.04 mm; standing heights 0.14501 / 0.14555 m |
| MuJoCo: ramped crouch / abduction | body lowers, stays upright |
| MuJoCo: 20 s random servo targets | always finite |
| Isaac Sim 6.0.1 import → `usd/fox.usd` | 17 bodies, 16 DOFs, 4 foot mimics |
| Isaac Sim: stands 4 s with the servo map (GPU PhysX, 200 Hz) | base 0.155 → 0.14684 m, upright 0.999514 |
| Isaac Sim: servo map in PhysX (held robot) + foot mimics | gear → shin −0.30, pivot → femur +0.30, pendulum; mimic error 0.0015 rad |
| `FoxServoActuator.compute()` vs the servo map | 1e-8 N·m, power balance holds |
| `FOX_SERVO_REAL_CFG`: latency, backlash, torque-speed line (`tools/test_fox_actuator.py`) | target delayed exactly N steps, no torque within the play, half the stall torque at half the no-load speed |
| Policy interface for the Pi (`tools/check_policy_io.py`, v3) | observation rebuilt from raw signals = Isaac Lab's (0 error); ONNX = PyTorch (3.6e-7) |
| Isaac Lab 3.0.0-beta2 (`isaaclab/fox_sim.py`, 4 robots, servo motions) | runs headless and in the Kit viewer; all upright (≥ 0.945) |

Renders: `validation/mujoco_servo_roles.png` (rows: gear servo, pivot servo, pendulum), `mujoco_hip_abduction.png`,
`mujoco_front_left.png`, `mujoco_poses.png`, `isaacsim_stand.png`.
Re-run: `~/.venvs/fox_rl/bin/python tools/validate_mujoco.py`, `tools/mechanism_mujoco.py`, `tools/pose_test_mujoco.py`,
`tools/random_actions_mujoco.py`; `~/isaacenv/bin/python tools/test_fox_actuator.py`; Isaac Sim as above;
`OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv/bin/python tools/check_policy_io.py policies/fox_flat_blind_v3/model_1999.pt`.
