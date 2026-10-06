"""Isaac Lab articulation config for the Fox quadruped (drop-in analogue of UNITREE_GO1_CFG).

Usage inside an Isaac Lab task (tested pattern: isaaclab_tasks/manager_based/locomotion/velocity/config/go1):
    from fox_cfg import FOX_CFG
    self.scene.robot = FOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
Body names for the velocity task: base = "base", feet = ".*_calf", undesired contacts = ".*_thigh".
Convert the URDF first (Isaac Lab >= 2.x):
    ./isaaclab.sh -p scripts/tools/convert_urdf.py ~/Documents/fox_rl_model/urdf/fox.urdf \
        ~/Documents/fox_rl_model/usd/fox.usd --merge-joints --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none
(or use the USD already produced by tools/isaacsim_import.py if present).
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

FOX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Corona DS-843MG at 6 V: 0.47 N m stall torque, 0.10 s / 60 deg (10.5 rad/s). The hip-pitch servo drives the femur
# directly; the knee is driven through a gear pair + crank linkage (ratio not modelled -> kept at the servo values).
FOX_SERVO_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    saturation_effort=0.47,
    effort_limit=0.30,      # continuous-ish limit (~65% of stall); tune after bench tests
    velocity_limit=10.5,
    stiffness=2.0,          # PD gains used by the MuJoCo validation (N m / rad, N m s / rad)
    damping=0.05,
    friction=0.0,
    armature=0.0005,        # reflected rotor inertia estimate for a geared hobby servo (helps solver stability)
)

FOX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(FOX_DIR, "usd", "fox.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.16),          # base origin is 0.1528 m above the feet at the zero (standing) pose
        joint_pos={".*": 0.0},          # zero = the CAD standing pose
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"legs": FOX_SERVO_CFG},
)
"""Fox quadruped: 13 links (base + {FL,FR,RL,RR}_{hip,thigh,calf}), 12 revolute joints, 0.542 kg (no battery yet)."""
