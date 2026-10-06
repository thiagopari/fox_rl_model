"""Isaac Lab articulation config for the Fox quadruped (analogue of UNITREE_GO1_CFG), driven like the real robot.

Each leg has three DS-843MG servos: hip (abduction), pivot (turns the femur) and gear (pinion -> 12:12 gear -> crank ->
Quad Link; the hip-crank-Quad Link-knee parallelogram keeps the shin at the crank's angle). urdf/fox.urdf is that
mechanism reduced to a tree (base + per leg hip, thigh, calf, foot); the servo map makes it exact:
    thigh = pivot        calf = -gear - pivot        foot = mimic of calf (foot four-bar, linear part)
So the gear servo alone swings the shin, the pivot servo alone bends the knee (extension), and pivot +a with gear -a
swings the whole leg like a pendulum.

FoxServoActuator reads the position targets of <leg>_hip_joint / _thigh_joint / _calf_joint as the HIP / PIVOT /
GEAR SERVO angles (rad from the CAD stance), runs one saturated PD per servo, and maps the servo torques onto the
joints (virtual work):  tau_hip = t_hip,  tau_thigh = t_pivot - t_gear,  tau_calf = -t_gear.
The foot joints carry no actuator; their <mimic> constraint (NewtonMimicAPI in the USD) moves them.
Body names for the velocity task: base = "base", feet = ".*_foot", undesired contacts = ".*_thigh|.*_calf".
FOX_SERVO_REAL_CFG adds what the hardware does on top (FOX_SERVO_CFG is the ideal servo): command latency, gear backlash
(free play) and the DC motor's torque-speed line, each episode drawn at random within the cfg's bounds.

USD: tools/isaacsim_import.py writes usd/fox.usd (Isaac Sim 6.0.1 URDF importer, foot drives zeroed). Runs under
Isaac Lab 3.0.0-beta2 (isaaclab/fox_sim.py); the servo map is also checked in MuJoCo (tools/mechanism_mujoco.py), Isaac Sim
(tools/isaacsim_stand_test.py), and this class's compute() by tools/test_fox_actuator.py.
"""
from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.types import ArticulationActions

FOX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGS = ("FL", "FR", "RL", "RR")


class FoxServoActuator(ActuatorBase):
    """Hip, pivot and gear servos of every leg; targets on hip/thigh/calf joints are servo angles (see module doc)."""

    cfg: FoxServoActuatorCfg
    is_implicit_model = False

    def __init__(self, cfg: FoxServoActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        names = list(self.joint_names)
        pick = lambda part: [names.index("%s_%s_joint" % (leg, part)) for leg in LEGS]  # noqa: E731
        self._hip, self._thigh, self._calf = pick("hip"), pick("thigh"), pick("calf")
        self._delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._play = torch.zeros(self._num_envs, 3 * len(LEGS), device=self._device)   # backlash of each servo, this episode
        self._play_max = torch.tensor(cfg.backlash, device=self._device).repeat_interleave(len(LEGS))

    def reset(self, env_ids):
        n = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._delay.set_time_lag(torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (n,), dtype=torch.int,
                                               device=self._device), env_ids)
        self._delay.reset(env_ids)
        self._play[ids] = torch.rand(n, 3 * len(LEGS), device=self._device) * self._play_max

    def servo_angles(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """(num_envs, 12) servo angles [hip, pivot, gear] x legs from joint angles (also for observations / hardware)."""
        return torch.cat([joint_pos[:, self._hip], joint_pos[:, self._thigh],
                          -(joint_pos[:, self._thigh] + joint_pos[:, self._calf])], dim=1)

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> ArticulationActions:
        tgt = self._delay.compute(control_action.joint_positions)      # command latency (physics steps)
        idx = self._hip + self._thigh + self._calf      # servo k drives through joint idx[k] (gains/limits per servo)
        kp, kd, lim, vmax = self.stiffness[:, idx], self.damping[:, idx], self.effort_limit[:, idx], self.velocity_limit[:, idx]
        s, v = self.servo_angles(joint_pos), self.servo_angles(joint_vel)
        e = tgt[:, idx] - s
        t = torch.where(e.abs() < self._play, 0.0, kp * (e - torch.sign(e) * self._play) - kd * v)   # free play: no torque
        if self.cfg.torque_speed:                       # DC motor: stall torque at rest, none at the no-load speed
            t = torch.clamp(t, -lim * torch.clamp(1.0 + v / vmax, 0.0, 1.0), lim * torch.clamp(1.0 - v / vmax, 0.0, 1.0))
        else:
            t = torch.clamp(t, -lim, lim)
        n = len(LEGS)
        t_hip, t_pivot, t_gear = t[:, :n], t[:, n:2 * n], t[:, 2 * n:]
        effort = torch.zeros_like(joint_pos)
        effort[:, self._hip] = t_hip
        effort[:, self._thigh] = t_pivot - t_gear
        effort[:, self._calf] = -t_gear
        self.computed_effort = effort
        self.applied_effort = effort
        control_action.joint_efforts = effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class FoxServoActuatorCfg(ActuatorBaseCfg):
    class_type: type = FoxServoActuator
    min_delay: int = 0                  # command latency in physics steps (5 ms), drawn per episode in [min, max]
    max_delay: int = 0
    backlash: tuple = (0.0, 0.0, 0.0)   # max free play of the hip / pivot / gear servos (rad), drawn per episode in [0, max]
    torque_speed: bool = False          # limit torque by speed: stall torque at rest, 0 at velocity_limit (no-load speed)


# Corona DS-843MG at 6 V: 0.47 N m stall, 0.10 s / 60 deg (10.5 rad/s). The 12:12 gear passes the gear servo's torque
# to the crank unchanged; the parallelogram passes it to the shin.
FOX_SERVO_CFG = FoxServoActuatorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=0.47,      # per servo (stall); lower it for a continuous rating after bench tests
    effort_limit_sim=2.0,   # PhysX joint clip above the mapped torques (thigh gets pivot - gear: up to 0.94 N m)
    velocity_limit=10.5,
    stiffness=5.0,          # servo PD gains (N m / rad, N m s / rad), same as the MuJoCo models: a DS-843MG reaches
    damping=0.08,           # stall within ~5 deg of error (~5 N m/rad); kp 2 was too soft (randomized robots tipped over)
    friction=0.0,
    armature=0.0005,        # reflected rotor inertia estimate for a geared hobby servo (helps solver stability)
)
# The hardware, as far as it is known before bench tests (measure and set these): Pi -> PCA9685 at 50 Hz -> digital servo
# is 5-25 ms of command latency (sensor lag included); metal servo gears have ~1 deg of play, and the printed 12T gear
# pair adds ~1 deg on the gear servo.
FOX_SERVO_REAL_CFG = FOX_SERVO_CFG.replace(min_delay=1, max_delay=5, backlash=(0.02, 0.02, 0.04), torque_speed=True)

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
        joint_pos={".*": 0.0},          # zero = the CAD standing pose (servo angles 0 too)
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"servos": FOX_SERVO_CFG},
)
"""Fox quadruped: 17 links (base + {FL,FR,RL,RR}_{hip,thigh,calf,foot}), 16 revolute joints (4 foot mimics), 0.54 kg."""
