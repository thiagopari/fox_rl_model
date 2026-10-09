"""Fox walking task for Isaac Lab 3.0-beta2: flat ground, track a (vx, vy, yaw-rate) command.

Registers  Fox-Velocity-Flat             training (2048 robots, randomized); policy sees joint states + body velocity
           Fox-Velocity-Flat-Blind       hardware-ready: the policy sees only the IMU (gyro, gravity), the command and
                                         its own last servo commands (5-step history); realistic servos (latency,
                                         backlash, torque-speed line, gain spread); smooth trot, stands at zero command
           Fox-V7-Velocity-Flat-Blind    the same on the v7 robot (2:1 gear servo drive, no spacer): model in v7/
                                         (tools/build_robot.py --out v7; its USD from tools/isaacsim_import.py)
           Fox-V7-...-Blind-Straight     v7, forward / backward only, hips held at the stance (no hip servos needed)
           Fox-...-Play variants         driving / evaluation (few robots, no pushes or noise)
Built on Isaac Lab's quadruped velocity task (the Go1 recipe), re-scaled for a 0.54 kg, 15 cm tall robot. The policy
outputs the 12 SERVO targets (hip, pivot, gear servo angles around the stance, scale 0.25 rad; v7 gear 0.5) that FoxServoActuator
(fox_cfg.py) turns into joint torques through the leg mechanism. 50 Hz policy, 200 Hz physics.
Train with isaaclab/fox_train.py, drive with isaaclab/fox_play.py.
"""
import gymnasium as gym

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.string import ResolvableString
from isaaclab_physx.physics import PhysxCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import os

from fox_cfg import FOX_CFG, FOX_DIR, FOX_SERVO_REAL_CFG, fox_model_cfg, geared

SERVO_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]   # = hip / pivot / gear servo targets


@configclass
class FoxFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg(  # sized for ~2k small robots on an 8 GB GPU
        gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**18, gpu_found_lost_pairs_capacity=2**20,
        gpu_found_lost_aggregate_pairs_capacity=2**22, gpu_total_aggregate_pairs_capacity=2**20))

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs, self.scene.env_spacing = 2048, 1.0
        self.scene.robot = FOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # flat ground, no height scan, no terrain curriculum
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # actions: servo targets around the stance
        self.actions.joint_pos.joint_names = SERVO_JOINTS
        self.actions.joint_pos.scale = 0.25
        # commands sized for the robot; yaw rate is commanded directly (what a remote sends)
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (-0.3, 0.3), (-0.2, 0.2), (-1.0, 1.0)
        cmd.heading_command, cmd.rel_heading_envs, cmd.rel_standing_envs = False, 0.0, 0.1
        # randomization scaled to a 0.54 kg robot (+0.2 kg covers the battery that is not modelled yet)
        self.events.add_base_mass.params.update({"mass_distribution_params": (-0.03, 0.2), "operation": "add", "distribution": "uniform"})
        self.events.base_com.default.params["com_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)}  # a backend preset
        self.events.reset_base.params["pose_range"] = {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)}
        self.events.reset_base.params["velocity_range"] = {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.events.push_robot.params["velocity_range"] = {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        # rewards: tracking tolerance matched to +-0.3 m/s commands; small servo torques. A fall costs a lot: with only
        # penalties early on, the first run learned to fall within 0.3 s to end the episode
        self.rewards.track_lin_vel_xy_exp.weight, self.rewards.track_lin_vel_xy_exp.params["std"] = 1.5, 0.25
        self.rewards.track_ang_vel_z_exp.weight, self.rewards.track_ang_vel_z_exp.params["std"] = 0.75, 0.4
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.5)
        # the Go1 joint-acceleration weight dominated everything here: 2-3 g links accelerate a lot (-0.02/step)
        self.rewards.dof_acc_l2.weight = -2.5e-8
        self.rewards.dof_torques_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.2           # short legs, short steps
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh", ".*_calf"]
        # terminations: body touching the ground (default body name "base" already matches)


@configclass
class FoxFlatEnvCfg_PLAY(FoxFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs, self.scene.env_spacing = 4, 1.0
        self.episode_length_s = 1.0e6                                   # drive as long as you like
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class FoxFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "fox_flat"
    actor = RslRlMLPModelCfg(hidden_dims=[128, 128, 128], activation="elu", obs_normalization=False,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5))   # 1.0 knocks the light robot over in 0.5 s
    critic = RslRlMLPModelCfg(hidden_dims=[128, 128, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                     gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)


@configclass
class FoxBlindObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """What the real robot has: the BNO055 IMU, the remote's command and the servo commands it sent itself."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})   # as sent: hips 0 when held

        def __post_init__(self):
            self.enable_corruption, self.concatenate_terms = True, True
            self.history_length = 5      # 0.1 s of past readings stands in for the missing joint encoders

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged (training only): the full state helps the critic judge a blind actor."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption, self.concatenate_terms = False, True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class FoxBlindEnvCfg(FoxFlatEnvCfg):
    observations: FoxBlindObsCfg = FoxBlindObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # smooth, servo-friendly commands and a clean trot (v1 shuffled: steps < 0.2 s, jittery targets)
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # Spot's air-time term instead of feet_air_time: each foot phase should last ~0.15 s (a ~3 Hz trot), and at zero
        # command all feet stay down. Blind v1 (policies/fox_flat_blind_v1) trotted at 7 Hz, 0.06 s swings, because the gait
        # term's error is in seconds so short phases score best, and it marched in place when told to stop (drift 7 cm/s).
        # (Isaac Lab's feet_air_time also goes silent 2-16 s into each episode: float32 sensor time.) velocity_threshold
        # 1.0 m/s, beyond the Fox's speed: zero command always means stand; at 0.1 the policy drifted at 0.13 m/s to keep
        # collecting the walking rewards
        self.rewards.feet_air_time = None
        self.rewards.air_time = RewTerm(func=spot_mdp.air_time_reward, weight=3.0, params={
            "mode_time": 0.15, "velocity_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
        self.rewards.gait = RewTerm(func=spot_mdp.GaitReward, weight=1.0, params={
            "std": 0.1, "max_err": 0.2, "velocity_threshold": 1.0,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),   # diagonal pairs = trot
            "asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces")})
        self.commands.base_velocity.rel_standing_envs = 0.2                                # more practice standing still
        # the servos as the hardware has them (fox_cfg.FOX_SERVO_REAL_CFG): 5-25 ms latency, gear backlash, torque falling
        # with speed, and each servo's gains off by up to 20-30 % (unknown internal controller), redrawn every episode
        self.scene.robot.actuators = {"servos": FOX_SERVO_REAL_CFG}
        self.events.servo_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="reset", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=SERVO_JOINTS), "operation": "scale", "distribution": "uniform",
            "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.7, 1.3)})
        # v2 under-turned by 12 % (0.8 -> 0.71 rad/s): a 0.1 rad/s yaw error cost only 6 % of a lenient term
        # v4: forward / backward / standing without moving the hips (FoxServoAction), trained on 30 % straight commands
        # (fox_mdp.py is imported by name when the env is built: subclassing Isaac Lab's action / command terms imports
        # USD, which must not load before the simulator starts, and fox_train.py imports this file before that)
        self.actions.joint_pos.class_type = ResolvableString("fox_mdp:FoxServoAction")
        self.commands.base_velocity.class_type = ResolvableString("fox_mdp:FoxVelocityCommand")
        self.rewards.track_ang_vel_z_exp.weight, self.rewards.track_ang_vel_z_exp.params["std"] = 1.0, 0.3


@configclass
class FoxBlindV7EnvCfg(FoxBlindEnvCfg):
    """Blind task on the v7 model: its gear ratio, and its level stance as reset pose and as the servo targets' zero.
    The gear servos' actions are scaled by N (0.5 rad at 2:1): N times the servo angle moves the shin as one did at 1:1."""

    def __post_init__(self):
        super().__post_init__()
        robot, offsets = fox_model_cfg(os.path.join(FOX_DIR, "v7"))
        n = robot.actuators["servos"].gear_ratio
        self.scene.robot = robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = {"servos": geared(FOX_SERVO_REAL_CFG, n)}
        self.actions.joint_pos.use_default_offset = False
        self.actions.joint_pos.offset = offsets
        self.actions.joint_pos.scale = {".*_hip_joint": 0.25, ".*_thigh_joint": 0.25, ".*_calf_joint": 0.25 * n}
        # servo targets stay where the robot can follow (rad from home; policy.json carries them to deploy/fox_pi.py):
        # hips their joint range, pivots fox_pi's +-50 deg default, gear servos +-90 deg (their travel, home at 90)
        self.actions.joint_pos.clip = {".*_hip_joint": (-0.5, 0.5), ".*_thigh_joint": (-0.8727, 0.8727),
                                       ".*_calf_joint": (-1.5708, 1.5708)}
        # hold the stance: without this the v7 policy stood 33 mm crouched with the rear gear servos parked on their -90 deg
        # clamp and stepped in place (2.8 steps/foot/s). Spot's joint position penalty: joint-angle distance from the
        # stance (the default joint pos), 5x when commanded to stand
        self.rewards.joint_pos = RewTerm(func=spot_mdp.joint_position_penalty, weight=-0.2, params={
            "asset_cfg": SceneEntityCfg("robot"), "stand_still_scale": 5.0, "velocity_threshold": 0.1})


@configclass
class FoxBlindV7StraightEnvCfg(FoxBlindV7EnvCfg):
    """v7, forward / backward only: sideways and turning commands are 0, so FoxServoAction holds the hips at the stance
    on every step. For the first runs on the robot, without working hip servos."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


def _blind_play(cfg):
    cfg.scene.num_envs, cfg.scene.env_spacing = 4, 1.0
    cfg.episode_length_s = 1.0e6
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class FoxBlindEnvCfg_PLAY(FoxBlindEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _blind_play(self)


@configclass
class FoxBlindV7EnvCfg_PLAY(FoxBlindV7EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _blind_play(self)


@configclass
class FoxBlindV7StraightEnvCfg_PLAY(FoxBlindV7StraightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _blind_play(self)


@configclass
class FoxBlindPPORunnerCfg(FoxFlatPPORunnerCfg):
    max_iterations = 2000                    # the realistic servos take longer to learn
    experiment_name = "fox_flat_blind"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}


@configclass
class FoxBlindV7PPORunnerCfg(FoxBlindPPORunnerCfg):
    experiment_name = "fox_flat_blind_v7"


@configclass
class FoxBlindV7StraightPPORunnerCfg(FoxBlindPPORunnerCfg):
    experiment_name = "fox_flat_blind_v7_straight"


for _id, _cfg, _agent in (("Fox-Velocity-Flat", "FoxFlatEnvCfg", "FoxFlatPPORunnerCfg"),
                          ("Fox-Velocity-Flat-Play", "FoxFlatEnvCfg_PLAY", "FoxFlatPPORunnerCfg"),
                          ("Fox-Velocity-Flat-Blind", "FoxBlindEnvCfg", "FoxBlindPPORunnerCfg"),
                          ("Fox-Velocity-Flat-Blind-Play", "FoxBlindEnvCfg_PLAY", "FoxBlindPPORunnerCfg"),
                          ("Fox-V7-Velocity-Flat-Blind", "FoxBlindV7EnvCfg", "FoxBlindV7PPORunnerCfg"),
                          ("Fox-V7-Velocity-Flat-Blind-Play", "FoxBlindV7EnvCfg_PLAY", "FoxBlindV7PPORunnerCfg"),
                          ("Fox-V7-Velocity-Flat-Blind-Straight", "FoxBlindV7StraightEnvCfg", "FoxBlindV7StraightPPORunnerCfg"),
                          ("Fox-V7-Velocity-Flat-Blind-Straight-Play", "FoxBlindV7StraightEnvCfg_PLAY", "FoxBlindV7StraightPPORunnerCfg")):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{__name__}:{_cfg}", "rsl_rl_cfg_entry_point": f"{__name__}:{_agent}"})
