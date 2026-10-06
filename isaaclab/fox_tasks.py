"""Fox walking task for Isaac Lab 3.0-beta2: flat ground, track a (vx, vy, yaw-rate) command.

Registers  Fox-Velocity-Flat       training (2048 robots, randomized)
           Fox-Velocity-Flat-Play  driving / evaluation (few robots, no pushes or noise)
Built on Isaac Lab's quadruped velocity task (the Go1 recipe), re-scaled for a 0.54 kg, 15 cm tall robot. The policy
outputs the 12 SERVO targets (hip, pivot, gear servo angles around the stance, scale 0.25 rad) that FoxServoActuator
(fox_cfg.py) turns into joint torques through the leg mechanism. 50 Hz policy, 200 Hz physics.
Train with isaaclab/fox_train.py, drive with isaaclab/fox_play.py.
"""
import gymnasium as gym

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from fox_cfg import FOX_CFG

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


for _id, _cfg in (("Fox-Velocity-Flat", "FoxFlatEnvCfg"), ("Fox-Velocity-Flat-Play", "FoxFlatEnvCfg_PLAY")):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{__name__}:{_cfg}", "rsl_rl_cfg_entry_point": f"{__name__}:FoxFlatPPORunnerCfg"})
