"""Action and command terms of the Fox blind task (fox_tasks.py names them by string, so this module is imported only
once the simulator runs: these subclasses import USD)."""
import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def hips_free(command: torch.Tensor) -> torch.Tensor:
    """(N, 1) True when a command asks for sideways or turning motion. Otherwise (forward / backward / standing) the hip
    servos hold the stance. deploy/fox_pi.py applies the same rule on the robot."""
    return (command[:, 1:].abs() > 1e-3).any(dim=1, keepdim=True)


class FoxServoAction(mdp.JointPositionAction):
    """Servo targets as JointPositionAction, with the hip targets held at the stance (0) for straight commands. The
    stored raw actions (what last_action(action_name=...) observes) are the ones the servos were sent."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._hips = [i for i, n in enumerate(self._joint_names) if n.endswith("_hip_joint")]

    def process_actions(self, actions: torch.Tensor):
        actions = actions.clone()
        free = hips_free(self._env.command_manager.get_command("base_velocity"))
        actions[:, self._hips] = torch.where(free, actions[:, self._hips], 0.0)
        super().process_actions(actions)


class FoxVelocityCommand(mdp.UniformVelocityCommand):
    """Isaac Lab's velocity command plus straight ones (forward / backward only), which uniform sampling never hits."""

    STRAIGHT_SHARE = 0.3

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device)
        self.vel_command_b[ids[torch.rand(len(ids), device=self.device) < self.STRAIGHT_SHARE], 1:] = 0.0

