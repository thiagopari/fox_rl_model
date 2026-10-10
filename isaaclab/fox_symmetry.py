"""Left-right symmetry of the Fox blind tasks' observations and actions, for RSL-RL's symmetry augmentation (each batch
is trained on itself and on its mirror image, so left and right legs learn mirror-image motions). The left and right
legs share joint conventions (servo map, stance offsets), so mirroring swaps FL <-> FR and RL <-> RR and negates the hip
(abduction) angles; in the body frame y flips: velocities (x, -y, z), rates (-x, y, -z), gravity (x, -y, z), command
(vx, -vy, -yaw). Front-back is no symmetry here: front and rear legs differ. Terms are found by name in the
observation manager, so history stacking does not matter."""
import torch

LR = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
VEC = {"base_lin_vel": (1.0, -1.0, 1.0), "base_ang_vel": (-1.0, 1.0, -1.0), "projected_gravity": (1.0, -1.0, 1.0),
       "velocity_commands": (1.0, -1.0, -1.0)}
_cache = {}


def _perm(names, device):
    """Index of each name's mirror partner and the sign (-1 for hips)."""
    return (torch.tensor([names.index(LR[n[:2]] + n[2:]) for n in names], device=device),
            torch.tensor([-1.0 if "_hip" in n else 1.0 for n in names], device=device))


def _maps(env):
    u = env.unwrapped
    if id(u) not in _cache:
        joints = _perm(list(u.scene["robot"].joint_names), u.device)
        actions = _perm(list(u.action_manager.get_term("joint_pos")._joint_names), u.device)
        om, layout = u.observation_manager, {}
        for group in om.active_terms:
            start, terms = 0, []
            for name, dim in zip(om.active_terms[group], om.group_obs_term_dim[group]):
                size = int(torch.tensor(dim).prod())
                terms.append((name, start, size))
                start += size
            layout[group] = terms
        _cache[id(u)] = joints, actions, layout
    return _cache[id(u)]


def mirror_group(env, group, obs):
    """The mirror image of one observation group's tensor (num_envs, dim)."""
    (jp, js), (ap, asg), layout = _maps(env)
    out, n = obs.clone(), obs.shape[0]
    for name, start, size in layout[group]:
        block = obs[:, start:start + size]
        if name in VEC:
            out[:, start:start + size] = (block.view(n, -1, 3) * torch.tensor(VEC[name], device=obs.device)).view(n, size)
        elif name in ("joint_pos", "joint_vel"):
            out[:, start:start + size] = (block.view(n, -1, len(jp))[:, :, jp] * js).view(n, size)
        elif name == "actions":
            out[:, start:start + size] = (block.view(n, -1, len(ap))[:, :, ap] * asg).view(n, size)
        else:
            raise KeyError("fox_symmetry: no mirror rule for observation term %r" % name)
    return out


@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
    """RSL-RL data augmentation: the batch, then its left-right mirror image."""
    obs_aug = actions_aug = None
    if obs is not None:
        b = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            obs_aug[group][b:] = mirror_group(env, group, obs[group])
    if actions is not None:
        _, (ap, asg), _ = _maps(env)
        actions_aug = torch.cat([actions, actions[:, ap] * asg], dim=0)
    return obs_aug, actions_aug
