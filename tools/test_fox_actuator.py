#!/usr/bin/env python3
"""Check isaaclab/fox_cfg.py's FoxServoActuator.compute() without Isaac Lab installed: the few isaaclab names it imports
are replaced by minimal stand-ins, then its torques are compared with the servo map written out by hand
(the one tools/isaacsim_stand_test.py runs in PhysX). Needs torch:  ~/isaacenv/bin/python tools/test_fox_actuator.py"""
import os, sys, types
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _ActuatorBase:  # the attributes FoxServoActuator uses from isaaclab.actuators.ActuatorBase
    def __init__(self, cfg, joint_names, joint_ids, num_envs, device):
        self.cfg, self._joint_names, self._device = cfg, joint_names, device
        full = lambda v: torch.full((num_envs, len(joint_names)), float(v))  # noqa: E731
        self.stiffness, self.damping, self.effort_limit = full(cfg.stiffness), full(cfg.damping), full(cfg.effort_limit)

    @property
    def joint_names(self):
        return self._joint_names


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m


_stub('isaaclab')
_stub('isaaclab.sim', UsdFileCfg=_Cfg, RigidBodyPropertiesCfg=_Cfg, ArticulationRootPropertiesCfg=_Cfg)
_stub('isaaclab.actuators', ActuatorBase=_ActuatorBase, ActuatorBaseCfg=_Cfg)
_stub('isaaclab.assets')
_stub('isaaclab.assets.articulation', ArticulationCfg=type('ArticulationCfg', (_Cfg,), {'InitialStateCfg': _Cfg}))
_stub('isaaclab.utils', configclass=lambda c: c)
_stub('isaaclab.utils.types', ArticulationActions=_Cfg)
sys.path.insert(0, os.path.join(ROOT, 'isaaclab'))
import fox_cfg  # noqa: E402

LEGS = fox_cfg.LEGS
names = ['%s_%s_joint' % (l, p) for p in ('hip', 'thigh', 'calf') for l in LEGS]   # PhysX order: all hips, thighs, calves
rng = np.random.default_rng(0)
order = list(rng.permutation(len(names)))                                            # any order must work (name lookup)
names = [names[i] for i in order]
act = fox_cfg.FoxServoActuator(fox_cfg.FOX_SERVO_CFG, names, list(range(12)), 5, 'cpu')
q, qd, tgt = (torch.tensor(rng.uniform(-1, 1, (5, 12)), dtype=torch.float32) for _ in range(3))
out = act.compute(_Cfg(joint_positions=tgt.clone(), joint_velocities=None, joint_efforts=None), q, qd)
eff = out.joint_efforts.numpy()
kp, kd, lim = fox_cfg.FOX_SERVO_CFG.stiffness, fox_cfg.FOX_SERVO_CFG.damping, fox_cfg.FOX_SERVO_CFG.effort_limit
worst = 0.0
for e in range(5):
    for l in LEGS:
        h, t, c = (names.index('%s_%s_joint' % (l, p)) for p in ('hip', 'thigh', 'calf'))
        servo = lambda x: np.array([x[e, h], x[e, t], -(x[e, t] + x[e, c])])       # hip, pivot, gear angles
        s, v, goal = servo(q.numpy()), servo(qd.numpy()), np.array([tgt[e, h], tgt[e, t], tgt[e, c]])
        th, tp, tg = np.clip(kp * (goal - s) - kd * v, -lim, lim)
        ref = {h: th, t: tp - tg, c: -tg}
        worst = max(worst, max(abs(eff[e, j] - r) for j, r in ref.items()))
        dq = rng.normal(size=12)                                                     # virtual work: joint power = servo power
        ds = np.array([dq[h], dq[t], -(dq[t] + dq[c])])
        worst = max(worst, abs(eff[e, h] * dq[h] + eff[e, t] * dq[t] + eff[e, c] * dq[c] - (np.array([th, tp, tg]) @ ds)))
assert out.joint_positions is None and out.joint_velocities is None, 'explicit actuator must clear the position/velocity targets'
assert worst < 1e-5, 'FoxServoActuator disagrees with the servo map: %.2e' % worst
print('FoxServoActuator matches the servo map (max deviation %.1e N m, power balance included) - PASS' % worst)
