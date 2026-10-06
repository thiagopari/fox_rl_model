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

    def replace(self, **kw):
        return type(self)(**{**self.__dict__, **kw})


class _ActuatorBase:  # the attributes FoxServoActuator uses from isaaclab.actuators.ActuatorBase
    def __init__(self, cfg, joint_names, joint_ids, num_envs, device):
        self.cfg, self._joint_names, self._num_envs, self._device = cfg, joint_names, num_envs, device
        full = lambda v: torch.full((num_envs, len(joint_names)), float(v))  # noqa: E731
        self.stiffness, self.damping, self.effort_limit = full(cfg.stiffness), full(cfg.damping), full(cfg.effort_limit)
        self.velocity_limit = full(cfg.velocity_limit)

    @property
    def joint_names(self):
        return self._joint_names


class _DelayBuffer:  # isaaclab.utils.buffers.DelayBuffer: compute() returns the input of `lag` calls ago (or the oldest)
    def __init__(self, history_length, batch_size, device):
        self.lag, self.hist = torch.zeros(batch_size, dtype=torch.long), []

    def set_time_lag(self, lag, batch_ids=None):
        self.lag[slice(None) if batch_ids is None else batch_ids] = lag.long()

    def reset(self, batch_ids=None):
        self.hist = []

    def compute(self, data):
        self.hist.append(data.clone())
        k = torch.clamp(self.lag, max=len(self.hist) - 1)
        return torch.stack([self.hist[-1 - int(k[b])][b] for b in range(len(k))])


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
_stub('isaaclab.utils.buffers', DelayBuffer=_DelayBuffer)
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

# FOX_SERVO_REAL_CFG: latency, backlash, torque-speed line. Servo-space checks on one leg (FL), joints at rest.
real = fox_cfg.FOX_SERVO_REAL_CFG
h, t, c = (names.index('FL_%s_joint' % p) for p in ('hip', 'thigh', 'calf'))
lag = 3
act = fox_cfg.FoxServoActuator(real.replace(min_delay=lag, max_delay=lag), names, list(range(12)), 1, 'cpu')
act.reset(None)
zero, kp = torch.zeros(1, 12), real.stiffness
hip_torque = lambda goal, q=zero, qd=zero: float(act.compute(  # noqa: E731
    _Cfg(joint_positions=goal, joint_velocities=None, joint_efforts=None), q, qd).joint_efforts[0, h])
goals = [torch.zeros(1, 12) for _ in range(6)]
for k, g in enumerate(goals):
    g[0, h] = 0.01 * (k + 1)                                     # a ramp on the FL hip target
seen = [hip_torque(g) for g in goals]
play = float(act._play[0, LEGS.index('FL')])                    # this episode's hip backlash (hip servos come first)
want = [max(0.01 * (max(k - lag, 0) + 1) - play, 0.0) * kp for k in range(6)]   # first `lag` steps: the oldest target
assert np.allclose(seen, want, atol=1e-6), ('latency/backlash', seen, want)
assert 0.0 <= play <= real.backlash[0]
act.reset(None)                                                   # torque-speed: big error, servo already turning
for _ in range(lag + 1):
    out = act.compute(_Cfg(joint_positions=torch.full((1, 12), 5.0), joint_velocities=None, joint_efforts=None),
                      zero, torch.where(torch.arange(12) == h, real.velocity_limit / 2, 0.0)[None].float())
assert abs(float(out.joint_efforts[0, h]) - real.effort_limit / 2) < 1e-6, 'half the stall torque at half the no-load speed'
print('FOX_SERVO_REAL_CFG: %d-step latency, %.4f rad backlash, torque-speed line - PASS' % (lag, play))
