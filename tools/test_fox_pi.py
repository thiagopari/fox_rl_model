#!/usr/bin/env python3
"""Check deploy/fox_pi.py's policy contract and gear-ratio handling, no hardware or Isaac Lab needed: FoxPolicy's servo
targets = offset + scale * output with the hips held for straight commands, the safety clamp scales with each servo's
action scale, no policy.json means the v1-v4 contract, servo_calib.json's gear_ratio widens the gear servos' default
range, and at 2:1 the leg IK still shortens a leg by exactly 10 mm (ankle above the same point) with twice the gear
angle (a front leg can lengthen only 2.2 mm from home).
    ~/.venvs/fox_rl/bin/python tools/test_fox_pi.py"""
import json, os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'deploy'))
import fox_pi  # noqa: E402

LEVEL, STRAIGHT, SIDEWAYS = np.array([0.0, 0.0, -1.0]), np.array([0.2, 0.0, 0.0]), np.array([0.0, 0.1, 0.0])
with tempfile.TemporaryDirectory() as d:
    c = fox_pi.policy_contract(os.path.join(d, 'policy.onnx'))
    assert c['scale'] == [0.25] * 12 and c['offset_rad'] == [0.0] * 12 and c['gear_ratio'] == 1.0, 'v1-v4 contract'
    with open(os.path.join(d, 'policy.json'), 'w') as f:
        json.dump({'scale': [0.25] * 8 + [0.5] * 4, 'offset_rad': [0.01 * i for i in range(12)], 'gear_ratio': 2.0,
                   'command_ranges': {'lin_vel_x': [-0.3, 0.3], 'lin_vel_y': [0.0, 0.0], 'ang_vel_z': [0.0, 0.0]}}, f)
    c = fox_pi.policy_contract(os.path.join(d, 'policy.onnx'))
    assert c['gear_ratio'] == 2.0 and c['servos'] == fox_pi.SERVOS
scale, offset, out = np.array(c['scale']), np.array(c['offset_rad']), np.linspace(-1.0, 1.0, 12)
pol = fox_pi.FoxPolicy(lambda obs: out, c['scale'], c['offset_rad'])
assert np.allclose(pol.reset(np.zeros(3), LEVEL, STRAIGHT), offset + scale * np.r_[np.zeros(4), out[4:]]), 'hips held'
assert np.allclose(pol.step(np.zeros(3), LEVEL, SIDEWAYS), offset + scale * out), 'hips free'
assert np.allclose(pol.clip(np.full(12, 10.0)) - offset, 0.8 * scale / 0.25), 'clamp: 0.8 rad at 0.25, 1.6 at 0.5'
lim = fox_pi.FoxPolicy(lambda obs: np.full(12, 9.0), c['scale'], c['offset_rad'], [[-0.5, 0.3]] * 12)
assert np.allclose(lim.reset(np.zeros(3), LEVEL, SIDEWAYS), 0.3), 'training limits replace the safety span'
keys = fox_pi.Keys(c['command_ranges'])
keys.close()
assert np.allclose(keys.lo, [-0.3, 0.0, 0.0]) and np.allclose(keys.hi, [0.3, 0.0, 0.0]), 'command ranges'

with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, 'servo_calib.json'), 'w') as f:
        json.dump({'home_deg': {'RR_gear': 90.0, 'RR_pivot': 90.0}, 'gear_ratio': 2.0}, f)
    fox_pi.load_calib(os.path.join(d, 'servo_calib.json'))
assert fox_pi.GEAR_RATIO == 2.0 and fox_pi.LIMIT_DEG['RR_gear'] == (-10.0, 190.0) and fox_pi.LIMIT_DEG['RR_pivot'] == (40.0, 140.0)
for leg in ('FL', 'RR'):
    x0, z0 = fox_pi.leg_fk(leg, 0.0, 0.0)
    piv, gear = fox_pi.leg_ik(leg, -10.0)
    x, z = fox_pi.leg_fk(leg, piv, gear)
    assert abs(x - x0) < 1e-6 and abs(z - (z0 + 10.0)) < 1e-6, 'IK at 2:1'
    fox_pi.GEAR_RATIO = 1.0
    p1, g1 = fox_pi.leg_ik(leg, -10.0)
    fox_pi.GEAR_RATIO = 2.0
    assert abs(p1 - piv) < 1e-12 and abs(2.0 * g1 - gear) < 1e-12, '2:1: same pivot angle, twice the gear servo angle'
print('fox_pi: policy contract, hips held when straight, per-servo clamp, command ranges, gear ratio (limits, IK) - PASS')
