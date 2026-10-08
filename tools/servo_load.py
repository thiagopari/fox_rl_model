#!/usr/bin/env python3
"""Static servo load per leg at the CAD pose and at the level stance of a model built by tools/build_robot.py.

A vertical ground force F on each foot's sole point (the robot's weight shared by the two stance feet of a trot, times a
dynamic factor) gives joint torques J^T F on the reduced tree (the foot follows the calf through the four-bar); the servo
map turns them into servo torques:  pivot = tau_thigh - tau_calf  (= F x knee offset from the hip),
gear = -tau_calf / N  (= F x sole offset from the knee / N). Printed as % of the DS-843MG stall torque.
Usage:  ~/.venvs/fox_rl/bin/python tools/servo_load.py [model dir: . (v6) | v7] [--mass 0.75] [--factor 1.5]
"""
import argparse, json, os
import numpy as np
import mujoco

STALL, G = 0.47, 9.81
LEGS = ('FL', 'FR', 'RL', 'RR')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', nargs='?', default='.')
    ap.add_argument('--mass', type=float, default=0.75, help='kg, with the battery')
    ap.add_argument('--factor', type=float, default=1.5, help='dynamic factor on the trot load (2 feet down)')
    a = ap.parse_args()
    mech = json.load(open(os.path.join(a.model, 'mechanism.json')))
    n = float(mech['servo_map'].get('pinion_per_gear', 1.0))
    m = mujoco.MjModel.from_xml_path(os.path.join(a.model, 'mjcf', 'fox_reduced.xml'))
    d = mujoco.MjData(m)
    F = a.mass * G / 2 * a.factor
    adr = lambda j: m.jnt_qposadr[m.joint(j).id]                       # noqa: E731
    dof = lambda j: m.jnt_dofadr[m.joint(j).id]                        # noqa: E731
    sole = {}
    for leg in LEGS:                                                    # sole = centre of the flat bottom at the CAD pose
        g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(leg + '_foot').id and m.geom_contype[i])
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        mujoco.mj_kinematics(m, d)
        w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        flat = w[:, 2] < w[:, 2].min() + 0.0005                  # the sole is flat: centre of its lowest 0.5 mm
        c = w[flat].mean(axis=0)
        c[2] = w[:, 2].min()
        sole[leg] = (g, d.geom_xmat[g].reshape(3, 3).T @ (c - d.geom_xpos[g]))
    poses = {'CAD pose': {}, 'level stance': mech['stance']['joints_rad']}
    print('%s: gear %g:1, F = %.2f N per stance foot (%.2f kg, factor %.1f)' % (os.path.abspath(a.model), n, F, a.mass, a.factor))
    for name, q in poses.items():
        d.qpos[:] = m.qpos0
        for j, v in q.items():
            d.qpos[adr(j)] = v
        mujoco.mj_forward(m, d)
        rows, worst = [], 0.0
        for leg in LEGS:
            g, v = sole[leg]
            p = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ v
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, p, m.body(leg + '_foot').id)
            tau = jp.T @ np.array([0.0, 0.0, F])
            dfoot = np.polynomial.polynomial.polyval(d.qpos[adr(leg + '_calf_joint')],
                                                     np.polynomial.polynomial.polyder(mech['legs'][leg]['foot_poly']))
            t_thigh, t_calf = tau[dof(leg + '_thigh_joint')], tau[dof(leg + '_calf_joint')] + dfoot * tau[dof(leg + '_foot_joint')]
            pivot, gear = t_thigh - t_calf, -t_calf / n
            worst = max(worst, abs(pivot), abs(gear))
            rows.append('%s pivot %5.1f %% (lever %4.1f mm)  gear %5.1f %% (lever %4.1f mm / %g)' % (
                leg, 100 * abs(pivot) / STALL, 1000 * abs(pivot) / F, 100 * abs(gear) / STALL, 1000 * abs(t_calf) / F, n))
        print('  %-31s worst servo %3.0f %% of stall | ' % (name, 100 * worst / STALL) + ' | '.join(rows))


if __name__ == '__main__':
    main()
