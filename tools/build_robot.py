#!/usr/bin/env python3
"""Build the Fox quadruped RL package from the Fusion mechanism export in raw/.

raw/raw.json + raw/meshes/*.stl come from tools/fusion/FoxExportRaw run on 'Fox_Prototype_03_v6_RL_Mechanism'
(frame: x forward, y left, z up; origin between the hips; cm / kg in raw.json, metres in the STLs). Each leg is the real
mechanism: hip servo -> abduction; pivot servo -> femur; gear servo -> pinion -> 12:12 gear with crank -> Quad Link ->
tibia. hip-crank-Quad Link-knee is a parallelogram (the tibia keeps the crank's angle); the foot hangs from a second
four-bar (Component77 front, Calf Link rear - a parallelogram on the rear legs). Outputs:
  mjcf/fox.xml (+ scene.xml)   exact mechanism: loops closed with <connect>, the gear pair with a joint equality,
                               position servos on the 12 servo joints (hip, pivot = femur, gear = pinion)
  mjcf/fox_reduced.xml         the same robot as a tree (base + per leg hip, thigh, calf, foot): the gear servo acts on
                               a fixed tendon (-thigh - calf), the foot follows the four-bar through a joint equality
  urdf/fox.urdf                that tree for Isaac (URDF cannot hold loops); foot joint as <mimic> of the calf joint
  mechanism.json               servo <-> joint map + four-bar fits, used by isaaclab/fox_cfg.py and the checks
Servo map (all pitch angles about +y, zero = CAD stance):  thigh = pivot,  calf = -gear_servo - pivot,  foot = f(calf).
Usage:  ~/.venvs/fox_rl/bin/python tools/build_robot.py
"""
import json, os, sys
import numpy as np
import trimesh
import fast_simplification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'raw')
CM = 0.01
# DS-843MG at 6 V (servodatabase): 0.47 N m stall, 0.10 s/60 deg = 10.5 rad/s. The gear pair is 1:1, so the gear
# servo's torque reaches the crank unchanged (reversed).
EFFORT, VELOCITY = 0.47, 10.5
ARMATURE, DAMPING, FRICTIONLOSS = 0.0005, 0.01, 0.002   # servo joints (estimates; reflected rotor inertia dominates)
PIN_ARMATURE, PIN_DAMPING = 1e-6, 1e-4                  # free pins of the linkages
KP, KV = 5.0, 0.08                                      # position-servo gains (N m / rad, N m s / rad): stall at ~5 deg error
VIS_FACES, VIS_DEFAULT = {'base': 60000}, 20000         # decimation targets for visual meshes
LEGS = ('FL', 'FR', 'RL', 'RR')
RGBA = {'base': (0.33, 0.42, 0.52, 1), 'hip': (0.12, 0.12, 0.13, 1), 'pinion': (0.3, 0.3, 0.32, 1), 'gear': (0.3, 0.3, 0.32, 1),
        'femur': (0.45, 0.68, 0.9, 1), 'quad': (0.85, 0.6, 0.3, 1), 'tibia': (0.62, 0.8, 0.95, 1), 'link': (0.85, 0.6, 0.3, 1),
        'foot': (0.2, 0.55, 0.35, 1), 'thigh': (0.45, 0.68, 0.9, 1), 'calf': (0.62, 0.8, 0.95, 1)}
REDUCED = {'hip': ('hip', 'pinion', 'gear'), 'thigh': ('femur', 'quad'), 'calf': ('tibia', 'link'), 'foot': ('foot',)}
REDUCED_JOINT = {'hip': 'hip_joint', 'thigh': 'femur_joint', 'calf': 'knee_joint', 'foot': 'ankle_joint'}
SERVOS = (('hip', 'hip_joint'), ('pivot', 'femur_joint'), ('gear', 'pinion_joint'))   # actuator name -> servo joint


def rgba(n):
    return RGBA['base'] if n == 'base' else RGBA[n.split('_')[1]]


def axis_of(j):
    """Rotation axis measured in Fusion, signed so abduction turns about ~+x (forward) and every pitch pin about +y."""
    a = np.array(j['axis_measured'] or j['primary'], dtype=float)
    a /= np.linalg.norm(a)
    return np.round(a if a[0 if j['name'].endswith('_hip_joint') else 1] > 0 else -a, 6)


def inertia_about_com(I_origin, com, m):
    """Fusion getXYZMomentsOfInertia (about world origin, [xx,yy,zz,xy,yz,xz], kg cm^2) -> about COM (fusion2urdf rule)."""
    x, y, z = com
    shift = [y * y + z * z, x * x + z * z, x * x + y * y, -x * y, -y * z, -x * z]
    xx, yy, zz, xy, yz, xz = [i - m * s for i, s in zip(I_origin, shift)]
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * 1e-4  # kg m^2


def rot_y_angle(u, v):
    """Angle about +y that turns x-z vector u into v (x' = x cos + z sin, z' = -x sin + z cos)."""
    return np.arctan2(u[1] * v[0] - u[0] * v[1], u[0] * v[0] + u[1] * v[1])


MIN_TRANSMISSION = np.radians(10.0)   # linkage dead-point margin: the knee range stops where a loop's pins line up


def four_bar(K, C, A, Cp, q):
    """Foot angle relative to the tibia when the knee (tibia relative to femur, about +y) is at q, and the sine of the
    transmission angle at the foot pin (0 = dead point, where the loop can flip). Points are x-z coordinates at the
    zero pose: K knee, C femur-link pin, A ankle, Cp link-foot pin."""
    l1, l2 = np.linalg.norm(Cp - C), np.linalg.norm(Cp - A)
    out, mu, prev = [], [], Cp.copy()
    for qq in q:                                   # march outwards from 0 so the branch stays continuous
        c, s = np.cos(qq), np.sin(qq)
        a = K + np.array([[c, s], [-s, c]]) @ (A - K)
        d = np.linalg.norm(a - C)
        if d > l1 + l2 or d < abs(l1 - l2):
            out.append(np.nan)
            mu.append(0.0)
            continue
        t = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(max(l1 ** 2 - t ** 2, 0.0))
        e = (a - C) / d
        sols = [C + t * e + h * np.array([-e[1], e[0]]), C + t * e - h * np.array([-e[1], e[0]])]
        prev = min(sols, key=lambda p: np.linalg.norm(p - prev))
        out.append(rot_y_angle(Cp - A, prev - a) - qq)
        u, w = prev - C, prev - a
        mu.append(abs(u[0] * w[1] - u[1] * w[0]) / (np.linalg.norm(u) * np.linalg.norm(w)))
    return np.array(out), np.array(mu)


def foot_fit(K, C, A, Cp, span=1.2):
    """Knee range around 0 where the foot four-bar stays clear of its dead point, and a quartic fit (MuJoCo's
    joint-equality polynomial) of the foot angle over that range."""
    side = []
    for sgn in (-1, 1):
        q = np.linspace(0, sgn * span, 241)
        f, mu = four_bar(K, C, A, Cp, q)
        n = next((i for i in range(len(q)) if not (np.isfinite(f[i]) and mu[i] >= np.sin(MIN_TRANSMISSION))), len(q))
        side.append((q[:n], f[:n]))
    q = np.concatenate([side[0][0][::-1], side[1][0][1:]])
    f = np.concatenate([side[0][1][::-1], side[1][1][1:]])
    coef = np.polynomial.polynomial.polyfit(q, f, 4)
    err = np.abs(np.polynomial.polynomial.polyval(q, coef) - f).max()
    return coef, float(err), [float(q.min()), float(q.max())]


def parallelogram_range(H, Q1, K):
    """Knee range keeping the crank (H->Q1) and femur (H->K) at least MIN_TRANSMISSION from lining up."""
    a = abs(rot_y_angle(Q1 - H, K - H))
    return [-(a - MIN_TRANSMISSION), np.pi - a - MIN_TRANSMISSION]


def main():
    raw = json.load(open(os.path.join(RAW, 'raw.json')))
    links, joints = raw['links'], raw['joints']
    J = {j['name']: j for j in joints}
    parent, tree, loops = {'base': None}, [], []
    for j in joints:                                  # first joint that reaches a part is its tree joint; later ones close loops
        (loops if j['child'] in parent else tree).append(j)
        parent.setdefault(j['child'], j['parent'])
    assert set(parent) == set(links) and len(loops) == 2 * len(LEGS), (len(loops), set(links) - set(parent))
    tj = {j['child']: j for j in tree}
    frame = {n: (np.array(tj[n]['origin_cm']) * CM if n in tj else np.zeros(3)) for n in links}
    world = {n: np.array(j['origin_cm']) * CM for n, j in J.items()}
    for d in ('meshes/visual', 'meshes/collision', 'urdf', 'mjcf'):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    raw_mesh = {n: trimesh.load(os.path.join(RAW, 'meshes', L['stl']), force='mesh') for n, L in links.items()}
    report = {'links': {}, 'joints': {}, 'reduced': {}}

    def write_meshes(n, mesh, origin):              # visual (decimated) + collision (convex hull) in the link frame
        mesh = mesh.copy()
        mesh.apply_translation(-origin)
        tgt = VIS_FACES.get(n, VIS_DEFAULT)
        vis = mesh
        if len(mesh.faces) > tgt:
            v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, 1.0 - tgt / len(mesh.faces))
            vis = trimesh.Trimesh(v, f, process=True)
        vis.export(os.path.join(ROOT, 'meshes', 'visual', n + '.stl'))
        mesh.convex_hull.export(os.path.join(ROOT, 'meshes', 'collision', n + '.stl'))
        return len(mesh.faces), len(vis.faces)

    def inertial(names, origin):                    # merged mass properties of parts, COM in the link frame
        m = sum(links[n]['mass_kg'] for n in names)
        com = sum(links[n]['mass_kg'] * np.array(links[n]['com_cm']) for n in names) / m
        I = inertia_about_com(np.sum([links[n]['inertia_origin_kgcm2'] for n in names], axis=0), com, m)
        ev = np.linalg.eigvalsh(I)
        valid = ev.min() > 0 and all(ev[i] + ev[j] >= ev[k] - 1e-12 for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)))
        return m, com * CM - origin, I, bool(valid)

    # ---------------- four-bar foot + servo map ----------------
    mech = {'servo_map': {'thigh': 'pivot', 'calf': '-gear - pivot', 'foot': 'f(calf)', 'gear_ratio': -1.0,
                          'note': 'pitch angles about +y from the CAD stance; gear = 12:12 external mesh'}, 'legs': {}}
    for leg in LEGS:
        P = {k: world[leg + n][[0, 2]] for k, n in (('K', '_knee_joint'), ('C', '_link_pin_joint'), ('A', '_ankle_joint'), ('Cp', '_foot_pin_joint'))}
        coef, err, rng = foot_fit(P['K'], P['C'], P['A'], P['Cp'])
        H, Q1, Q2, K = (world[leg + n][[0, 2]] for n in ('_femur_joint', '_crank_pin_joint', '_quad_pin_joint', '_knee_joint'))
        up = sorted(parallelogram_range(H, Q1, K))
        knee = [round(max(rng[0], up[0], -1.2), 4), round(min(rng[1], up[1], 1.2), 4)]   # also capped at the +-1.2 placeholder
        mech['legs'][leg] = {'foot_poly': coef.round(6).tolist(), 'foot_fit_err_rad': err, 'knee_range_rad': knee,
                             'foot_loop_range_rad': np.round(rng, 4).tolist(), 'upper_loop_range_rad': np.round(up, 4).tolist(),
                             'parallelogram_err_m': float(np.linalg.norm((Q1 - H) - (Q2 - K)) + np.linalg.norm((K - H) - (Q2 - Q1)))}
    json.dump(mech, open(os.path.join(ROOT, 'mechanism.json'), 'w'), indent=1)

    # ---------------- exact mechanism MJCF ----------------
    inert = {}
    for n in links:
        full, vis = write_meshes(n, raw_mesh[n], frame[n])
        inert[n] = inertial([n], frame[n])
        report['links'][n] = {'mass_g': round(inert[n][0] * 1000, 2), 'com_link_m': np.round(inert[n][1], 5).tolist(),
                              'physically_valid': inert[n][3], 'mesh_faces_full': full, 'visual_faces': vis,
                              'mesh_z_min_world_m': float(raw_mesh[n].bounds[0][2])}
    feet_z = min(r['mesh_z_min_world_m'] for r in report['links'].values())
    kids = {}
    for j in tree:
        kids.setdefault(j['parent'], []).append(j)

    def body(n, indent, joint=None):
        m, c, I, _ = inert[n]
        sp = '  ' * indent
        pos = (0.0, 0.0, -feet_z + 0.002) if joint is None else frame[n] - frame[joint['parent']]
        out = ['%s<body name="%s" pos="%.6f %.6f %.6f">' % (sp, n, *pos)]
        if joint is None:
            out.append('%s  <freejoint name="root"/>' % sp)
        else:
            servo, leg = joint['lower'] is not None, joint['name'][:2]
            rng = ' range="%.4f %.4f"' % ((joint['lower'], joint['upper']) if servo else tuple(mech['legs'][leg]['knee_range_rad']))
            out.append('%s  <joint name="%s" axis="%g %g %g"%s/>' % (sp, joint['name'], *axis_of(joint),
                       rng if servo else ' class="pin"' + (rng if joint['name'].endswith('_knee_joint') else '')))
        out.append('%s  <inertial pos="%.6f %.6f %.6f" mass="%.6f" fullinertia="%.4e %.4e %.4e %.4e %.4e %.4e"/>' %
                   (sp, *c, m, I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
        out.append('%s  <geom class="visual" mesh="%s_vis" rgba="%g %g %g %g"/>' % ((sp, n) + rgba(n)))
        out.append('%s  <geom class="collision" mesh="%s_col"/>' % (sp, n))
        if n == 'base':
            out.append('%s  <site name="imu" pos="0 0 0"/>' % sp)
        for cj in kids.get(n, []):
            out += body(cj['child'], indent + 1, cj)
        out.append('%s</body>' % sp)
        return out

    def header(name, names, stem=lambda n: n):
        X = ['<mujoco model="%s">' % name,
             '  <compiler angle="radian" meshdir="../meshes" autolimits="true"/>',
             '  <option timestep="0.002" integrator="implicitfast"/>',
             '  <default>',
             '    <joint armature="%g" damping="%g" frictionloss="%g"/>' % (ARMATURE, DAMPING, FRICTIONLOSS),
             '    <default class="pin"><joint armature="%g" damping="%g" frictionloss="0"/></default>' % (PIN_ARMATURE, PIN_DAMPING),
             '    <position kp="%g" kv="%g" forcerange="%g %g" inheritrange="1"/>' % (KP, KV, -EFFORT, EFFORT),
             # near-hard loops: with the default solimp (0.9) the 2-3 g linkage parts let the pins gap ~2 mm under load
             '    <equality solref="0.004 1" solimp="0.999 0.9999 0.0005 0.5 2"/>',
             '    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/></default>',
             '    <default class="collision"><geom type="mesh" contype="1" conaffinity="0" group="3" density="0" friction="1.0 0.02 0.01" condim="3"/></default>',
             '  </default>', '  <asset>']
        for n in names:
            X += ['    <mesh name="%s_vis" file="visual/%s.stl"/>' % (n, stem(n)), '    <mesh name="%s_col" file="collision/%s.stl"/>' % (n, stem(n))]
        return X + ['  </asset>', '  <worldbody>']
    sensors = ['  <sensor>', '    <framequat name="imu_quat" objtype="site" objname="imu"/>',
               '    <gyro name="imu_gyro" site="imu"/>', '    <accelerometer name="imu_acc" site="imu"/>', '  </sensor>', '</mujoco>']
    X = header('fox', links) + body('base', 2) + ['  </worldbody>', '  <equality>']
    for j in loops:                                   # anchor in the parent part's frame (both coincide at qpos0)
        a = world[j['name']] - frame[j['parent']]
        X.append('    <connect name="%s" body1="%s" body2="%s" anchor="%.6f %.6f %.6f"/>' % (j['name'].replace('_joint', ''), j['parent'], j['child'], *a))
    for leg in LEGS:
        ml = J[leg + '_pinion_joint'].get('motion_links') or J[leg + '_gear_joint'].get('motion_links')
        assert ml and ml[0][2], 'gear motion link missing or not reversed: %s' % ml
        X.append('    <joint name="%s_gear_mesh" joint1="%s_gear_joint" joint2="%s_pinion_joint" polycoef="0 -1 0 0 0"/>' % (leg, leg, leg))
    X += ['  </equality>', '  <actuator>']
    X += ['    <position name="%s_%s" joint="%s_%s"/>' % (leg, a, leg, sj) for leg in LEGS for a, sj in SERVOS]
    X += ['  </actuator>'] + sensors
    open(os.path.join(ROOT, 'mjcf', 'fox.xml'), 'w').write('\n'.join(X) + '\n')
    open(os.path.join(ROOT, 'mjcf', 'scene.xml'), 'w').write(SCENE % 'fox.xml')
    for j in joints:
        report['joints'][j['name']] = {'parent': j['parent'], 'child': j['child'], 'origin_world_m': np.round(world[j['name']], 5).tolist(),
                                       'axis': axis_of(j).tolist(), 'servo': j['lower'] is not None, 'closes_loop': j in loops,
                                       'limits_rad': [j['lower'], j['upper']] if j['lower'] is not None else None}

    # ---------------- reduced tree: URDF + MJCF ----------------
    red = ['base'] + ['%s_%s' % (leg, k) for leg in LEGS for k in REDUCED]
    parts = {'base': ['base'], **{'%s_%s' % (leg, k): ['%s_%s' % (leg, p) for p in v] for leg in LEGS for k, v in REDUCED.items()}}
    rjoint = {'%s_%s' % (leg, k): J[leg + '_' + REDUCED_JOINT[k]] for leg in LEGS for k in REDUCED}
    rframe = {n: (world[rjoint[n]['name']] if n in rjoint else np.zeros(3)) for n in red}
    rparent = {n: ('base' if n.endswith('_hip') else n[:3] + {'thigh': 'hip', 'calf': 'thigh', 'foot': 'calf'}[n[3:]]) for n in red if n != 'base'}
    rin, stem = {}, (lambda n: n if n == 'base' else n + '_reduced')   # own files: XX_hip/XX_foot exist in the exact model too
    for n in red:
        if n != 'base':
            write_meshes(stem(n), trimesh.util.concatenate([raw_mesh[p] for p in parts[n]]), rframe[n])
        rin[n] = inertial(parts[n], rframe[n])
        report['reduced'][n] = {'parts': parts[n], 'mass_g': round(rin[n][0] * 1000, 2), 'physically_valid': rin[n][3]}
    rlim = {'hip': (J['FL_hip_joint']['lower'], J['FL_hip_joint']['upper']), 'thigh': (J['FL_femur_joint']['lower'], J['FL_femur_joint']['upper']),
            'foot': (-np.pi, np.pi)}   # calf (knee): per leg, from the linkage dead points (mechanism.json knee_range_rad)

    def origin(xyz):
        return '<origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % tuple(xyz)
    U = ['<?xml version="1.0"?>', '<robot name="fox">']
    for n in red:
        m, c, I, _ = rin[n]
        U += ['  <link name="%s">' % n, '    <inertial>', '      ' + origin(c), '      <mass value="%.6f"/>' % m,
              '      <inertia ixx="%.4e" ixy="%.4e" ixz="%.4e" iyy="%.4e" iyz="%.4e" izz="%.4e"/>' % (I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]),
              '    </inertial>',
              '    <visual>', '      ' + origin((0, 0, 0)), '      <geometry><mesh filename="../meshes/visual/%s.stl"/></geometry>' % stem(n),
              '      <material name="%s_mat"><color rgba="%g %g %g %g"/></material>' % ((n,) + rgba(n)), '    </visual>',
              '    <collision>', '      ' + origin((0, 0, 0)), '      <geometry><mesh filename="../meshes/collision/%s.stl"/></geometry>' % stem(n), '    </collision>',
              '  </link>']
    for n in red[1:]:
        k, leg = n[3:], n[:2]
        lo, hi = mech['legs'][leg]['knee_range_rad'] if k == 'calf' else rlim[k]
        U += ['  <joint name="%s_joint" type="revolute">' % n, '    ' + origin(rframe[n] - rframe[rparent[n]]),
              '    <parent link="%s"/>' % rparent[n], '    <child link="%s"/>' % n, '    <axis xyz="%g %g %g"/>' % tuple(axis_of(rjoint[n])),
              # thigh torque = pivot - gear servo torques (up to 2 x stall); PhysX clips joint efforts at this value
              '    <limit lower="%.4f" upper="%.4f" effort="%.3f" velocity="%.2f"/>' % (lo, hi, {'thigh': 2 * EFFORT, 'foot': 10.0}.get(k, EFFORT),
                                                                                       VELOCITY if k != 'foot' else 50.0),
              # damping/friction 0 on purpose: Isaac's URDF importer writes URDF damping into the per-degree USD field;
              # joint damping/armature belong in the actuator config (isaaclab/fox_cfg.py) and the MJCF defaults.
              '    <dynamics damping="0" friction="0"/>']
        if k == 'foot':   # linear part of the four-bar (exact for the rear parallelogram); fox_cfg.py can apply the full fit
            U.append('    <mimic joint="%s_calf_joint" multiplier="%.6f" offset="0"/>' % (leg, mech['legs'][leg]['foot_poly'][1]))
        U.append('  </joint>')
    U.append('</robot>')
    open(os.path.join(ROOT, 'urdf', 'fox.urdf'), 'w').write('\n'.join(U) + '\n')
    rkids = {}
    for n in red[1:]:
        rkids.setdefault(rparent[n], []).append(n)

    def rbody(n, indent):
        m, c, I, _ = rin[n]
        sp = '  ' * indent
        pos = (0.0, 0.0, -feet_z + 0.002) if n == 'base' else rframe[n] - rframe[rparent[n]]
        out = ['%s<body name="%s" pos="%.6f %.6f %.6f">' % (sp, n, *pos)]
        if n == 'base':
            out.append('%s  <freejoint name="root"/>' % sp)
        else:
            k = n[3:]
            lim = tuple(mech['legs'][n[:2]]['knee_range_rad']) if k == 'calf' else rlim.get(k)
            out.append('%s  <joint name="%s_joint" axis="%g %g %g"%s/>' % (sp, n, *axis_of(rjoint[n]),
                       ' class="pin"' if k == 'foot' else ' range="%.4f %.4f"' % lim))
        out.append('%s  <inertial pos="%.6f %.6f %.6f" mass="%.6f" fullinertia="%.4e %.4e %.4e %.4e %.4e %.4e"/>' %
                   (sp, *c, m, I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
        out.append('%s  <geom class="visual" mesh="%s_vis" rgba="%g %g %g %g"/>' % ((sp, n) + rgba(n)))
        out.append('%s  <geom class="collision" mesh="%s_col"/>' % (sp, n))
        if n == 'base':
            out.append('%s  <site name="imu" pos="0 0 0"/>' % sp)
        for cn in rkids.get(n, []):
            out += rbody(cn, indent + 1)
        out.append('%s</body>' % sp)
        return out
    R = header('fox_reduced', red, stem) + rbody('base', 2) + ['  </worldbody>', '  <tendon>']
    R += ['    <fixed name="%s_gear_servo"><joint joint="%s_thigh_joint" coef="-1"/><joint joint="%s_calf_joint" coef="-1"/></fixed>' % (l, l, l) for l in LEGS]
    R += ['  </tendon>', '  <equality>']
    R += ['    <joint name="%s_four_bar" joint1="%s_foot_joint" joint2="%s_calf_joint" polycoef="%s"/>' % (l, l, l, ' '.join('%.6g' % c for c in mech['legs'][l]['foot_poly'])) for l in LEGS]
    R += ['  </equality>', '  <actuator>']
    for l in LEGS:
        R += ['    <position name="%s_hip" joint="%s_hip_joint"/>' % (l, l), '    <position name="%s_pivot" joint="%s_thigh_joint"/>' % (l, l),
              '    <position name="%s_gear" tendon="%s_gear_servo" inheritrange="0" ctrlrange="%.4f %.4f"/>' % (l, l, J[l + '_pinion_joint']['lower'], J[l + '_pinion_joint']['upper'])]
    R += ['  </actuator>'] + sensors
    open(os.path.join(ROOT, 'mjcf', 'fox_reduced.xml'), 'w').write('\n'.join(R) + '\n')
    open(os.path.join(ROOT, 'mjcf', 'scene_reduced.xml'), 'w').write(SCENE % 'fox_reduced.xml')
    report['standing_height_base_origin_m'] = -feet_z + 0.002
    report['total_mass_g'] = round(sum(l['mass_kg'] for l in links.values()) * 1000, 1)
    report['mechanism'] = mech
    json.dump(report, open(os.path.join(ROOT, 'build_report.json'), 'w'), indent=1)
    bad = [n for n, r in report['links'].items() if not r['physically_valid']] + [n for n, r in report['reduced'].items() if not r['physically_valid']]
    print('parts %d  joints %d (tree %d, loops %d)  reduced links %d  mass %.1f g  base height %.4f m  invalid inertias %s'
          % (len(links), len(joints), len(tree), len(loops), len(red), report['total_mass_g'], report['standing_height_base_origin_m'], bad))
    for leg, v in mech['legs'].items():
        print('  %s foot = f(calf): poly %s  fit err %.1e rad | knee range %s (foot loop %s, upper loop %s) | parallelogram err %.1e m'
              % (leg, np.round(v['foot_poly'], 4).tolist(), v['foot_fit_err_rad'], v['knee_range_rad'], v['foot_loop_range_rad'],
                 v['upper_loop_range_rad'], v['parallelogram_err_m']))
    return 0 if not bad else 1


SCENE = '''<mujoco model="fox scene">
  <include file="%s"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/><global azimuth="135" elevation="-20" offwidth="1280" offheight="960"/></visual>
  <asset><texture type="2d" name="grid" builtin="checker" rgb1="0.25 0.3 0.35" rgb2="0.2 0.22 0.25" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/></asset>
  <worldbody><light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid" contype="0" conaffinity="1" friction="1.0 0.02 0.01"/></worldbody>
</mujoco>
'''

if __name__ == '__main__':
    sys.exit(main())
