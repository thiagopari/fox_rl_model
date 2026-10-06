#!/usr/bin/env python3
"""Build the Fox quadruped RL package (URDF + MJCF + meshes) from the Fusion export in raw/.

raw/raw.json + raw/meshes/*.stl come from the Fusion add-in run on 'Fox_Prototype_03_v6_RL_URDF_v2'
(frame: x forward, y left, z up; origin between the hips; units cm / kg in raw.json, metres in the STLs).
Usage:  ~/.venvs/fox_rl/bin/python tools/build_robot.py
"""
import json, os, sys
import numpy as np
import trimesh
import fast_simplification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'raw')
CM = 0.01
# DS-843MG at 6 V (servodatabase): 0.47 N m stall, 0.10 s/60 deg = 10.5 rad/s. Gear/linkage ratios folded in later.
EFFORT, VELOCITY = 0.47, 10.5
ARMATURE, DAMPING, FRICTIONLOSS = 0.0005, 0.01, 0.002   # estimates for a geared hobby servo (reflected inertia dominates)
KP, KV = 2.0, 0.05                                      # MuJoCo position-servo gains (N m / rad, N m s / rad)
VIS_FACES = {'base': 60000}                             # decimation targets for visual meshes
VIS_DEFAULT = 20000
RGBA = {'base': (0.33, 0.42, 0.52, 1), 'hip': (0.12, 0.12, 0.13, 1), 'thigh': (0.45, 0.68, 0.9, 1), 'calf': (0.62, 0.8, 0.95, 1)}


def rgba(link):
    return RGBA['base'] if link == 'base' else RGBA[link.split('_')[1]]


def axis_of(j):
    """Rotation axis measured in Fusion, signed so abduction turns about ~+x (forward) and pitch about +y (left).
    The abduction axes are the real hip-servo spline axes: tilted 10 deg (front) / 20 deg (rear) out of the x-y plane."""
    a = np.array(j['axis_measured'], dtype=float)
    a /= np.linalg.norm(a)
    a = a if a[0 if j['name'].endswith('_hip_joint') else 1] > 0 else -a
    return tuple(float(round(c, 6)) for c in a)


def inertia_about_com(I_origin, com, m):
    """Fusion getXYZMomentsOfInertia (about world origin, [xx,yy,zz,xy,yz,xz], kg cm^2) -> about COM (fusion2urdf rule)."""
    x, y, z = com
    shift = [y * y + z * z, x * x + z * z, x * x + y * y, -x * y, -y * z, -x * z]
    xx, yy, zz, xy, yz, xz = [i - m * s for i, s in zip(I_origin, shift)]
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * 1e-4  # kg m^2


def main():
    raw = json.load(open(os.path.join(RAW, 'raw.json')))
    links, joints = raw['links'], raw['joints']
    parent_joint = {j['child']: j for j in joints}
    frame = {n: np.zeros(3) for n in links}
    for n in links:
        if n in parent_joint:
            frame[n] = np.array(parent_joint[n]['origin_cm']) * CM
    os.makedirs(os.path.join(ROOT, 'meshes', 'visual'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'meshes', 'collision'), exist_ok=True)
    report = {'links': {}, 'joints': {}}
    inertial = {}
    for n, L in links.items():
        mesh = trimesh.load(os.path.join(RAW, 'meshes', L['stl']), force='mesh')
        mesh.apply_translation(-frame[n])                       # into the link frame
        tgt = VIS_FACES.get(n, VIS_DEFAULT)
        if len(mesh.faces) > tgt:
            v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, 1.0 - tgt / len(mesh.faces))
            vis = trimesh.Trimesh(v, f, process=True)
        else:
            vis = mesh
        vis.export(os.path.join(ROOT, 'meshes', 'visual', n + '.stl'))
        hull = mesh.convex_hull
        hull.export(os.path.join(ROOT, 'meshes', 'collision', n + '.stl'))
        m = L['mass_kg']
        com_world_cm = np.array(L['com_cm'])
        I = inertia_about_com(L['inertia_origin_kgcm2'], com_world_cm, m)
        ev = np.linalg.eigvalsh(I)
        tri = ev.min() > 0 and all(ev[i] + ev[j] >= ev[k] - 1e-12 for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)))
        inertial[n] = (m, com_world_cm * CM - frame[n], I)
        report['links'][n] = {'mass_g': round(m * 1000, 2), 'com_link_m': np.round(com_world_cm * CM - frame[n], 5).tolist(),
                              'inertia_eigs_kgm2': ev.tolist(), 'physically_valid': bool(tri),
                              'mesh_faces_full': int(len(mesh.faces)), 'visual_faces': int(len(vis.faces)),
                              'hull_faces': int(len(hull.faces)), 'mesh_z_min_world_m': float(mesh.bounds[0][2] + frame[n][2])}
    # ---------------- URDF ----------------
    def origin(xyz):
        return '<origin xyz="%.6f %.6f %.6f" rpy="0 0 0"/>' % tuple(xyz)
    U = ['<?xml version="1.0"?>', '<robot name="fox">']
    for n in links:
        m, c, I = inertial[n]
        U += ['  <link name="%s">' % n,
              '    <inertial>', '      ' + origin(c), '      <mass value="%.6f"/>' % m,
              '      <inertia ixx="%.4e" ixy="%.4e" ixz="%.4e" iyy="%.4e" iyz="%.4e" izz="%.4e"/>' % (I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]),
              '    </inertial>',
              '    <visual>', '      ' + origin((0, 0, 0)), '      <geometry><mesh filename="../meshes/visual/%s.stl"/></geometry>' % n,
              '      <material name="%s_mat"><color rgba="%g %g %g %g"/></material>' % ((n,) + rgba(n)), '    </visual>',
              '    <collision>', '      ' + origin((0, 0, 0)), '      <geometry><mesh filename="../meshes/collision/%s.stl"/></geometry>' % n, '    </collision>',
              '  </link>']
    for j in joints:
        o = frame[j['child']] - frame[j['parent']]
        U += ['  <joint name="%s" type="revolute">' % j['name'], '    ' + origin(o),
              '    <parent link="%s"/>' % j['parent'], '    <child link="%s"/>' % j['child'],
              '    <axis xyz="%g %g %g"/>' % axis_of(j),
              '    <limit lower="%.4f" upper="%.4f" effort="%.3f" velocity="%.2f"/>' % (j['lower'], j['upper'], EFFORT, VELOCITY),
              # damping/friction 0 on purpose: Isaac's URDF importer writes URDF damping into the per-degree USD field;
              # joint damping/armature belong in the actuator config (isaaclab/fox_cfg.py) and the MJCF defaults.
              '    <dynamics damping="0" friction="0"/>',
              '  </joint>']
        report['joints'][j['name']] = {'parent': j['parent'], 'child': j['child'], 'origin_in_parent_m': np.round(o, 5).tolist(),
                                       'axis': axis_of(j), 'limits_rad': [j['lower'], j['upper']]}
    U.append('</robot>')
    os.makedirs(os.path.join(ROOT, 'urdf'), exist_ok=True)
    open(os.path.join(ROOT, 'urdf', 'fox.urdf'), 'w').write('\n'.join(U) + '\n')
    # ---------------- MJCF ----------------
    children = {}
    for j in joints:
        children.setdefault(j['parent'], []).append(j)
    feet_z = min(r['mesh_z_min_world_m'] for r in report['links'].values())
    def body(n, pos, indent, joint=None):
        m, c, I = inertial[n]
        sp = '  ' * indent
        out = ['%s<body name="%s" pos="%.6f %.6f %.6f">' % (sp, n, *pos)]
        if joint is None:
            out.append('%s  <freejoint name="root"/>' % sp)
        else:
            out.append('%s  <joint name="%s" axis="%g %g %g" range="%.4f %.4f"/>' % (sp, joint['name'], *axis_of(joint), joint['lower'], joint['upper']))
        out.append('%s  <inertial pos="%.6f %.6f %.6f" mass="%.6f" fullinertia="%.4e %.4e %.4e %.4e %.4e %.4e"/>' %
                   (sp, *c, m, I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
        out.append('%s  <geom class="visual" mesh="%s_vis" rgba="%g %g %g %g"/>' % ((sp, n) + rgba(n)))
        out.append('%s  <geom class="collision" mesh="%s_col"/>' % (sp, n))
        if n == 'base':
            out.append('%s  <site name="imu" pos="0 0 0"/>' % sp)
        for cj in children.get(n, []):
            out += body(cj['child'], frame[cj['child']] - frame[n], indent + 1, cj)
        out.append('%s</body>' % sp)
        return out
    X = ['<mujoco model="fox">',
         '  <compiler angle="radian" meshdir="../meshes" autolimits="true"/>',
         '  <option timestep="0.002" integrator="implicitfast"/>',
         '  <default>',
         '    <joint armature="%g" damping="%g" frictionloss="%g"/>' % (ARMATURE, DAMPING, FRICTIONLOSS),
         '    <position kp="%g" kv="%g" forcerange="%g %g" inheritrange="1"/>' % (KP, KV, -EFFORT, EFFORT),
         '    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/></default>',
         '    <default class="collision"><geom type="mesh" contype="1" conaffinity="0" group="3" density="0" friction="1.0 0.02 0.01" condim="3"/></default>',
         '  </default>',
         '  <asset>']
    for n in links:
        X.append('    <mesh name="%s_vis" file="visual/%s.stl"/>' % (n, n))
        X.append('    <mesh name="%s_col" file="collision/%s.stl"/>' % (n, n))
    X += ['  </asset>', '  <worldbody>']
    X += body('base', (0.0, 0.0, -feet_z + 0.002), 2)
    X += ['  </worldbody>', '  <actuator>']
    for j in joints:
        X.append('    <position name="%s" joint="%s"/>' % (j['name'].replace('_joint', ''), j['name']))
    X += ['  </actuator>', '  <sensor>', '    <framequat name="imu_quat" objtype="site" objname="imu"/>',
          '    <gyro name="imu_gyro" site="imu"/>', '    <accelerometer name="imu_acc" site="imu"/>', '  </sensor>', '</mujoco>']
    os.makedirs(os.path.join(ROOT, 'mjcf'), exist_ok=True)
    open(os.path.join(ROOT, 'mjcf', 'fox.xml'), 'w').write('\n'.join(X) + '\n')
    open(os.path.join(ROOT, 'mjcf', 'scene.xml'), 'w').write('''<mujoco model="fox scene">
  <include file="fox.xml"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/><global azimuth="135" elevation="-20" offwidth="1280" offheight="960"/></visual>
  <asset><texture type="2d" name="grid" builtin="checker" rgb1="0.25 0.3 0.35" rgb2="0.2 0.22 0.25" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/></asset>
  <worldbody><light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid" contype="0" conaffinity="1" friction="1.0 0.02 0.01"/></worldbody>
</mujoco>
''')
    report['standing_height_base_origin_m'] = -feet_z + 0.002
    report['total_mass_g'] = round(sum(l['mass_kg'] for l in links.values()) * 1000, 1)
    json.dump(report, open(os.path.join(ROOT, 'build_report.json'), 'w'), indent=1)
    bad = [n for n, r in report['links'].items() if not r['physically_valid']]
    print('links', len(links), 'joints', len(joints), 'total mass g', report['total_mass_g'],
          'standing base height m', round(report['standing_height_base_origin_m'], 4), 'invalid inertias', bad)
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
