#!/usr/bin/env python3
"""raw_v7_nospacer/: the v7 robot as built WITHOUT the rear hip spacer, as a FoxExportRaw-style export for build_robot.py.

That robot is the v6 CAD (rear hips not raised) with the v7 drive: 12T pinion -> 24T crank gear, 2:1 reversed. So: the
v6 export (raw/) with the 8 gear and pinion links of the v7 export (raw_v7/: meshes, mass, inertia) moved back to the v6
positions (each by its own joint's v6 - v7 offset: the v7 frame sits 9.4 mm lower, and its rear is 20 mm up the rear
screw axis), the v7 motion links (2:1) and the v7 pinion range. Base, legs and every joint position stay v6.
Usage:  ~/.venvs/fox_rl/bin/python tools/raw_v7_without_spacer.py && ~/.venvs/fox_rl/bin/python tools/build_robot.py --raw raw_v7_nospacer --out v7
"""
import json, os, shutil
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6, V7, OUT = (os.path.join(ROOT, d) for d in ('raw', 'raw_v7', 'raw_v7_nospacer'))
LEGS = ('FL', 'FR', 'RL', 'RR')


def moved_inertia(link, t):
    """Mass properties of a link translated by t (cm): Fusion's inertia is about the world origin."""
    m, c = link['mass_kg'], np.array(link['com_cm'])
    S = lambda c: np.array([c[1] ** 2 + c[2] ** 2, c[0] ** 2 + c[2] ** 2, c[0] ** 2 + c[1] ** 2,      # noqa: E731
                            -c[0] * c[1], -c[1] * c[2], -c[0] * c[2]])
    I_com = np.array(link['inertia_origin_kgcm2']) - m * S(c)
    out = dict(link)
    out['com_cm'] = (c + t).tolist()
    out['inertia_origin_kgcm2'] = (I_com + m * S(c + t)).tolist()
    return out


def main():
    a, b = json.load(open(os.path.join(V6, 'raw.json'))), json.load(open(os.path.join(V7, 'raw.json')))
    J6, J7 = {j['name']: j for j in a['joints']}, {j['name']: j for j in b['joints']}
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    shutil.copytree(V6, OUT)
    swapped = {}
    for leg in LEGS:
        for part in ('gear', 'pinion'):
            n, jn = '%s_%s' % (leg, part), '%s_%s_joint' % (leg, part)
            t = np.array(J6[jn]['origin_cm']) - np.array(J7[jn]['origin_cm'])
            a['links'][n] = moved_inertia(b['links'][n], t)
            mesh = trimesh.load(os.path.join(V7, 'meshes', b['links'][n]['stl']), force='mesh')
            mesh.apply_translation(t * 0.01)                                       # STLs are in metres
            mesh.export(os.path.join(OUT, 'meshes', b['links'][n]['stl']))
            swapped[n] = np.round(t, 5).tolist()
            # the part sits on its joint exactly as in v7
            r6 = np.array(a['links'][n]['com_cm']) - np.array(J6[jn]['origin_cm'])
            r7 = np.array(b['links'][n]['com_cm']) - np.array(J7[jn]['origin_cm'])
            assert np.abs(r6 - r7).max() < 1e-9, n
        for jn in ('%s_pinion_joint' % leg, '%s_gear_joint' % leg):        # 2:1 gear mesh, v7 pinion range
            J6[jn]['motion_links'] = J7[jn]['motion_links']
            J6[jn]['lower'], J6[jn]['upper'] = J7[jn]['lower'], J7[jn]['upper']
    a['derived'] = {'from': 'raw (v6 export) + raw_v7 gear/pinion links', 'tool': 'tools/raw_v7_without_spacer.py',
                    'gear_pinion_shift_cm': swapped, 'v7_doc_version': b.get('doc_version')}
    json.dump(a, open(os.path.join(OUT, 'raw.json'), 'w'), indent=1)
    print('wrote %s: v6 base/legs/joints + v7 gear and pinion links (shifted %s), 2:1 motion links, pinion range %s'
          % (OUT, sorted({tuple(v) for v in swapped.values()}), (J7['FL_pinion_joint']['lower'], J7['FL_pinion_joint']['upper'])))
    print('mass: v6 %.1f g, no-spacer v7 %.1f g, v7 with spacer %.1f g' % tuple(
        sum(l['mass_kg'] for l in r['links'].values()) * 1000 for r in (json.load(open(os.path.join(V6, 'raw.json'))), a, b)))


if __name__ == '__main__':
    main()
