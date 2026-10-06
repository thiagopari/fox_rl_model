#!/usr/bin/env python3
"""Fastener analysis -> link membership for FoxBuildLinkModel (writes analysis/link_map.json).

Input: the hole scan of the design copy made by tools/fusion/FoxScrewScan (analysis/screw_scan.json; source frame, cm).
A "line" is a set of coaxial small cylindrical faces (holes, pins, splines) of different bodies whose axial extents touch.
  pivot line : coaxial with a servo output spline, or shared by two leg-linkage parts (the designed pin joints);
  rigid pair : bodies sharing any other line (a screw), or a body sitting on a spline of another servo (horn / bracket);
  anchors    : servo cases, leg-linkage parts, the frame and the electronics, whose link is known from the CAD names;
  every rigid group takes the link of its anchors (two different links in one group = error, nothing is written);
  a loose part with no anchor joins the most distal link on its pivot line (pins), else the smallest body whose box it
  overlaps most (only root|Body27, a 0.07 cm^3 block on the rear-right hip servo, needs this).
Usage: ~/.venvs/fox_rl/bin/python tools/screw_graph.py [analysis/screw_scan.json]
"""
import json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MID_X, MID_Y = -2.476478, -8.6
LINE_TOL, ANG_TOL, GAP = 0.02, 0.9998, 0.06
SPLINE_TOL = 0.06   # hand-placed servo models sit up to ~0.5 mm off the pivot they drive (front hip servos)
ELEC = ('Raspberry Pi 4 Model B', '12Channel PWM v2', 'XL4015 StepDown DC-DC 5A (CC-CV) v1', 'arduino nano', 'Adafruit_BNO055_AP203')
DEPTH = {'base': 0, 'hip': 1, 'thigh': 2, 'calf': 3}


def leg_of(c):
    return ('F' if c[1] < MID_Y else 'R') + ('L' if c[0] > MID_X else 'R')


def kind_of(leaf):
    if leaf.startswith(('Femur', 'Quad Link')):
        return 'thigh'
    if leaf.startswith(('Tibia', 'Component7', 'Foot', 'Main Foot', 'Calf Link', 'Component77')):
        return 'calf'
    if leaf.startswith('Spur Gear') or leaf == 'm' or leaf.startswith('m('):
        return 'hip'
    return 'base'


def centre(b):
    return [(b['bbmin'][i] + b['bbmax'][i]) / 2 for i in range(3)]


def line_key(h):
    """hole [o, a, r, tmin, tmax] -> (unit dir with first nonzero comp > 0, closest point to origin, interval on dir)."""
    o, a = np.array(h[0:3]), np.array(h[3:6]) / np.linalg.norm(h[3:6])
    s = 1.0 if next(v for v in a if abs(v) > 1e-9) > 0 else -1.0
    u, t0 = a * s, float(np.dot(o, a * s))
    iv = sorted([t0 + s * h[7], t0 + s * h[8]])
    return u, o - t0 * u, iv


def coaxial(a, p, b, q, tol=LINE_TOL):
    return abs(np.dot(a, b)) > ANG_TOL and np.linalg.norm(np.cross(q - p, a)) < tol


def touch(u, v):
    return min(u[1], v[1]) - max(u[0], v[0]) > -GAP


def anchor_and_roles(bodies):
    servo = defaultdict(list)
    for b in bodies:
        if 'DS-843' in b['comp']:
            servo[b['src'].split('|')[0]].append(b)
    for occ, bs in servo.items():
        c = [(min(x['bbmin'][i] for x in bs) + max(x['bbmax'][i] for x in bs)) / 2 for i in range(3)]
        abd = (MID_Y < c[1] < -1.5) or (-15.5 < c[1] < MID_Y)     # the inboard servo of each quadrant drives abduction
        for b in bs:
            b['servo'], b['leg'], b['abd'] = occ, leg_of(c), abd
            b['anchor'] = None if b.get('spline_cyls') else ('base' if abd else leg_of(c) + '_hip')
    for b in bodies:
        if 'servo' in b:
            continue
        k = kind_of(b['comp'])
        b['linkage'] = k != 'base'
        top = b['src'].split('+')[0].split('|')[0].rsplit(':', 1)[0]
        b['anchor'] = (leg_of(centre(b)) + '_' + k if k != 'base' else
                       'base' if (b['vol'] > 20 or top in ELEC) else None)


PART_OF = (('Quad Link', 'quad'), ('Femur', 'femur'), ('Component77', 'link'), ('Calf Link', 'link'), ('Main Foot', 'foot'),
           ('Foot', 'foot'), ('Spur Gear', 'gear'))


def mechanism(bodies, link, edges):
    """Split each leg link into its mechanism parts and locate the pins (source frame, cm; every pin runs along x).
    hip: bracket + both clamped servo cases + caps + hip-servo spline | pinion: gear-servo spline + the gear on it
    gear: the gear it drives (1:1, with the crank arm), turning on the hip axis | femur: femur + pivot-servo spline
    (+ horn screw + the gear's spacer/washer it clamps) | quad: Quad Link | tibia: tibia + knee pin | foot | link:
    Component77 (front) / Calf Link (rear). Pins: H hip, K knee, Q1 crank-Quad Link, Q2 Quad Link-tibia, A ankle,
    C femur-link, Cp link-foot, P gear-servo axis."""
    is_pinion = lambda b: b['comp'] == 'm' or b['comp'].startswith('m(')
    pinion_splines = {i for i, j, why in edges if why == 'on spline' and is_pinion(bodies[j])}
    part = {}
    for b in bodies:
        l = link[b['id']]
        if l == 'base':
            part[b['id']] = 'base'
            continue
        leg, seg = l.split('_')
        role = next((r for k, r in PART_OF if b['comp'].startswith(k)), None)
        if is_pinion(b) or b['id'] in pinion_splines:
            role = 'pinion'
        elif role == 'gear' and b['vol'] < 1.0:
            role = 'femur'                               # spacer + washer under the horn screw
        elif role is None:
            role = {'hip': 'hip', 'thigh': 'femur', 'calf': 'tibia'}[seg]
        part[b['id']] = leg + '_' + role
    pivots = {}
    for leg in ('FL', 'FR', 'RL', 'RR'):
        def xl(role, comp=''):
            return [(p[1], p[2], h[6], iv[0], iv[1]) for b in bodies if part[b['id']] == leg + '_' + role and b['comp'].startswith(comp)
                    for h in b.get('holes', []) for u, p, iv in [line_key(h)] if abs(u[0]) > 0.999]

        def pin(ra, rb):
            for y, z, r, a0, a1 in xl(ra):
                for y2, z2, r2, b0, b1 in xl(rb):
                    if abs(y - y2) < 0.03 and abs(z - z2) < 0.03:
                        return [round((min(a0, b0) + max(a1, b1)) / 2, 4), y, z]
            raise AssertionError('%s: no pin between %s and %s' % (leg, ra, rb))
        spline = next(np.array(splines_xyz[s]) for s in splines_xyz if part[s] == leg + '_femur')
        bores = [(y, z, a0, a1) for y, z, r, a0, a1 in xl('femur', 'Femur') if r > 0.13]   # the femur's own hip + knee bores
        H = min(bores, key=lambda q: np.hypot(q[0] - spline[1], q[1] - spline[2]))
        K = max(bores, key=lambda q: np.hypot(q[0] - H[0], q[1] - H[1]))
        P = next(np.array(splines_xyz[s]) for s in splines_xyz if part[s] == leg + '_pinion')
        pv = {'H': [round((H[2] + H[3]) / 2, 4), H[0], H[1]], 'K': [round((K[2] + K[3]) / 2, 4), K[0], K[1]],
              'Q1': pin('gear', 'quad'), 'Q2': pin('quad', 'tibia'), 'A': pin('tibia', 'foot'), 'C': pin('femur', 'link'),
              'Cp': pin('link', 'foot'), 'P': [round(float(P[0]), 4), round(float(P[1]), 5), round(float(P[2]), 5)]}
        v = {k: np.array(q[1:]) for k, q in pv.items()}
        err = np.linalg.norm((v['Q1'] - v['H']) - (v['Q2'] - v['K'])) + np.linalg.norm((v['K'] - v['H']) - (v['Q2'] - v['Q1']))
        assert err < 0.01, '%s: hip-crank-Quad Link-knee loop is not a parallelogram (%.4f cm)' % (leg, err)
        pivots[leg] = pv
    return part, pivots


def main():
    scan_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'analysis', 'screw_scan.json')
    scan = json.load(open(scan_path))
    bodies = scan['bodies']
    anchor_and_roles(bodies)
    splines = {}                                        # spline body id -> output axis (largest cylinder), axial extent
    for b in bodies:
        if b.get('spline_cyls'):
            u, p, _ = line_key(max(b['spline_cyls'], key=lambda h: h[6]))
            ivs = [line_key(h)[2] for h in b['spline_cyls'] if coaxial(u, p, *line_key(h)[:2])]
            splines[b['id']] = (u, p, [min(i[0] for i in ivs), max(i[1] for i in ivs)])
    global splines_xyz                                  # spline id -> point on its axis at its mid-length (source, cm)
    splines_xyz = {i: (p + u * (iv[0] + iv[1]) / 2).tolist() for i, (u, p, iv) in splines.items()}
    clusters = []                                       # greedy coaxial clustering of every hole line
    for b in bodies:
        for h in b.get('holes', []):
            u, p, iv = line_key(h)
            for c in clusters:
                if coaxial(c['u'], c['p'], u, p):
                    c['m'].append((b['id'], iv))
                    break
            else:
                clusters.append({'u': u, 'p': p, 'm': [(b['id'], iv)]})
    pairs = defaultdict(list)                           # cluster -> touching body pairs
    for ci, c in enumerate(clusters):
        per = defaultdict(list)
        for bid, iv in c['m']:
            per[bid].append(iv)
        ids = sorted(per)
        for x, i in enumerate(ids):
            for j in ids[x + 1:]:
                if any(touch(u, v) for u in per[i] for v in per[j]):
                    pairs[ci].append((i, j))
        on_spline = any(coaxial(u, p, c['u'], c['p'], SPLINE_TOL) for u, p, _ in splines.values())
        c['pivot'] = on_spline or any(bodies[i].get('linkage') and bodies[j].get('linkage') for i, j in pairs[ci])
    parent = list(range(len(bodies)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = [(i, j, 'screw') for ci, c in enumerate(clusters) if not c['pivot'] for i, j in pairs[ci]]
    for s, (u, p, iv) in splines.items():              # whatever sits on a spline turns with it (horn, femur, pinion, bracket)
        for b in bodies:
            if b['id'] != s and b.get('servo') != bodies[s]['servo'] and any(
                    coaxial(u, p, *line_key(h)[:2], SPLINE_TOL) and touch(iv, line_key(h)[2]) for h in b.get('holes', [])):
                edges.append((s, b['id'], 'on spline'))
    for i, j, why in edges:
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for b in bodies:
        groups[find(b['id'])].append(b['id'])
    link, conflicts = {}, []
    for g, ids in groups.items():
        anchors = {bodies[i]['anchor'] for i in ids if bodies[i]['anchor']}
        if len(anchors) > 1:
            conflicts.append({'anchors': sorted(anchors), 'members': [bodies[i]['src'] for i in ids]})
        elif anchors:
            a = anchors.pop()
            for i in ids:
                link[i] = a
    for g, ids in groups.items():                      # loose parts: pins on pivot lines -> most distal link on the line
        if ids[0] in link:
            continue
        cand = [link[k] for ci, c in enumerate(clusters) if c['pivot'] for i, j in pairs[ci] for k in (i, j)
                if (i in ids or j in ids) and k not in ids and k in link]
        if cand:
            for i in ids:
                link[i] = max(cand, key=lambda l: DEPTH[l.split('_')[-1]])
    for b in bodies:                                    # anything left (no holes): the resolved body whose box it touches most
        if b['id'] in link:
            continue
        def overlap(o):
            d = [min(b['bbmax'][k], o['bbmax'][k]) - max(b['bbmin'][k], o['bbmin'][k]) for k in range(3)]
            return min(d)
        best = max((o for o in bodies if o['id'] in link), key=lambda o: (round(overlap(o), 3), -o['vol']))  # tie: most specific
        link[b['id']] = link[best['id']]
        b['touch'] = [best['src'], round(overlap(best), 4)]
    default = lambda b: ('base' if b['abd'] else b['leg'] + '_hip') if 'servo' in b else (b['anchor'] or 'base')  # previous rule
    moved = [[b['src'], default(b), link[b['id']], b['vol'], b.get('touch')] for b in bodies if link[b['id']] != default(b)]
    abd_axes = {}
    for s, (u, p, _) in splines.items():
        b = bodies[s]
        if b['abd']:
            d = -u if u[1] > 0 else u                  # source -y = model +x (forward)
            q = p + np.dot(np.array(centre(b)) - p, d) * d
            abd_axes[b['leg']] = {'point_cm': np.round(q, 5).tolist(), 'dir': np.round(d / np.linalg.norm(d), 6).tolist()}
    part, pivots = mechanism(bodies, link, edges)
    out = {'scan': os.path.basename(scan_path), 'suppressed': scan.get('suppressed'), 'conflicts': conflicts,
           'links': {b['src']: [link[b['id']], b['vol']] for b in bodies}, 'abduction_axes': abd_axes,
           'parts': {b['src']: part[b['id']] for b in bodies}, 'pivots': pivots,
           'moved_vs_previous_rule': moved,
           'edges': [[bodies[i]['src'], bodies[j]['src'], why] for i, j, why in edges if not (bodies[i]['src'].startswith(ELEC) and bodies[j]['src'].startswith(ELEC))]}
    assert sorted(abd_axes) == ['FL', 'FR', 'RL', 'RR'], abd_axes
    assert {i for i, j, why in edges if why == 'on spline'} == set(splines), 'a servo spline drives nothing'
    if not conflicts:
        json.dump(out, open(os.path.join(ROOT, 'analysis', 'link_map.json'), 'w'), indent=1)
    print('bodies %d  lines %d  pivot lines %d  rigid edges %d  conflicts %d  changed vs previous model %d'
          % (len(bodies), len(clusters), sum(c['pivot'] for c in clusters), len(edges), len(conflicts), len(moved)))
    for c in conflicts:
        print('CONFLICT', c)
    for m in moved:
        print('  %-66s %-9s -> %-9s %7.3f %s' % (m[0][:66], m[1], m[2], m[3], m[4] or ''))
    for leg, a in sorted(abd_axes.items()):
        print('  abduction', leg, a)
    from collections import Counter
    print('  parts:', dict(sorted(Counter(part.values()).items())))
    for leg, pv in pivots.items():
        print('  pivots', leg, {k: [round(c, 3) for c in q] for k, q in pv.items()})
    return out


if __name__ == '__main__':
    main()
