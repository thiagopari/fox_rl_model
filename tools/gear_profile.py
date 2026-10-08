#!/usr/bin/env python3
"""Tooth outlines for the v7 2:1 gear-servo drive -> tools/fusion/FoxMakeV7/profiles.json (mm, centred on the axis).

24T crank gear + 12T pinion, module 1.0, 20 deg, profile shift -0.3 / +0.3 (the 12T pinion would undercut unshifted; the
shifts cancel, so the centre distance stays the standard 18 mm of the 12:12 m1.5 pair it replaces). Each outline is cut
by rolling an ISO 53 A rack (tip radius 0.38 m) past a blank, so the root fillets are the real generated ones, and every
tooth is THIN mm thinner at the pitch circle for print backlash. Self-check: the pair is rolled through a gear pitch at
2:1 and must never overlap; the free play and tip lands are printed.
Usage:  ~/.venvs/fox_rl/bin/python tools/gear_profile.py
"""
import json, math, os
import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

M, ALPHA, A = 1.0, math.radians(20.0), 18.0
THIN = 0.10                  # mm off each tooth's circular thickness at the pitch circle (0.2 mm total backlash)
RHO = 0.38 * M               # rack tip radius -> root fillet
SIMPLIFY = 0.004             # mm, outline simplification for the Fusion sketch
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fusion', 'FoxMakeV7', 'profiles.json')


def rack_tooth(m, thin, n_arc=10):
    """One cutter tooth pointing down (-v), datum line v = 0, half-width pi m / 4 + thin / 2 there; flanks at ALPHA, tip at
    v = -1.25 m rounded with RHO. Above v = +1 m it never reaches the blank, so it simply ends."""
    hw0, ha, vt, t = math.pi * m / 4 + thin / 2, 1.25 * m, 1.5 * m, math.tan(ALPHA)
    vc = -ha + RHO
    uc = hw0 + t * vc - RHO * math.sqrt(1 + t * t)
    assert uc > 0, 'tip fillets overlap'
    arc = [(uc + RHO * math.cos(a), vc + RHO * math.sin(a)) for a in np.linspace(-ALPHA, -math.pi / 2, n_arc)]
    right = [(hw0 + t * vt, vt)] + arc
    poly = Polygon(right + [(-u, v) for u, v in reversed(right)])
    assert poly.is_valid
    return poly


def gear(z, x, m=M, thin=THIN, step_deg=0.08):
    """Outline of a z-tooth gear with profile shift x; a tooth SPACE is centred on +y."""
    r = m * z / 2
    ra, rf = r + m * (1 + x), r - m * (1.25 - x)
    tooth, span = rack_tooth(m, thin), math.radians(60.0) * 12 / z
    cuts = []
    for ph in np.arange(-span, span + 1e-9, math.radians(step_deg)):   # rack rolls on the pitch circle, datum at r + x m
        p = affinity.translate(tooth, xoff=-r * ph, yoff=r + x * m)
        cuts.append(affinity.rotate(p, -ph, origin=(0, 0), use_radians=True))
    gap = unary_union(cuts)
    gaps = unary_union([affinity.rotate(gap, 2 * math.pi * k / z, origin=(0, 0), use_radians=True) for k in range(z)])
    return Point(0, 0).buffer(ra, 2048).difference(gaps), dict(z=z, x=x, m=m, r=r, ra=ra, rf=rf, rb=r * math.cos(ALPHA))


def main():
    G, gi = gear(24, -0.3)
    P, pi_ = gear(12, +0.3)
    P = affinity.rotate(P, 180.0 / 12, origin=(0, 0))                 # pinion: a TOOTH centred on +y
    P_mesh = affinity.translate(affinity.rotate(P, 180.0, origin=(0, 0)), yoff=A)   # at (0, A), tooth toward the gear
    worst, clear = 0.0, 9.0
    for th in np.linspace(-7.5, 7.5, 61):                                # gear th, pinion -2 th: one gear pitch
        g, p = affinity.rotate(G, th, origin=(0, 0)), affinity.rotate(P_mesh, -2 * th, origin=(0, A))
        worst, clear = max(worst, g.intersection(p).area), min(clear, g.distance(p))
    assert worst < 1e-6, 'teeth overlap while meshing: %.6f mm2' % worst

    def play(sign):
        return next(d for d in np.arange(0, 10, 0.02) if G.intersection(affinity.rotate(P_mesh, sign * d, origin=(0, A))).area > 1e-6)
    lands = {}
    for name, poly, info in (('pinion', P, pi_), ('gear', G, gi)):
        ring = Point(0, 0).buffer(info['ra'] + 0.01, 2048).difference(Point(0, 0).buffer(info['ra'] - 0.05, 2048))
        lands[name] = poly.intersection(ring).area / info['z'] / 0.05
    rb1, rb2 = pi_['rb'], gi['rb']
    eps = (math.sqrt(pi_['ra'] ** 2 - rb1 ** 2) + math.sqrt(gi['ra'] ** 2 - rb2 ** 2) - A * math.sin(ALPHA)) / (math.pi * M * math.cos(ALPHA))
    out = {name: {'info': info, 'outline': np.round(np.array(poly.simplify(SIMPLIFY, preserve_topology=True).exterior.coords)[:-1], 5).tolist()}
           for name, poly, info in (('gear', G, gi), ('pinion', P, pi_))}
    out['note'] = ('mm, centred on the axis; gear: tooth SPACE centred on +y; pinion: TOOTH centred on +y; module %.1f, 20 deg, '
                   'teeth %.2f mm thin, backlash %.2f mm at the pinion pitch circle' % (M, THIN, (play(1) + play(-1)) * math.pi / 180 * pi_['r']))
    json.dump(out, open(OUT, 'w'))
    print('24T gear  ra %.3f rf %.3f | 12T pinion ra %.3f rf %.3f | contact ratio %.2f' % (gi['ra'], gi['rf'], pi_['ra'], pi_['rf'], eps))
    print('meshing: no overlap, min clearance %.3f mm, pinion free play +-%.2f deg; tip lands pinion %.2f mm, gear %.2f mm'
          % (clear, play(1), lands['pinion'], lands['gear']))
    print('wrote %s (%d + %d outline points)' % (OUT, len(out['gear']['outline']), len(out['pinion']['outline'])))


if __name__ == '__main__':
    main()
