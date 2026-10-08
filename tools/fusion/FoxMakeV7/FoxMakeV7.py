# One-shot Fusion add-in. Copy this folder into
#   <prefix>/drive_c/users/<you>/AppData/Roaming/Autodesk/Autodesk Fusion 360/API/AddIns/
# start Fusion, wait for C:\fusion_jobs\make_v7.json, then delete the folder again (it runs at every start).
#
# Makes NEW_NAME, the design copy 'Fox_Prototype_03_v6_RL' with the mechanical improvements: the source is opened,
# edited in memory and saved under the new name only if every check passes (the source itself is never saved).
#   1. 2:1 gear-servo drive: the 12T m1.5 pinion -> 24T crank gear pair (1:1) becomes 12T -> 24T, module 1.0, 20 deg,
#      profile shift +0.3 / -0.3, on the same 18 mm centres (outlines: profiles.json from tools/gear_profile.py). The old
#      teeth are cut off and the new ones joined; gear and pinion are shared components, so all 4 legs change. The gear
#      teeth now start 0.3 mm above the gear plate like the pinion's, so they clear the pinion hub, and the plate's
#      chamfer under them is filled (2.3 mm print overhang instead of 3.7).
#   2. Rear hips 20 mm higher: everything behind REAR_Y except the plate (rear pelvis, rear legs and servos, the boards on
#      the rear pelvis) moves LIFT up the two rear pelvis screws (20 deg from vertical) onto a new printed spacer
#      ('Rear Hip Spacer', screws 20 mm longer). The rear legs reach 20 mm further down to stand level.
#   3. The rear pinions turn about their own axis so the rear gear pairs mesh as well (the front pairs fix the phase).
#   4. The three Combine features that fused both rear hip brackets into the rear pelvis are suppressed: the brackets turn
#      with the hip servos (FoxScrewScan / FoxBuildLinkModel always suppressed them in memory), and the fused body would
#      not follow the rear module when it moves.
# Optional make.json next to this file (defaults = the v7 above): new_name, lift (cm; 0 = rear not raised), spacer (bool),
# bushing ('printed': Ø10.4 running-fit gear seat with the lip 0.3 mm below the clamped bushing, '623': Ø10.0 press fit
# for a 623ZZ; both relieve the femur hub to a Ø4.4 boss), m4_casing_holes (bool: the 4 pelvis screw holes of the
# casing Ø4.6), shots / result (render folder, result file). v7b = no spacer, printed bushings, M4 holes.
# Checks: no tooth overlap on any leg, the rear moved by exactly the lift and nothing else moved, no new interference,
# spacer seated on both faces, timeline healthy. Renders go to C:\fusion_jobs\<shots>\.
import adsk.core, adsk.fusion, json, math, os, threading, time, traceback

DRY_RUN = False                       # True: build + check + render, then discard
COPY_ID, SOURCE_NAME = 'urn:adsk.wipprod:dm.lineage:sWQJByHaTNKGDxKGjmsT6g', 'Fox_Prototype_03_v6_RL'
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make.json')
_cfg = json.load(open(_cfg_path)) if os.path.exists(_cfg_path) else {}
NEW_NAME = _cfg.get('new_name', 'Fox_Prototype_03_v7_RL')
HERE = os.path.dirname(os.path.abspath(__file__))
PROF = json.load(open(os.path.join(HERE, 'profiles.json')))
PIV = json.load(open(os.path.join(HERE, 'link_map.json')))['pivots']   # v6 pins, source frame (cm)
OUT = r'C:\fusion_jobs'
SHOTS = os.path.join(OUT, _cfg.get('shots', 'v7'))
RESULT = _cfg.get('result', 'make_v7.json')
MID_X, REAR_Y = -2.476478, -6.0      # sagittal plane; the rear module lies behind source y = REAR_Y ...
HALF_WIDTH = 7.0                     # ... within this of the sagittal plane (a spare battery bay sits beside the robot)
UNJOIN = [('Combine1', 'Component41(Mirror) (1)'), ('Combine2', 'Component41(Mirror) (1)'), ('Combine2', 'Servo Pelv Upper')]
PLATE = 'raspberry_pi_mount'         # root body that carries both pelvises: stays
LIFT = float(_cfg.get('lift', 2.0))  # cm along the rear screws
SPACER_ON = bool(_cfg.get('spacer', True)) and LIFT > 0
BUSHING = _cfg.get('bushing')         # None | 'printed' | '623'
M4_HOLES = bool(_cfg.get('m4_casing_holes', False))
FEMURS = {'FL': 'Front Leg:1+Femur(Mirror) (1):1', 'FR': 'Front Leg(Mirror):1+Femur(Mirror) (1)(Mirror):1',
          'RL': 'Hind Leg:1+Femur:1', 'RR': 'Hind Leg(Mirror):1+Femur(Mirror) (2):1'}
CASING_HOLES = [((-1.695, -3.263, 3.171), (0.0, -0.342020, -0.939693)), ((-3.258, -3.263, 3.171), (0.0, -0.342020, -0.939693)),
                ((-3.258, -13.444, 3.524), (0.0, 0.173648, -0.984808)), ((-1.695, -13.444, 3.524), (0.0, 0.173648, -0.984808))]
DOWN = (0.0, -0.342020, -0.939693)   # rear pelvis screw axis, pointing down (hole scan)
SCREWS = ((-1.695, -3.263, 3.171), (-3.258, -3.263, 3.171))   # where the screws leave the pelvis flange top: s = 0
# spacer, in s along DOWN (pelvis flange bottom 0.80 before the lift, plate tab top 0.82), u along x, w across (+w toward
# the plate's step, which has a fillet at its foot: kept 1.5 mm clear)
SPACER = dict(s0=0.80 - LIFT, s1=0.82, half_u=1.25, w0=-0.49, w1=0.40, hole_r=0.16)
# gear parts: component, the front occurrence that fixes the tooth phase, the rear one, bodies -> front leg, the pin
# the part turns on, and the edits as (from, to) cm from the body's inboard end + radii (cm)
PARTS = [dict(kind='gear', comp='Spur Gear (15 teeth)', occ='Spur Gear (15 teeth):2', rear='Hind Leg:1+Spur Gear (15 teeth):1',
              bodies={'Body1': 'FL', 'Body4': 'FR'}, pin='H', length=1.0, cut=(0.30, 1.05, 0.69, 1.20),
              fill=(0.15, 0.33, 0.68, 1.045), teeth=(0.33, 1.00, 0.68), rename='Spur Gear (24 teeth)'),
         dict(kind='pinion', comp='m(Mirror)', occ='Pevis(Mirror) (1):1+m(Mirror):1', rear='m(Mirror):2',
              bodies={'Body1': 'FL', 'Body2': 'FR'}, pin='P', length=1.0, cut=(0.33, 1.05, 0.49, 1.20),
              fill=None, teeth=(0.33, 1.00, 0.48), rename=None)]
PAIRS = {'FL': (('Spur Gear (15 teeth):2', 'Body1'), ('Pevis(Mirror) (1):1+m(Mirror):1', 'Body1')),
         'FR': (('Spur Gear (15 teeth):2', 'Body4'), ('Pevis(Mirror) (1):1+m(Mirror):1', 'Body2')),
         'RL': (('Hind Leg:1+Spur Gear (15 teeth):1', 'Body1'), ('m(Mirror):2', 'Body1')),
         'RR': (('Hind Leg:1+Spur Gear (15 teeth):1', 'Body4'), ('m(Mirror):2', 'Body2'))}
NEW_INTERFERENCE = 1e-3              # cm^3 (1 mm^3)
EVT = 'FoxMakeV7Run'
_handlers, _res, _done = [], {'checks': {}}, threading.Event()


def check(name, ok, detail=None):
    _res['checks'][name] = {'ok': bool(ok), 'detail': detail}
    return ok


def P3(x, y, z):
    return adsk.core.Point3D.create(x, y, z)


def V3(x, y, z):
    return adsk.core.Vector3D.create(x, y, z)


def moved(p, m):
    q = p.copy()
    q.transformBy(m)
    return q


def inverse(m):
    m = m.copy()
    m.invert()
    return m


def wrap(a, period):                 # -> (-period/2, period/2]
    a = math.fmod(a + period / 2, period)
    return (a + period if a <= 0 else a) - period / 2


def rot3(m):
    a = m.asArray()
    return [[a[0], a[1], a[2]], [a[4], a[5], a[6]], [a[8], a[9], a[10]]]


def x_turn(ma, mb):
    """Angle about world x taking placement ma's orientation to mb's, and how far the relative rotation is from a pure
    x rotation (0 = pure)."""
    A, B = rot3(ma), rot3(mb)
    rel = [[sum(B[i][k] * A[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    return math.atan2(rel[2][1], rel[1][1]), abs(rel[0][0] - 1) + abs(rel[0][1]) + abs(rel[0][2]) + abs(rel[1][0]) + abs(rel[2][0])


def owner(t):
    try:
        return t.entity.parentComponent.name
    except Exception:
        return None


def by_path(root, path):
    for i in range(root.allOccurrences.count):
        o = root.allOccurrences.item(i)
        if o.fullPathName == path:
            return o
    return None


def line_angle(leg):                 # direction gear pin -> pinion pin in the source y-z plane (rad)
    h, p = PIV[leg]['H'], PIV[leg]['P']
    return math.atan2(p[2] - h[2], p[1] - h[1])


def exact_copy(body, occ, mgr):      # world copy of a (proxy) body; keep whichever method matches it exactly
    def dev(src, temp):
        a, b = src.boundingBox, temp.boundingBox
        return max(abs(a.minPoint.x - b.minPoint.x), abs(a.maxPoint.y - b.maxPoint.y), abs(a.minPoint.z - b.minPoint.z),
                   abs(a.maxPoint.x - b.maxPoint.x), abs(a.minPoint.y - b.minPoint.y), abs(a.maxPoint.z - b.maxPoint.z))
    cands = []
    if occ is not None:
        t = mgr.copy(body.nativeObject or body)
        mgr.transform(t, occ.transform2)
        cands.append(t)
    cands.append(mgr.copy(body))
    return min(((dev(body, t), t) for t in cands), key=lambda x: x[0])


def bodies(root):                    # every visible solid body: key -> (proxy body, occurrence or None)
    out = {}
    for b in root.bRepBodies:
        if b.isVisible and b.isSolid and b.volume > 1e-6:
            out['root|' + b.name] = (b, None)
    for i in range(root.allOccurrences.count):
        o = root.allOccurrences.item(i)
        for b in o.bRepBodies:
            if b.isVisible and b.isSolid and b.volume > 1e-6:
                out[o.fullPathName + '|' + b.name] = (b, o)
    return out


def box_of(b):
    bb = b.boundingBox
    return [bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z]


def overlap(a, b, pad=0.0):
    return all(a[i] - pad <= b[i + 3] and b[i] - pad <= a[i + 3] for i in range(3))


def common_volume(a, b, mgr):
    t = mgr.copy(a)
    try:
        mgr.booleanOperation(t, b, adsk.fusion.BooleanTypes.IntersectionBooleanType)
        return t.volume
    except Exception:                # no common volume, or a boolean Fusion could not do: counted
        _res['boolean_failures'] = _res.get('boolean_failures', 0) + 1
        return 0.0


def plane_at(comp, z):
    pi = comp.constructionPlanes.createInput()
    pi.setByOffset(comp.xYConstructionPlane, adsk.core.ValueInput.createByReal(z))
    pl = comp.constructionPlanes.add(pi)
    pl.isLightBulbOn = False
    return pl


def ring_profile(sk):                # the profile bounded by two loops (outline/outer circle + inner circle)
    ps = [sk.profiles.item(i) for i in range(sk.profiles.count)]
    two = [p for p in ps if p.profileLoops.count == 2]
    return two[0] if len(two) == 1 else None


def extrude(comp, sk, z0, z1, op, body, name):
    prof = ring_profile(sk)
    if prof is None:
        raise RuntimeError('%s: no two-loop profile in %s (%d profiles)' % (name, sk.name, sk.profiles.count))
    normal = sk.transform.getAsCoordinateSystem()[-1]         # sketch z in component space
    up = (z1 - z0) * normal.z > 0
    ext = comp.features.extrudeFeatures
    ei = ext.createInput(prof, op)
    ei.setOneSideExtent(adsk.fusion.DistanceExtentDefinition.create(adsk.core.ValueInput.createByReal(abs(z1 - z0))),
                        adsk.fusion.ExtentDirections.PositiveExtentDirection if up else adsk.fusion.ExtentDirections.NegativeExtentDirection)
    ei.participantBodies = [body]
    f = ext.add(ei)
    f.name = name
    sk.isVisible = False
    return f


def edit_part(spec, root, mgr):
    """Cut the old teeth off the part's two bodies and join the new ones, phased at the front leg to mesh."""
    occ = by_path(root, spec['occ'])
    comp = occ.component
    mi = inverse(occ.transform2)
    info = PROF[spec['kind']]
    outline = info['outline']
    report = {}
    for bname, leg in spec['bodies'].items():
        body = comp.bRepBodies.itemByName(bname)
        pin = PIV[leg][spec['pin']]
        a = moved(P3(*pin), mi)                                # axis point in component space
        ends = [f.geometry.origin.z for f in body.faces if f.geometry.objectType == adsk.core.Plane.classType()
                and abs(abs(f.geometry.normal.z) - 1) < 1e-6]
        z0, z1 = min(ends), max(ends)
        xw = lambda z: moved(P3(a.x, a.y, z), occ.transform2).x
        if not check('%s %s %s: %.1f cm long along component z' % (spec['kind'], leg, bname, spec['length']),
                     abs((z1 - z0) - spec['length']) < 0.002 and abs(abs(xw(z1) - xw(z0)) - spec['length']) < 0.002,
                     [round(z0, 4), round(z1, 4)]):
            return None
        e0, s = (z0, 1.0) if abs(xw(z0) - MID_X) < abs(xw(z1) - MID_X) else (z1, -1.0)   # inboard end, outward sign
        bore = [round(math.hypot(f.geometry.origin.x - a.x, f.geometry.origin.y - a.y), 4) for f in body.faces
                if f.geometry.objectType == adsk.core.Cylinder.classType() and abs(abs(f.geometry.axis.z) - 1) < 1e-6
                and f.geometry.radius < 0.65]
        if not check('%s %s: its bores are on the %s pin' % (spec['kind'], leg, spec['pin']), bore and min(bore) < 0.002, bore[:6]):
            return None
        vol0 = body.volume
        th = line_angle(leg)
        turn = th - math.pi / 2 if spec['kind'] == 'gear' else th + math.pi / 2   # gear: space toward the pinion; pinion: tooth toward the gear
        c, sn = math.cos(turn), math.sin(turn)

        def world_pt(px, py):        # outline point (mm) -> component space, on the axis plane through a
            y = pin[1] + (c * px - sn * py) / 10.0
            z = pin[2] + (sn * px + c * py) / 10.0
            return moved(P3(pin[0], y, z), mi)

        def ring_sketch(z, r_in, r_out=None, name=''):
            sk = comp.sketches.add(plane_at(comp, z))
            sk.name = name
            sk.isComputeDeferred = True
            centre = sk.modelToSketchSpace(P3(a.x, a.y, z))
            if r_out is None:        # the tooth outline
                pts = []
                for px, py in outline:
                    q = world_pt(px, py)
                    pts.append(sk.modelToSketchSpace(P3(q.x, q.y, z)))
                lines = sk.sketchCurves.sketchLines
                first = prev = lines.addByTwoPoints(pts[0], pts[1])
                for p in pts[2:]:
                    prev = lines.addByTwoPoints(prev.endSketchPoint, p)
                lines.addByTwoPoints(prev.endSketchPoint, first.startSketchPoint)
            else:
                sk.sketchCurves.sketchCircles.addByCenterRadius(centre, r_out)
            sk.sketchCurves.sketchCircles.addByCenterRadius(centre, r_in)
            sk.isComputeDeferred = False
            return sk

        tag = '%s %s' % ('Crank gear 24T' if spec['kind'] == 'gear' else 'Pinion 12T', leg)
        f0, f1, r0, r1 = spec['cut']
        extrude(comp, ring_sketch(e0 + s * f0, r0, r1, tag + ' cut'), e0 + s * f0, e0 + s * f1,
                adsk.fusion.FeatureOperations.CutFeatureOperation, body, tag + ': old teeth off')
        if spec['fill']:
            f0, f1, r0, r1 = spec['fill']
            extrude(comp, ring_sketch(e0 + s * f0, r0, r1, tag + ' fill'), e0 + s * f0, e0 + s * f1,
                    adsk.fusion.FeatureOperations.JoinFeatureOperation, comp.bRepBodies.itemByName(bname), tag + ': plate under the teeth')
        f0, f1, r0 = spec['teeth']
        extrude(comp, ring_sketch(e0 + s * f0, r0, None, tag + ' teeth'), e0 + s * f0, e0 + s * f1,
                adsk.fusion.FeatureOperations.JoinFeatureOperation, comp.bRepBodies.itemByName(bname), tag + ': teeth m1.0')
        body = comp.bRepBodies.itemByName(bname)
        report[leg] = {'volume_before': round(vol0, 4), 'volume_after': round(body.volume, 4), 'lumps': body.lumps.count,
                       'inboard_end_z': round(e0, 4), 'outward': s, 'phase_turn_deg': round(math.degrees(turn), 3)}
        check('%s %s: one solid after the edit' % (spec['kind'], leg), body.isSolid and body.lumps.count == 1, report[leg])
    return report


def tube(mgr, p, d, s0, s1, r_in, r_out):        # world cylinder (r_in 0) or tube along the line p + s d, s0..s1
    at = lambda s: P3(p[0] + d[0] * s, p[1] + d[1] * s, p[2] + d[2] * s)     # noqa: E731
    outer = mgr.createCylinderOrCone(at(s0), r_out, at(s1), r_out)
    if r_in > 0:
        mgr.booleanOperation(outer, mgr.createCylinderOrCone(at(s0), r_in, at(s1), r_in), adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return outer


def combine(comp, target, tool_world, occ, op, name, mgr):
    """Add a world-space tool body to the component (base feature) and combine it with the target body."""
    if occ is not None:
        mgr.transform(tool_world, inverse(occ.transform2))
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    comp.bRepBodies.add(tool_world, bf)
    bf.finishEdit()
    bf.name = name + ' (tool)'
    tools = adsk.core.ObjectCollection.create()
    tools.add(bf.bodies.item(0))
    ci = comp.features.combineFeatures.createInput(target, tools)
    ci.operation, ci.isKeepToolBodies = op, False
    comp.features.combineFeatures.add(ci).name = name


def bushing_edits(root, mgr, lift_v):
    """Femur hubs: Ø4.4 boss + 0.3 mm relief (they clamp only the bushing / bearing inner ring). Gear seat: 'printed' =
    Ø10.4 running fit with the lip 0.3 mm below the clamped bushing; '623' = Ø10.0 press fit. Shared gear component."""
    X = (1, 0, 0)
    for leg, path in FEMURS.items():
        occ = by_path(root, path)
        if not check('%s femur found, its component used once' % leg, occ is not None and root.allOccurrencesByComponent(occ.component).count == 1, path):
            return False
        H = [PIV[leg]['H'][i] + (lift_v[i] if leg[0] == 'R' else 0.0) for i in range(3)]
        s = 1.0 if H[0] > MID_X else -1.0
        body = occ.bRepBodies.itemByName('Body1')
        ends = [f.pointOnFace.x for f in body.faces if f.geometry.objectType == adsk.core.Plane.classType()
                and abs(abs(f.geometry.normal.x) - 1) < 1e-6 and math.hypot(f.pointOnFace.y - H[1], f.pointOnFace.z - H[2]) < 0.305]
        if not check('%s femur: hub end face found' % leg, ends, ends):
            return False
        x_end = max(ends) if s > 0 else min(ends)
        comp, native = occ.component, body.nativeObject
        v0 = native.volume
        combine(comp, native, tube(mgr, (x_end - s * 0.03, H[1], H[2]), X, 0.0, s * 0.08, 0.22, 0.32), occ,
                adsk.fusion.FeatureOperations.CutFeatureOperation, 'Hub relief: bushing inner part only', mgr)
        native = comp.bRepBodies.itemByName('Body1')
        want = math.pi * (0.30 ** 2 - 0.22 ** 2) * 0.03
        check('%s femur hub relief' % leg, abs((v0 - native.volume) - want) < 0.0003 and native.lumps.count == 1, [round(v0 - native.volume, 5), round(want, 5)])
    gocc = by_path(root, PARTS[0]['occ'])
    for bname, leg in PARTS[0]['bodies'].items():
        native = gocc.component.bRepBodies.itemByName(bname)
        H = PIV[leg]['H']
        s = 1.0 if H[0] > MID_X else -1.0
        bb = gocc.bRepBodies.itemByName(bname).boundingBox
        x_in = bb.minPoint.x if s > 0 else bb.maxPoint.x
        v0 = native.volume
        if BUSHING == 'printed':
            tool, op, want = tube(mgr, (x_in, H[1], H[2]), X, s * 0.07, s * 0.545, 0.0, 0.52), adsk.fusion.FeatureOperations.CutFeatureOperation, \
                -(math.pi * (0.52 ** 2 - 0.51 ** 2) * 0.44 + math.pi * (0.52 ** 2 - 0.40 ** 2) * 0.03)
        else:
            tool, op, want = tube(mgr, (x_in, H[1], H[2]), X, s * 0.10, s * 0.54, 0.50, 0.53), adsk.fusion.FeatureOperations.JoinFeatureOperation, \
                math.pi * (0.51 ** 2 - 0.50 ** 2) * 0.44
        combine(gocc.component, native, tool, gocc, op, 'Seat for a %s' % ('printed bushing (10.4 mm)' if BUSHING == 'printed' else '623ZZ (10.0 mm)'), mgr)
        native = gocc.component.bRepBodies.itemByName(bname)
        check('%s gear seat (%s)' % (leg, BUSHING), abs((native.volume - v0) - want) < 0.0005 and native.lumps.count == 1, [round(native.volume - v0, 5), round(want, 5)])
    return True


def casing_m4(root, mgr):
    """The casing's 4 pelvis screw holes Ø2.8 -> Ø4.6 (M4 clearance), along each tilted axis, over its 5 mm tab."""
    native = root.bRepBodies.itemByName(PLATE)
    for p, d in CASING_HOLES:
        shell = tube(mgr, p, d, 0.85, 1.29, 0.23, 0.33)
        t = mgr.copy(native)
        mgr.booleanOperation(t, shell, adsk.fusion.BooleanTypes.IntersectionBooleanType)
        if not check('casing: >= 1 mm wall around the M4 hole at %s' % (p,), t.volume / shell.volume > 0.999, round(t.volume / shell.volume, 4)):
            return False
    v0 = native.volume
    for k, (p, d) in enumerate(CASING_HOLES):
        combine(root, root.bRepBodies.itemByName(PLATE), tube(mgr, p, d, 0.72, 1.42, 0.0, 0.23), None,
                adsk.fusion.FeatureOperations.CutFeatureOperation, 'M4 clearance hole %d (4.6 mm)' % (k + 1), mgr)
    native = root.bRepBodies.itemByName(PLATE)
    want = 4 * math.pi * (0.23 ** 2 - 0.14 ** 2) * 0.50
    return check('casing: 4 M4 clearance holes, one solid', abs((v0 - native.volume) - want) < 0.02 * want and native.lumps.count == 1,
                 [round(v0 - native.volume, 5), round(want, 5)])


def render(app, name, target, eye, extent):
    vp = app.activeViewport
    cam = vp.camera
    cam.isSmoothTransition = False
    cam.cameraType = adsk.core.CameraTypes.OrthographicCameraType
    cam.target, cam.eye, cam.upVector = P3(*target), P3(*eye), V3(0, 0, 1)
    try:
        cam.viewExtents = extent
    except Exception:
        pass
    vp.camera = cam
    vp.refresh()
    adsk.doEvents()
    return vp.saveAsImageFile(os.path.join(SHOTS, name), 1600, 1000)


def _work():
    app = adsk.core.Application.get()
    try:
        df = app.data.findFileById(COPY_ID)
    except Exception:
        df = None
    if df is None or df.name != SOURCE_NAME:
        _res['status'] = 'retry: source not reachable yet'
        return
    folder = df.parentFolder
    files = folder.dataFiles                                 # fetched once: each access pages the cloud folder
    if any(files.item(i).name == NEW_NAME for i in range(files.count)):
        _res['status'] = 'ABORT: %s already exists' % NEW_NAME
        return
    doc = app.documents.open(df, True)
    _res['doc'], _res['source_version'] = doc, doc.dataFile.versionNumber
    design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
    root, tl, mgr = design.rootComponent, design.timeline, adsk.fusion.TemporaryBRepManager.get()
    if not check('timeline marker at the end', tl.markerPosition == tl.count, [tl.markerPosition, tl.count]):
        return
    sick = lambda: sorted(tl.item(i).name for i in range(tl.count) if int(tl.item(i).healthState) in (1, 2))
    sick0 = sick()
    hit = [i for i in range(tl.count) if (tl.item(i).name, owner(tl.item(i))) in UNJOIN]
    for i in sorted(hit, reverse=True):
        tl.item(i).isSuppressed = True
    adsk.doEvents()
    _res['unjoined'] = [[tl.item(i).name, owner(tl.item(i))] for i in hit]
    if not check('rear hip brackets un-fused from the pelvis (3 Combine features suppressed), no new errors',
                 len(hit) == len(UNJOIN) and set(sick()) <= set(sick0), {'found': _res['unjoined'], 'sick': sick()}):
        return
    _res['joints'] = {'joints': sum(1 for _ in root.allJoints), 'as_built': sum(1 for _ in root.allAsBuiltJoints),
                      'rigid_groups': sum(1 for _ in root.allRigidGroups)}
    B0 = bodies(root)
    box0 = {k: box_of(b) for k, (b, o) in B0.items()}
    temp0 = {k: exact_copy(b, o, mgr)[1] for k, (b, o) in B0.items()}
    edited = {'%s|%s' % (o, b) for leg, pair in PAIRS.items() for o, b in pair}
    lift = [-DOWN[0] * LIFT, -DOWN[1] * LIFT, -DOWN[2] * LIFT]
    rear_keys = {k for k in B0 if k != 'root|' + PLATE and (box0[k][1] + box0[k][4]) / 2 > REAR_Y
                 and abs((box0[k][0] + box0[k][3]) / 2 - MID_X) < HALF_WIDTH}
    # 1. new teeth (shared components: done once, in the front placement)
    _res['parts'] = {}
    for spec in PARTS:
        r = edit_part(spec, root, mgr)
        if r is None:
            return
        _res['parts'][spec['kind']] = r
    adsk.doEvents()
    # 3. rear pinion phase: the gear and pinion components sit turned differently at the rear; turn the rear pinion pair
    #    (both rear pinions share one axis line) so the rear meshes like the front does
    gF, gR = by_path(root, PARTS[0]['occ']), by_path(root, PARTS[0]['rear'])
    pF, pR = by_path(root, PARTS[1]['occ']), by_path(root, PARTS[1]['rear'])
    dG, offG = x_turn(gF.transform2, gR.transform2)
    dP, offP = x_turn(pF.transform2, pR.transform2)
    if not check('rear gear / pinion placements differ from the front only by a turn about x', offG < 1e-6 and offP < 1e-6, [offG, offP]):
        return
    thF, thR = line_angle('FL'), line_angle('RL')
    delta = wrap(thF + dG - thR, 2 * math.pi / 24)                  # rear gear space vs its meshing position
    turn = wrap((thR + math.pi - 2 * delta) - (thF + math.pi + dP), 2 * math.pi / 12)
    _res['rear_phase'] = {'gear_turn_front_to_rear_deg': round(math.degrees(dG), 3), 'pinion_turn_front_to_rear_deg': round(math.degrees(dP), 3),
                          'line_front_deg': round(math.degrees(thF), 3), 'line_rear_deg': round(math.degrees(thR), 3),
                          'gear_offset_deg': round(math.degrees(delta), 3), 'pinion_turn_deg': round(math.degrees(turn), 3)}
    # 2. rear module: top-level occurrences + root bodies behind REAR_Y (all of them, hidden ones too)
    T = adsk.core.Matrix3D.create()
    T.translation = V3(-DOWN[0] * LIFT, -DOWN[1] * LIFT, -DOWN[2] * LIFT)
    occs = [root.occurrences.item(i) for i in range(root.occurrences.count)]
    cy = lambda bb: (bb.minPoint.y + bb.maxPoint.y) / 2
    valid = lambda bb: bb is not None and bb.minPoint.x <= bb.maxPoint.x
    rear = lambda bb: cy(bb) > REAR_Y and abs((bb.minPoint.x + bb.maxPoint.x) / 2 - MID_X) < HALF_WIDTH
    move_occ = [o for o in occs if valid(o.boundingBox) and rear(o.boundingBox)]
    move_root = [b for b in root.bRepBodies if b.name != PLATE and rear(b.boundingBox)]
    _res['moved'] = {'occurrences': [o.name for o in move_occ], 'root_bodies': [b.name for b in move_root]}
    if not check('moved occurrences lie wholly behind REAR_Y', all(o.boundingBox.minPoint.y > REAR_Y for o in move_occ),
                 [[o.name, round(o.boundingBox.minPoint.y, 3)] for o in move_occ if o.boundingBox.minPoint.y <= REAR_Y]):
        return
    if not check('rear pinion pair is a top-level occurrence', any(o.name == PARTS[1]['rear'] for o in move_occ)):
        return
    pin_axis_c = moved(P3(*PIV['RL']['P']), inverse(pR.transform2))
    gear_origin0 = gR.transform2.translation
    phase = adsk.core.Matrix3D.create()
    phase.setToRotation(turn, V3(1, 0, 0), P3(*PIV['RL']['P']))
    errs = []
    for o in (move_occ if LIFT else [o for o in move_occ if o.name == PARTS[1]['rear']]):
        try:
            m = o.transform2.copy()
            if o.name == PARTS[1]['rear']:
                m.transformBy(phase)                              # pre-multiply: turn about the pinion axis first
            m.transformBy(T)                                      # then lift
            g = o.isGrounded
            if g:
                o.isGrounded = False
            o.transform2 = m
            if g:
                o.isGrounded = True
        except Exception as e:
            errs.append([o.name, str(e)])
    if not check('rear occurrences moved', not errs, errs):
        return
    if design.snapshots.hasPendingSnapshot:
        design.snapshots.add()
        _res['snapshot'] = True
    coll = adsk.core.ObjectCollection.create()
    for b in move_root:
        coll.add(b)
    if coll.count and LIFT:
        mi_ = root.features.moveFeatures.createInput2(coll)
        mi_.defineAsFreeMove(T)
        mf = root.features.moveFeatures.add(mi_)
        mf.name = 'Rear module up %.0f mm (rear bodies)' % (LIFT * 10)
    adsk.doEvents()
    # bodies modelled in the plate's context (the rear pelvis, a hip bracket) stay put when their occurrence moves:
    # move those inside their own component by the same lift
    stuck, partial = {}, []
    for k, (b, o) in bodies(root).items():
        if k not in rear_keys or k in edited:
            continue
        got = [box_of(b)[i] - box0[k][i] for i in range(6)]
        if max(abs(got[i] - lift[i % 3]) for i in range(6)) < 1e-3:
            continue
        if o is not None and max(abs(x) for x in got) < 1e-3:
            stuck.setdefault(o.fullPathName, (o, []))[1].append(b.nativeObject)
        else:
            partial.append([k, [round(x, 4) for x in got[:3]]])
    if not check('no rear body moved only part of the way', not partial, partial[:10]):
        return
    _res['moved']['in_context'] = {}
    for path, (o, natives) in stuck.items():
        comp = o.component
        if not check('%s: its component is used once (safe to move its bodies)' % path,
                     root.allOccurrencesByComponent(comp).count == 1, root.allOccurrencesByComponent(comp).count):
            return
        coll = adsk.core.ObjectCollection.create()
        for nb in natives:
            coll.add(nb)
        mi_ = comp.features.moveFeatures.createInput2(coll)
        mi_.defineAsFreeMove(T)   # in context, Fusion applies the move in world coordinates (W^-1 T W landed rotated by W)
        f = comp.features.moveFeatures.add(mi_)
        f.name = 'Rear module up %.0f mm (body modelled in context)' % (LIFT * 10)
        _res['moved']['in_context'][path] = [nb.name for nb in natives]
    adsk.doEvents()
    centre = None
    if SPACER_ON:
        # spacer: block between the lifted pelvis flange and the plate tab, two M3 clearance holes on the screw axes
        d = V3(*DOWN)
        w = d.crossProduct(V3(1, 0, 0))
        sp = SPACER
        mid = P3((SCREWS[0][0] + SCREWS[1][0]) / 2, SCREWS[0][1], SCREWS[0][2])
        sm, wm = (sp['s0'] + sp['s1']) / 2, (sp['w0'] + sp['w1']) / 2
        centre = moved(mid, adsk.core.Matrix3D.create())
        centre.translateBy(V3(d.x * sm + w.x * wm, d.y * sm + w.y * wm, d.z * sm + w.z * wm))
        block = mgr.createBox(adsk.core.OrientedBoundingBox3D.create(centre, V3(1, 0, 0), w, 2 * sp['half_u'], sp['w1'] - sp['w0'], sp['s1'] - sp['s0']))
        for s0 in SCREWS:
            p, q = P3(*s0), P3(*s0)
            p.translateBy(V3(d.x * (sp['s0'] - 0.1), d.y * (sp['s0'] - 0.1), d.z * (sp['s0'] - 0.1)))
            q.translateBy(V3(d.x * (sp['s1'] + 0.1), d.y * (sp['s1'] + 0.1), d.z * (sp['s1'] + 0.1)))
            mgr.booleanOperation(block, mgr.createCylinderOrCone(p, sp['hole_r'], q, sp['hole_r']), adsk.fusion.BooleanTypes.DifferenceBooleanType)
        so = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        sc = so.component
        sc.name = 'Rear Hip Spacer'
        bf = sc.features.baseFeatures.add()
        bf.startEdit()
        sc.bRepBodies.add(block, bf)
        bf.finishEdit()
        bf.name = 'Spacer %.0f mm (2x M3 clearance)' % (LIFT * 10)
        sc.bRepBodies.item(0).name = 'Rear Hip Spacer %.1f mm' % ((sp['s1'] - sp['s0']) * 10)
        _res['spacer'] = {'volume_cm3': round(sc.bRepBodies.item(0).volume, 4), 'size_cm': [2 * sp['half_u'], round(sp['w1'] - sp['w0'], 3), round(sp['s1'] - sp['s0'], 3)]}
        adsk.doEvents()
    lift_v = [-DOWN[0] * LIFT, -DOWN[1] * LIFT, -DOWN[2] * LIFT]
    if BUSHING:
        if not bushing_edits(root, mgr, lift_v):
            return
    if M4_HOLES:
        if not casing_m4(root, mgr):
            return
    adsk.doEvents()
    # checks
    B1 = bodies(root)
    bad, missing = [], sorted(set(B0) - set(B1))
    for k, (b, o) in B1.items():
        if k in edited or k not in box0:
            continue
        want = lift if k in rear_keys else [0, 0, 0]
        got = [box_of(b)[i] - box0[k][i] for i in range(6)]
        if max(abs(got[i] - want[i % 3]) for i in range(6)) > 1e-3:
            bad.append([k, [round(x, 4) for x in got[:3]], 'rear' if k in rear_keys else 'front'])
    new_keys = sorted(set(B1) - set(B0))
    check('rear bodies moved by exactly the lift, the rest stayed', not bad and not missing,
          {'wrong': bad[:20], 'n_wrong': len(bad), 'missing': missing[:10], 'n_rear': len(rear_keys), 'new': new_keys})
    gO1 = by_path(root, PARTS[0]['rear']).transform2.translation
    pin_axis_w = moved(pin_axis_c, by_path(root, PARTS[1]['rear']).transform2)
    want_g = [gear_origin0.x + lift[0], gear_origin0.y + lift[1], gear_origin0.z + lift[2]]
    want_p = [PIV['RL']['P'][i] + lift[i] for i in range(3)]
    axis_err = max([abs(a - b) for a, b in zip([gO1.x, gO1.y, gO1.z], want_g)] + [abs(a - b) for a, b in zip([pin_axis_w.x, pin_axis_w.y, pin_axis_w.z], want_p)])
    check('rear gear and pinion axes moved by exactly the lift', axis_err < 1e-4, round(axis_err, 6))
    T1 = {k: exact_copy(b, o, mgr)[1] for k, (b, o) in B1.items()}
    mesh = {}
    for leg, (g, p) in PAIRS.items():
        kg, kp = '%s|%s' % g, '%s|%s' % p
        mesh[leg] = round(common_volume(T1[kg], T1[kp], mgr) * 1000, 4)   # mm^3
    _res['tooth_overlap_mm3'] = mesh
    check('teeth mesh without overlap on all 4 legs', all(v < 0.01 for v in mesh.values()), mesh)
    # new interference: moved (and new) bodies against the rest, compared with the same pairs before the edit
    new_int, n_pairs = [], 0
    movers = [k for k in B1 if k in rear_keys or k in new_keys]
    others = [k for k in B1 if k not in rear_keys and k not in new_keys]
    bx1 = {k: box_of(B1[k][0]) for k in B1}
    pairs = [(a, b) for a in movers for b in others] + [(a, b) for a in new_keys for b in rear_keys if b in B1]   # spacer vs rear too
    for a, b in pairs:
        if not overlap(bx1[a], bx1[b], 0.01):
            continue
        n_pairs += 1
        v1 = common_volume(T1[a], T1[b], mgr)
        if v1 < NEW_INTERFERENCE:
            continue
        v0 = common_volume(temp0[a], temp0[b], mgr) if a in temp0 and b in temp0 else 0.0
        if v1 - v0 > NEW_INTERFERENCE:
            new_int.append([a, b, round(v1 * 1000, 2), round(v0 * 1000, 2)])
    check('no new interference from the lifted rear module or the spacer (mm^3: after, before)', not new_int,
          {'pairs_tested': n_pairs, 'new': new_int[:30]})
    if SPACER_ON:
        spk = next(k for k in new_keys if k.startswith('Rear Hip Spacer'))
        mm = app.measureManager
        seat = {k: round(mm.measureMinimumDistance(B1[spk][0], B1[k][0]).value * 10, 3) if k in B1 else 'missing'
                for k in ('Pevis:1|Body1', 'root|' + PLATE)}
        check('spacer seated on the pelvis flange and the plate tab (gap mm)', all(v != 'missing' and v < 0.05 for v in seat.values()), seat)
    sick1 = sick()
    check('no new timeline errors / warnings', set(sick1) <= set(sick0), {'before': sick0, 'after': sick1})
    # renders (views only; nothing is hidden)
    os.makedirs(SHOTS, exist_ok=True)
    H, P = PIV['FL']['H'], PIV['FL']['P']
    Hr, Pr = [PIV['RL']['H'][i] + lift[i] for i in range(3)], [PIV['RL']['P'][i] + lift[i] for i in range(3)]
    _res['renders'] = {
        'v7_left.png': render(app, 'v7_left.png', (MID_X, -8.6, -3.0), (MID_X + 60, -8.6, -3.0), 32.0),
        'v7_iso.png': render(app, 'v7_iso.png', (MID_X, -8.0, -3.0), (MID_X + 40, 30, 28), 38.0),
        'v7_FL_gears.png': render(app, 'v7_FL_gears.png', (3.1, (H[1] + P[1]) / 2, (H[2] + P[2]) / 2), (23.1, (H[1] + P[1]) / 2, (H[2] + P[2]) / 2), 6.0),
        'v7_RL_gears.png': render(app, 'v7_RL_gears.png', (3.1, (Hr[1] + Pr[1]) / 2, (Hr[2] + Pr[2]) / 2), (23.1, (Hr[1] + Pr[1]) / 2, (Hr[2] + Pr[2]) / 2), 6.0),
        'v7_rear_side.png': render(app, 'v7_rear_side.png', (MID_X, -1.0, -2.0), (MID_X + 60, -1.0, -2.0), 20.0)}
    if SPACER_ON:
        c_sp = [centre.x, centre.y, centre.z]
        hide = [o for o in occs if o.name in ('Hind Leg:1', 'Hind Leg(Mirror):1', 'XL4015 StepDown DC-DC 5A (CC-CV) v1:1',
                                              '12Channel PWM v2:1', 'DS-843MG v1(Mirror) (1):2', 'm(Mirror):2')]
        was = [o.isLightBulbOn for o in hide]
        for o in hide:
            o.isLightBulbOn = False
        _res['renders']['v7_spacer.png'] = render(app, 'v7_spacer.png', tuple(c_sp), (c_sp[0] + 20, c_sp[1] + 14, c_sp[2] + 8), 10.0)
        _res['renders']['v7_spacer_side.png'] = render(app, 'v7_spacer_side.png', tuple(c_sp), (c_sp[0] + 40, c_sp[1], c_sp[2]), 10.0)
        for o, on in zip(hide, was):
            o.isLightBulbOn = on
        check('visibility restored after the spacer views', [o.isLightBulbOn for o in hide] == was)


def _finish():
    doc = _res.pop('doc', None)
    ok = _res['checks'] and all(v['ok'] for v in _res['checks'].values()) and not _res.get('status', '').startswith(('ABORT', 'retry'))
    if doc and ok and not DRY_RUN:
        app = adsk.core.Application.get()
        folder = app.data.findFileById(COPY_ID).parentFolder
        design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
        if BUSHING:                  # renamed last too: body names are part of the check keys
            bcomp = design.allComponents.itemByName(PARTS[0]['comp'])
            for old, side in (('Body2', 'L'), ('Body5', 'R')):
                bcomp.bRepBodies.itemByName(old).name = ('printed bushing %s' if BUSHING == 'printed' else '623ZZ bearing %s') % side
        for spec in PARTS:           # renamed last: occurrence paths (the check keys) carry the component name
            if spec['rename']:
                design.allComponents.itemByName(spec['comp']).name = spec['rename']
        what = ['2:1 gear-servo drive (12T:24T m1.0)', 'rear hips +%.0f mm on a spacer' % (LIFT * 10) if SPACER_ON else 'rear not raised (no spacer)']
        what += ['gear on a %s' % ('printed bushing (10.4 mm running fit)' if BUSHING == 'printed' else '623ZZ (10.0 mm press fit)')] if BUSHING else []
        what += ['M4 casing holes'] if M4_HOLES else []
        _res['saved'] = doc.saveAs(NEW_NAME, folder, 'Fox v7: %s; from %s v%s' % (', '.join(what), SOURCE_NAME, _res.get('source_version')), '')
        _res['status'] = 'ok'
    elif doc:
        _res['status'] = 'ok (dry run, discarded)' if ok else 'FAILED CHECKS: discarded, nothing saved'
        doc.close(False)
    elif not _res.get('status'):
        _res['status'] = 'ABORT: a check failed before the document was opened'


def _write():  # result file appears only once, with a final status
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, RESULT), 'w') as f:
        json.dump(_res, f, indent=1, default=str)


class _Run(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            _work()
        except Exception:
            _res['status'] = 'ABORT: exception'
            _res['traceback'] = traceback.format_exc()
        try:
            _finish()
        except Exception:
            _res['finish_traceback'] = traceback.format_exc()
        if not _res.get('status', '').startswith('retry'):
            _write()
        _done.set()


def _waiter():
    app = adsk.core.Application.get()
    for _ in range(300):
        try:
            if app.isStartupComplete:
                break
        except Exception:
            pass
        time.sleep(1)
    time.sleep(5)
    for _ in range(25):
        _done.clear()
        _res['checks'] = {}
        _res.pop('status', None)
        app.fireCustomEvent(EVT)
        _done.wait(3000)
        if not _res.get('status', '').startswith('retry'):
            break
        time.sleep(5)
    if _res.get('status', '').startswith('retry'):
        _write()


def run(context):
    app = adsk.core.Application.get()
    try:
        app.unregisterCustomEvent(EVT)
    except Exception:
        pass
    h = _Run()
    app.registerCustomEvent(EVT).add(h)
    _handlers.append(h)
    threading.Thread(target=_waiter, daemon=True).start()


def stop(context):
    try:
        adsk.core.Application.get().unregisterCustomEvent(EVT)
    except Exception:
        pass
