# One-shot Fusion add-in. Copy this folder into
#   <prefix>/drive_c/users/<you>/AppData/Roaming/Autodesk/Autodesk Fusion 360/API/AddIns/
# start Fusion, wait for C:\fusion_jobs\v7_free_gear.json, then delete the folder again (it runs at every start).
#
# Makes each crank gear of 'Fox_Prototype_03_v7_RL' turn only with its pinion (saved as a new version of that design).
# The gear rides on a 623 ball bearing (3 x 10 x 4 mm: the 'disk' in the CAD) whose inner ring the horn screw and washer
# clamp to the femur hub. Two things let the femur drag the gear:
#   * the femur hub's end face is 6 mm across, wider than the bearing's inner ring (about 5.2 mm on a real 623-2Z, 4.1 in
#     the model), so tightening the screw also presses the shield / outer ring and the bearing locks. Fix: the hub keeps a
#     4.4 mm boss and the ring around it is relieved 0.3 mm (all 4 femurs);
#   * the gear's bearing seat is 10.2 mm, loose on the 10 mm outer ring, so the gear runs on the ring rather than with it.
#     Fix: a 10.0 mm press fit (shared component: all 4 gears).
# The bearing bodies are renamed '623ZZ bearing'. Checks: each edit changes its body by the expected volume, single
# solids, the gear seat now touches the bearing without overlap, timeline healthy. Exports the new gear and the femurs
# as STL (mm) to C:\fusion_jobs\v7_free_gear\.
import adsk.core, adsk.fusion, json, math, os, threading, time, traceback

DRY_RUN = False
COPY_ID, DOC_NAME = 'urn:adsk.wipprod:dm.lineage:sWQJByHaTNKGDxKGjmsT6g', 'Fox_Prototype_03_v7_RL'   # folder of the v6 copy
HERE = os.path.dirname(os.path.abspath(__file__))
PIV = json.load(open(os.path.join(HERE, 'link_map.json')))['pivots']   # v7 pins (design frame, cm)
OUT = r'C:\fusion_jobs'
STL = os.path.join(OUT, 'v7_free_gear')
MID_X = -2.476478
GEAR_COMP, GEAR_OCC = 'Spur Gear (24 teeth)', 'Spur Gear (24 teeth):2'          # shared; placed at the front here
GEARS = {'Body1': 'FL', 'Body4': 'FR'}
BEARINGS = {'Body2': '623ZZ bearing L', 'Body5': '623ZZ bearing R'}
FEMURS = {'FL': 'Front Leg:1+Femur(Mirror) (1):1', 'FR': 'Front Leg(Mirror):1+Femur(Mirror) (1)(Mirror):1',
          'RL': 'Hind Leg:1+Femur:1', 'RR': 'Hind Leg(Mirror):1+Femur(Mirror) (2):1'}
HUB_R, BOSS_R, RELIEF = 0.30, 0.22, 0.03          # cm: femur hub end radius, the boss left on it, relief depth
SEAT = (0.10, 0.54, 0.500, 0.510)                 # gear bearing seat: cm from the gear's inboard end, new / old radius
EVT = 'FoxV7FreeGearRun'
_handlers, _res, _done = [], {'checks': {}}, threading.Event()


def check(name, ok, detail=None):
    _res['checks'][name] = {'ok': bool(ok), 'detail': detail}
    return ok


def P3(x, y, z):
    return adsk.core.Point3D.create(x, y, z)


def by_path(root, path):
    for i in range(root.allOccurrences.count):
        o = root.allOccurrences.item(i)
        if o.fullPathName == path:
            return o
    return None


def inverse(m):
    m = m.copy()
    m.invert()
    return m


def ring(mgr, x0, x1, y, z, r_in, r_out):     # world ring along x through (y, z), from x0 to x1
    outer = mgr.createCylinderOrCone(P3(x0, y, z), r_out, P3(x1, y, z), r_out)
    mgr.booleanOperation(outer, mgr.createCylinderOrCone(P3(x0, y, z), r_in, P3(x1, y, z), r_in),
                         adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return outer


def combine(comp, target, tool_world, occ, op, name, mgr):
    """Add the world-space tool body to the component (as a base feature) and combine it with the target body."""
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
    cf = comp.features.combineFeatures.add(ci)
    cf.name = name
    return cf


def common_volume(a, b, mgr):
    t = mgr.copy(a)
    try:
        mgr.booleanOperation(t, b, adsk.fusion.BooleanTypes.IntersectionBooleanType)
        return t.volume
    except Exception:
        return 0.0


def world_copy(body, occ, mgr):
    t = mgr.copy(body.nativeObject or body)
    mgr.transform(t, occ.transform2)
    return t


def export_stl(design, body, path):
    em = design.exportManager
    o = em.createSTLExportOptions(body, path)
    o.isBinaryFormat = True
    o.unitType = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
    o.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
    return em.execute(o)


def _work():
    app = adsk.core.Application.get()
    try:
        folder = app.data.findFileById(COPY_ID).parentFolder
    except Exception:
        folder = None
    files = folder.dataFiles if folder else None
    df = next((files.item(i) for i in range(files.count) if files.item(i).name == DOC_NAME and files.item(i).fileExtension == 'f3d'), None) if files else None
    if df is None:
        _res['status'] = 'retry: %s not reachable yet' % DOC_NAME
        return
    doc = app.documents.open(df, True)
    _res['doc'], _res['version_before'] = doc, doc.dataFile.versionNumber
    design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
    root, tl, mgr = design.rootComponent, design.timeline, adsk.fusion.TemporaryBRepManager.get()
    if not check('timeline marker at the end', tl.markerPosition == tl.count, [tl.markerPosition, tl.count]):
        return
    sick = lambda: sorted(tl.item(i).name for i in range(tl.count) if int(tl.item(i).healthState) in (1, 2))
    sick0 = sick()
    gocc = by_path(root, GEAR_OCC)
    gcomp = gocc.component
    edits = {}
    # 1. femur hubs: keep a boss that only meets the bearing's inner ring
    for leg, path in FEMURS.items():
        occ = by_path(root, path)
        if not check('%s femur found, its component used once' % leg, occ is not None and
                     root.allOccurrencesByComponent(occ.component).count == 1, path):
            return
        comp, H = occ.component, PIV[leg]['H']
        s = 1.0 if H[0] > MID_X else -1.0                                # outboard along x
        body = occ.bRepBodies.itemByName('Body1')
        ends = [f.pointOnFace.x for f in body.faces if f.geometry.objectType == adsk.core.Plane.classType()
                and abs(abs(f.geometry.normal.x) - 1) < 1e-6 and math.hypot(f.pointOnFace.y - H[1], f.pointOnFace.z - H[2]) < HUB_R + 0.005]
        if not check('%s femur: hub end face found' % leg, ends, ends):
            return
        x_end = max(ends) if s > 0 else min(ends)
        native = body.nativeObject
        v0 = native.volume
        combine(comp, native, ring(mgr, x_end - s * RELIEF, x_end + s * 0.05, H[1], H[2], BOSS_R, HUB_R + 0.02), occ,
                adsk.fusion.FeatureOperations.CutFeatureOperation, 'Hub relief: bearing inner ring only', mgr)
        native = comp.bRepBodies.itemByName('Body1')
        want = math.pi * (HUB_R ** 2 - BOSS_R ** 2) * RELIEF
        edits[leg + ' femur'] = [round(v0 - native.volume, 5), round(want, 5)]
        check('%s femur: relief removed the ring around the boss, one solid' % leg,
              abs((v0 - native.volume) - want) < 0.0003 and native.lumps.count == 1, edits[leg + ' femur'])
    # 2. gear bearing seat 10.2 -> 10.0 mm (shared component: every leg)
    for bname, leg in GEARS.items():
        native = gcomp.bRepBodies.itemByName(bname)
        H = PIV[leg]['H']
        s = 1.0 if H[0] > MID_X else -1.0
        bb = by_path(root, GEAR_OCC).bRepBodies.itemByName(bname).boundingBox
        x_in = bb.minPoint.x if s > 0 else bb.maxPoint.x                   # inboard end (crank plate side)
        v0 = native.volume
        f0, f1, r_new, r_old = SEAT
        combine(gcomp, native, ring(mgr, x_in + s * f0, x_in + s * f1, H[1], H[2], r_new, r_old + 0.02), gocc,
                adsk.fusion.FeatureOperations.JoinFeatureOperation, 'Bearing seat 10.0 mm press fit', mgr)
        native = gcomp.bRepBodies.itemByName(bname)
        want = math.pi * (r_old ** 2 - r_new ** 2) * (f1 - f0)
        edits[leg + ' gear'] = [round(native.volume - v0, 5), round(want, 5)]
        check('%s gear: seat filled to 10.0 mm, one solid' % leg,
              abs((native.volume - v0) - want) < 0.0005 and native.lumps.count == 1, edits[leg + ' gear'])
    _res['volume_change_cm3 [got, want]'] = edits
    for old, new in BEARINGS.items():
        gcomp.bRepBodies.itemByName(old).name = new
    adsk.doEvents()
    # checks: every gear now sits on its bearing's outer ring without overlap; no gear touches its femur
    seat, gap = {}, {}
    pairs = {'FL': ('Spur Gear (24 teeth):2', 'Body1', BEARINGS['Body2']), 'FR': ('Spur Gear (24 teeth):2', 'Body4', BEARINGS['Body5']),
             'RL': ('Hind Leg:1+Spur Gear (24 teeth):1', 'Body1', BEARINGS['Body2']), 'RR': ('Hind Leg:1+Spur Gear (24 teeth):1', 'Body4', BEARINGS['Body5'])}
    mm = app.measureManager
    for leg, (path, g, b) in pairs.items():
        o = by_path(root, path)
        gb, bb_ = o.bRepBodies.itemByName(g), o.bRepBodies.itemByName(b)
        seat[leg] = [round(common_volume(world_copy(gb, o, mgr), world_copy(bb_, o, mgr), mgr) * 1000, 4),
                     round(mm.measureMinimumDistance(gb, bb_).value * 10, 4)]
        f = by_path(root, FEMURS[leg]).bRepBodies.itemByName('Body1')
        gap[leg] = round(mm.measureMinimumDistance(gb, f).value * 10, 3)
    _res['gear_on_bearing [overlap mm3, distance mm]'] = seat
    _res['gear_to_femur_mm'] = gap
    check('every gear seated on its bearing (touching, no overlap)', all(v[0] < 0.05 and v[1] < 0.001 for v in seat.values()), seat)
    check('no gear touches its femur (>= 0.5 mm)', all(v >= 0.5 for v in gap.values()), gap)
    check('no new timeline errors / warnings', set(sick()) <= set(sick0), {'before': sick0, 'after': sick()})
    os.makedirs(STL, exist_ok=True)
    _res['stl'] = {'gear.stl': export_stl(design, by_path(root, GEAR_OCC).bRepBodies.itemByName('Body1'), os.path.join(STL, 'gear.stl'))}
    for leg, path in FEMURS.items():
        _res['stl']['femur_%s.stl' % leg] = export_stl(design, by_path(root, path).bRepBodies.itemByName('Body1'), os.path.join(STL, 'femur_%s.stl' % leg))


def _finish():
    doc = _res.pop('doc', None)
    ok = _res['checks'] and all(v['ok'] for v in _res['checks'].values()) and not _res.get('status', '').startswith(('ABORT', 'retry'))
    if doc and ok and not DRY_RUN:
        _res['saved'] = doc.save('v7.1: each crank gear turns only with its pinion (623ZZ seat 10.0 mm press fit, femur hubs '
                                 'relieved to a 4.4 mm boss on the inner ring)')
        _res['status'] = 'ok'
    elif doc:
        _res['status'] = 'ok (dry run, discarded)' if ok else 'FAILED CHECKS: discarded, nothing saved'
        doc.close(False)
    elif not _res.get('status'):
        _res['status'] = 'ABORT: a check failed before the document was opened'


def _write():  # result file appears only once, with a final status
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'v7_free_gear.json'), 'w') as f:
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
        _done.wait(1800)
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
