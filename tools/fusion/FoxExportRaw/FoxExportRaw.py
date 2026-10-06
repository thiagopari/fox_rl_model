# One-shot, READ-ONLY Fusion add-in. Copy this folder into
#   <prefix>/drive_c/users/<you>/AppData/Roaming/Autodesk/Autodesk Fusion 360/API/AddIns/
# start Fusion, wait for C:\fusion_jobs\export\raw.json, then delete the folder again (it runs at every start).
# From 'Fox_Prototype_03_v6_RL_Mechanism' it exports one binary STL per link (metres, model frame: x fwd, y left, z up),
# each link's mass / COM / inertia (about the world origin, kg cm^2) and every joint's origin / axes / limits;
# 'axis_measured' is the rotation axis recovered from the child's motion when the joint is turned by 0.1 rad.
import adsk.core, adsk.fusion, json, math, os, threading, time, traceback

OUT = r'C:\fusion_jobs\export'
DOC_NAME = 'Fox_Prototype_03_v6_RL_Mechanism'
EVT = 'FoxExportRawRun'
_handlers, _res, _done = [], {}, threading.Event()


def v3(p):
    return [p.x, p.y, p.z]


def find_doc_file(app):
    for i in range(app.data.activeHub.dataProjects.count):
        proj = app.data.activeHub.dataProjects.item(i)
        files = proj.rootFolder.dataFiles
        for j in range(files.count):
            if files.item(j).name == DOC_NAME:
                return files.item(j)
    return None


def _work():
    app = adsk.core.Application.get()
    df = find_doc_file(app)
    if df is None:
        _res['status'] = 'retry: document not reachable yet'
        return
    doc = app.documents.open(df, True)
    design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
    root = design.rootComponent
    for i in range(root.asBuiltJoints.count):          # export at the zero (standing) pose
        root.asBuiltJoints.item(i).jointMotion.rotationValue = 0.0
    adsk.doEvents()
    os.makedirs(os.path.join(OUT, 'meshes'), exist_ok=True)
    links, em = {}, design.exportManager
    for i in range(root.occurrences.count):
        occ = root.occurrences.item(i)
        comp = occ.component
        if comp.bRepBodies.count == 0:
            continue
        pp = comp.getPhysicalProperties(adsk.fusion.CalculationAccuracy.HighCalculationAccuracy)
        ok, xx, yy, zz, xy, yz, xz = pp.getXYZMomentsOfInertia()
        path = os.path.join(OUT, 'meshes', comp.name + '.stl')
        opts = em.createSTLExportOptions(occ, path)
        opts.isBinaryFormat = True
        opts.unitType = adsk.fusion.DistanceUnits.MeterDistanceUnits
        opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementMedium
        opts.isOneFilePerBody = False
        em.execute(opts)
        links[comp.name] = {'mass_kg': pp.mass, 'com_cm': v3(pp.centerOfMass), 'inertia_origin_kgcm2': [xx, yy, zz, xy, yz, xz],
                            'volume_cm3': pp.volume, 'bodies': comp.bRepBodies.count, 'stl': comp.name + '.stl',
                            'occurrence_identity': all(abs(a - b) < 1e-9 for a, b in zip(occ.transform2.asArray(), adsk.core.Matrix3D.create().asArray()))}
    joints = []
    for i in range(root.asBuiltJoints.count):
        j = root.asBuiltJoints.item(i)
        g, lim = j.geometry, j.jointMotion.rotationLimits
        try:                                               # loop pins may move other parts instead: then no axis
            j.jointMotion.rotationValue = 0.1
            adsk.doEvents()
            m = j.occurrenceOne.transform2.asArray()       # row-major; rotation part R, axis from its skew part
            j.jointMotion.rotationValue = 0.0
            adsk.doEvents()
        except Exception:
            m = adsk.core.Matrix3D.create().asArray()
        w = [m[9] - m[6], m[2] - m[8], m[4] - m[1]]
        n = math.sqrt(sum(c * c for c in w)) or None
        joints.append({'name': j.name, 'child': j.occurrenceOne.component.name, 'parent': j.occurrenceTwo.component.name,
                       'origin_cm': v3(g.origin), 'primary': v3(g.primaryAxisVector), 'secondary': v3(g.secondaryAxisVector),
                       'third': v3(g.thirdAxisVector), 'type': j.jointMotion.objectType.split('::')[-1],
                       'axis_measured': [c / n for c in w] if n and n > 1e-4 else None,
                       'measured_angle': math.asin(min(1.0, n / 2)) if n else 0.0,
                       'motion_links': [[ml.jointOne.name, ml.jointTwo.name, ml.isReversed] for ml in j.motionLinks],
                       'lower': lim.minimumValue if lim.isMinimumValueEnabled else None,
                       'upper': lim.maximumValue if lim.isMaximumValueEnabled else None})
    _res.update({'doc_version': doc.dataFile.versionNumber, 'links': links, 'joints': joints, 'status': 'ok',
                 'grounded': [root.occurrences.item(i).component.name for i in range(root.occurrences.count) if root.occurrences.item(i).isGrounded]})
    doc.close(False)  # joint values were only touched for the export; Fusion would otherwise save them on exit


def _write():  # result file appears only once, with a final status
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'raw.json'), 'w') as f:
        json.dump(_res, f, indent=1)


class _Run(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            _work()
        except Exception:
            _res['status'] = 'ABORT: exception'
            _res['traceback'] = traceback.format_exc()
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
    for _ in range(25):  # the data service can lag right after start-up
        _done.clear()
        _res.clear()
        app.fireCustomEvent(EVT)
        _done.wait(900)
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
