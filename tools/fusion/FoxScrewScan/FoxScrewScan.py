# One-shot, READ-ONLY Fusion add-in (never saves). Copy this folder into
#   <prefix>/drive_c/users/<you>/AppData/Roaming/Autodesk/Autodesk Fusion 360/API/AddIns/
# start Fusion, wait for C:\fusion_jobs\screw_scan.json, then delete the folder again (it runs at every start).
# Scans every visible solid body of the design copy for small cylindrical faces (screw holes, pins, splines) after
# suppressing, in memory, the Combine features that join the rear hip brackets to the pelvis (same list as
# FoxBuildLinkModel). Feed the result to tools/screw_graph.py. Another design: put scan.json {"source": name} here.
import adsk.core, adsk.fusion, json, os, threading, time, traceback

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scan.json')   # optional: {"source": design name}
SOURCE_NAME = json.load(open(CFG))['source'] if os.path.exists(CFG) else 'Fox_Prototype_03_v6_RL'
OUT, RMAX = r'C:\fusion_jobs', 0.30
UNJOIN = [('Combine1', 'Component41(Mirror) (1)'), ('Combine2', 'Component41(Mirror) (1)'), ('Combine2', 'Servo Pelv Upper')]
EVT, _handlers, _res, _done = 'FoxScrewScanRun', [], {}, threading.Event()


def find_file(app, name):
    for i in range(app.data.activeHub.dataProjects.count):
        files = app.data.activeHub.dataProjects.item(i).rootFolder.dataFiles
        for j in range(files.count):
            if files.item(j).name == name and files.item(j).fileExtension == 'f3d':
                return files.item(j)
    return None


def owner(t):
    try:
        return t.entity.parentComponent.name
    except Exception:
        return None


def cyl_faces(body, rmax):
    out, cyl = [], adsk.core.Cylinder.classType()
    for f in body.faces:
        g = f.geometry
        if g.objectType != cyl or g.radius > rmax:
            continue
        a = g.axis
        a.normalize()
        o = g.origin
        ts = [(v.geometry.x - o.x) * a.x + (v.geometry.y - o.y) * a.y + (v.geometry.z - o.z) * a.z
              for e in f.edges for v in (e.startVertex, e.endVertex) if v] or [0.0]
        out.append([round(o.x, 5), round(o.y, 5), round(o.z, 5), round(a.x, 6), round(a.y, 6), round(a.z, 6),
                    round(g.radius, 5), round(min(ts), 5), round(max(ts), 5), 0])
    return out


def _work():
    app = adsk.core.Application.get()
    df = find_file(app, SOURCE_NAME)
    if df is None:
        _res['status'] = 'retry'
        return
    doc = app.documents.open(df, True)
    _res['source_version'] = doc.dataFile.versionNumber
    design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
    root, tl = design.rootComponent, design.timeline
    hit = [i for i in range(tl.count) if (tl.item(i).name, owner(tl.item(i))) in UNJOIN]
    for i in sorted(hit, reverse=True):
        tl.item(i).isSuppressed = True
    adsk.doEvents()
    _res['suppressed'] = [[tl.item(i).name, owner(tl.item(i))] for i in hit]
    bodies = []

    def add(b, occ):
        if not b.isVisible or not b.isSolid or b.volume < 1e-6:
            return
        bb = b.boundingBox
        rec = {'id': len(bodies), 'src': (occ.fullPathName if occ else 'root') + '|' + b.name,
               'comp': occ.component.name if occ else '', 'vol': round(b.volume, 5), 'faces': b.faces.count,
               'bbmin': [round(bb.minPoint.x, 4), round(bb.minPoint.y, 4), round(bb.minPoint.z, 4)],
               'bbmax': [round(bb.maxPoint.x, 4), round(bb.maxPoint.y, 4), round(bb.maxPoint.z, 4)]}
        rec['holes'] = cyl_faces(b, RMAX) if b.faces.count <= 4000 else []
        if occ and 'DS-843' in occ.component.name and b.volume < 0.03 and b.faces.count >= 50:
            rec['spline_cyls'] = cyl_faces(b, 1.0)            # servo output spline
        bodies.append(rec)
    for b in root.bRepBodies:
        add(b, None)
    for i in range(root.allOccurrences.count):
        o = root.allOccurrences.item(i)
        for b in o.bRepBodies:
            add(b, o)
    _res.update({'bodies': bodies, 'status': 'ok' if len(hit) == len(UNJOIN) else 'ABORT: bracket joins not found',
                 'health': [[tl.item(i).name, int(tl.item(i).healthState)] for i in range(tl.count) if int(tl.item(i).healthState) in (1, 2)]})


def _write():  # result file appears only once, with a final status
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'screw_scan.json'), 'w') as f:
        json.dump(_res, f)


class _Run(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            _work()
        except Exception:
            _res['status'] = 'ABORT: exception'
            _res['traceback'] = traceback.format_exc()
        try:
            d = adsk.core.Application.get().activeDocument
            if d and d.name == SOURCE_NAME:
                d.close(False)                                 # discard the in-memory suppression
        except Exception:
            pass
        if not _res.get('status') == 'retry':
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
        _res.clear()
        app.fireCustomEvent(EVT)
        _done.wait(1800)
        if _res.get('status') != 'retry':
            break
        time.sleep(5)
    if _res.get('status') == 'retry':
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
