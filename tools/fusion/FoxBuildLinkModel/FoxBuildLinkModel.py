# One-shot Fusion add-in (read-only on the source design). Copy this folder into
#   <prefix>/drive_c/users/<you>/AppData/Roaming/Autodesk/Autodesk Fusion 360/API/AddIns/
# start Fusion, wait for C:\fusion_jobs\build_link_model.json, then delete the folder again (it runs at every start).
#
# Put link_map.json (written by tools/screw_graph.py from the FoxScrewScan hole scan) next to this file first.
# From the design copy 'Fox_Prototype_03_v6_RL' it builds a NEW document (NEW_NAME) with:
#   * one top-level component per RL link: base, {FL,FR,RL,RR}_{hip,thigh,calf}, made of the VISIBLE solid bodies
#     (copied exactly in place), re-oriented to x-forward / y-left / z-up with the origin between the hips;
#     which link a body joins comes from link_map.json: parts screwed together, or sitting on a servo spline, share a link;
#   * materials: PLA 1.24 g/cm^3, DS-843MG servos 8.5 g each, electronics at typical real masses;
#   * 12 revolute as-built joints (Unitree naming); joint frame = sketch circle on the pivot (Z = axis); the abduction
#     axes are the real (tilted) hip-servo spline axes from link_map.json;
#   * a verification that every joint rotates its child rigidly about the intended line.
# The source is only changed in memory (UNJOIN) and closed without saving.
# Pin axes (HIP, KNEE) and the sagittal plane (MID_X) are measured from the current CAD: update them if legs move.
import adsk.core, adsk.fusion, json, math, os, threading, time, traceback

DRY_RUN = False                       # True: build + verify, then discard instead of saving
SOURCE_NAME = 'Fox_Prototype_03_v6_RL'
NEW_NAME = 'Fox_Prototype_03_v6_RL_URDF_v2'
# Combine (join) features that fuse the rear hip brackets into the pelvis, which would stop them turning with their
# abduction servos: suppressed in memory only (rear-left: 'Servo Pelv Upper'; rear-right: 'Component41(Mirror) (1)').
UNJOIN = [('Combine1', 'Component41(Mirror) (1)'), ('Combine2', 'Component41(Mirror) (1)'), ('Combine2', 'Servo Pelv Upper')]
MAP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'link_map.json')))
OUT = r'C:\fusion_jobs'
MID_X, MID_Y = -2.476478, -8.6                           # sagittal plane x / front-rear split y (source frame, cm)
HIP = {'F': (-17.076, 1.716), 'R': (0.000, 0.760)}      # hip-pitch pin axes (y, z), X-parallel
KNEE = {'F': (-14.449, -5.309), 'R': (-3.807, -5.702)}  # knee pin axes (y, z)
Z0 = (HIP['F'][1] + HIP['R'][1]) / 2
LIMIT = {'hip': 0.5, 'thigh': 1.2, 'calf': 1.2}          # +- rad around the standing pose (placeholders)
PLA, SERVO_G = 1.24, 8.5
ELEC_G = {'Raspberry Pi 4 Model B': 46.0, '12Channel PWM v2': 9.0, 'XL4015 StepDown DC-DC 5A (CC-CV) v1': 16.0,
          'arduino nano': 7.0, 'Adafruit_BNO055_AP203': 3.0}
LEGS = ['FL', 'FR', 'RL', 'RR']
LINKS = ['base'] + ['%s_%s' % (l, p) for l in LEGS for p in ('hip', 'thigh', 'calf')]
EVT = 'FoxBuildLinkModelRun'
_handlers, _res, _done = [], {'checks': {}}, threading.Event()


def check(name, ok, detail=None):
    _res['checks'][name] = {'ok': bool(ok), 'detail': detail}
    return ok


def find_file(app, name):
    for i in range(app.data.activeHub.dataProjects.count):
        files = app.data.activeHub.dataProjects.item(i).rootFolder.dataFiles
        for j in range(files.count):
            if files.item(j).name == name:
                return files.item(j)
    return None


def frame():  # source -> RL: move the centre between the hips to the origin, then +90 deg about Z
    t = adsk.core.Matrix3D.create()
    t.translation = adsk.core.Vector3D.create(-MID_X, -MID_Y, -Z0)
    r = adsk.core.Matrix3D.create()
    r.setToRotation(math.pi / 2, adsk.core.Vector3D.create(0, 0, 1), adsk.core.Point3D.create(0, 0, 0))
    m = t.copy()
    m.transformBy(r)  # transformBy pre-multiplies: m = r * t
    return m


def owner(t):
    try:
        return t.entity.parentComponent.name
    except Exception:
        return None


def collect(src_root):  # every visible solid body, its link from link_map.json, and its material class
    recs, unmapped = [], []

    def add(body, occ):
        if not body.isVisible or not body.isSolid or body.volume < 1e-6:
            return
        src = (occ.fullPathName if occ else 'root') + '|' + body.name
        leaf = occ.component.name if occ else ''
        top = occ.fullPathName.split('+')[0].rsplit(':', 1)[0] if occ else ''
        link, vol = MAP['links'].get(src, (None, 0.0))
        if link is None or abs(vol - body.volume) > 1e-4 + 1e-4 * body.volume:
            unmapped.append([src, round(body.volume, 5), vol])
        mat, group = 'pla', None
        if occ and 'DS-843' in leaf:
            mat, group = 'servo', occ.fullPathName
        elif top in ELEC_G:
            mat, group = 'elec', occ.fullPathName.split('+')[0]
        recs.append({'body': body, 'occ': occ, 'link': link, 'mat': mat, 'group': group, 'leaf': leaf, 'src': src, 'vol': body.volume})

    for b in src_root.bRepBodies:
        add(b, None)
    for i in range(src_root.allOccurrences.count):
        o = src_root.allOccurrences.item(i)
        for b in o.bRepBodies:
            add(b, o)
    return recs, unmapped


def pivot_circle(root, name, p, v):
    """Hidden sketch circle centred on p whose normal is v (the joint Z axis). Pitch axes run along y; the abduction
    axes lie in the x-z plane, so a plane normal to them is made normal to a short sketch line along v through p."""
    pi = root.constructionPlanes.createInput()
    pi.setByOffset(root.xZConstructionPlane, adsk.core.ValueInput.createByReal(p.y))
    pl = root.constructionPlanes.add(pi)
    pl.name, pl.isLightBulbOn = name + ' plane', False
    if abs(v.y) < 0.99:
        sk0 = root.sketches.add(pl)
        sk0.name = name + ' axis line'
        a, b = p.copy(), p.copy()
        a.translateBy(adsk.core.Vector3D.create(-v.x, -v.y, -v.z))
        b.translateBy(v)
        ln = sk0.sketchCurves.sketchLines.addByTwoPoints(sk0.modelToSketchSpace(a), sk0.modelToSketchSpace(b))
        sk0.isVisible = False
        pi = root.constructionPlanes.createInput()
        pi.setByDistanceOnPath(ln, adsk.core.ValueInput.createByReal(0.5))
        pl = root.constructionPlanes.add(pi)
        pl.name, pl.isLightBulbOn = name + ' normal plane', False
    sk = root.sketches.add(pl)
    sk.name = name + ' axis'
    return sk, sk.sketchCurves.sketchCircles.addByCenterRadius(sk.modelToSketchSpace(p), 0.15)


def lib_material(app):
    for i in range(app.materialLibraries.count):
        lib = app.materialLibraries.item(i)
        for n in ('ABS Plastic', 'Plastic', 'Nylon 6/6', 'Acrylic'):
            try:
                m = lib.materials.itemByName(n)
                if m:
                    return m
            except Exception:
                pass
    return None


def set_density(mat, g_cm3, probe_body):  # set, then verify against a real body and correct the unit factor
    prop = mat.materialProperties.itemById('structural_Density')
    prop.value = g_cm3 * 1000.0
    probe_body.material = mat
    for _ in range(3):
        eff = probe_body.physicalProperties.mass * 1000.0 / probe_body.volume
        if abs(eff - g_cm3) / g_cm3 < 0.005:
            break
        prop.value = prop.value * g_cm3 / eff
    return probe_body.physicalProperties.mass * 1000.0 / probe_body.volume


def exact_copy(r, mgr):  # native copy + explicit world transform, or direct copy; keep whichever matches exactly
    def dev(src, temp):
        n = src.vertices.count
        if n == 0:
            a, b = src.boundingBox, temp.boundingBox
            return max(abs(a.minPoint.x - b.minPoint.x), abs(a.maxPoint.x - b.maxPoint.x), abs(a.minPoint.z - b.minPoint.z)) * 0.01
        sv = [src.vertices.item(i).geometry for i in sorted({0, n // 2, n - 1})]
        tv = [temp.vertices.item(i).geometry for i in range(temp.vertices.count)]
        return max(min(p.distanceTo(q) for q in tv) for p in sv) if tv else 99.0
    cands = []
    if r['occ'] is not None:
        t = mgr.copy(r['body'].nativeObject or r['body'])
        mgr.transform(t, r['occ'].transform2)
        cands.append(t)
    cands.append(mgr.copy(r['body']))
    return min(((dev(r['body'], t), t) for t in cands), key=lambda x: x[0])


def _work():
    app = adsk.core.Application.get()
    df = find_file(app, SOURCE_NAME)
    if df is None:
        _res['status'] = 'retry: source not reachable yet'
        return
    src_doc = app.documents.open(df, True)
    _res['src_doc'] = src_doc
    folder = src_doc.dataFile.parentFolder
    name = NEW_NAME
    if any(folder.dataFiles.item(i).name == name for i in range(folder.dataFiles.count)):
        name = NEW_NAME + time.strftime(' %Y-%m-%d %H%M')
    _res['new_name'], _res['source_version'] = name, src_doc.dataFile.versionNumber
    src = adsk.fusion.Design.cast(src_doc.products.itemByProductType('DesignProductType'))
    tl = src.timeline
    hit = [i for i in range(tl.count) if (tl.item(i).name, owner(tl.item(i))) in UNJOIN]
    for i in sorted(hit, reverse=True):
        tl.item(i).isSuppressed = True
    adsk.doEvents()
    errs = [tl.item(i).name for i in range(tl.count) if tl.item(i).healthState == adsk.fusion.FeatureHealthStates.ErrorFeatureHealthState]
    if not check('rear bracket joins suppressed (in memory) without errors', len(hit) == len(UNJOIN) and not errs, {'timeline': hit, 'errors': errs}):
        return
    recs, unmapped = collect(src.rootComponent)
    _res['n_bodies'] = len(recs)
    if not check('every body has a link in link_map.json (else rerun FoxScrewScan + tools/screw_graph.py)',
                 not unmapped and len(recs) == len(MAP['links']), {'unmapped': unmapped[:10], 'bodies': len(recs), 'map': len(MAP['links'])}):
        return
    if not check('4 abduction axes in link_map.json', sorted(MAP['abduction_axes']) == sorted(LEGS), sorted(MAP['abduction_axes'])):
        return
    hind_tibia_app = next((r['body'].appearance for r in recs if r['src'].startswith('Hind Leg:1+Tibia:1+Component7:1|')), None)
    M, mgr, bad = frame(), adsk.fusion.TemporaryBRepManager.get(), []
    for r in recs:
        d, t = exact_copy(r, mgr)
        if d > 1e-4:
            bad.append([r['src'], round(d, 5)])
        mgr.transform(t, M)
        r['temp'] = t
    if not check('copies placed exactly', not bad, bad[:10]):
        return
    new_doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    _res['new_doc'], _res['folder'] = new_doc, folder
    nd = adsk.fusion.Design.cast(new_doc.products.itemByProductType('DesignProductType'))
    nd.designType = adsk.fusion.DesignTypes.ParametricDesignType
    nroot, occs, apps = nd.rootComponent, {}, {}
    for link in LINKS:
        rl = [r for r in recs if r['link'] == link]
        occ = nroot.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        occ.component.name = link
        occs[link] = occ
        if not rl:
            continue
        comp = occ.component
        bf = comp.features.baseFeatures.add()
        bf.startEdit()
        for r in rl:
            comp.bRepBodies.add(r['temp'], bf)
        bf.finishEdit()
        bf.name = link + ' bodies'
        for i, r in enumerate(rl):
            b = comp.bRepBodies.item(i)
            b.name = ('%03d ' % i + r['src'].split('+')[-1].replace(':', '_'))[:60]
            r['new'] = b
            sa = hind_tibia_app if (r['leaf'].startswith('Component7(Mirror) (1)') and hind_tibia_app) else r['body'].appearance
            if sa:
                try:
                    a = apps.get(sa.name) or nd.appearances.itemByName(sa.name) or nd.appearances.addByCopy(sa, sa.name)
                    apps[sa.name] = a
                    b.appearance = a
                except Exception:
                    pass
    occs['base'].isGrounded = True
    check('all links have bodies', all(occs[l].component.bRepBodies.count for l in LINKS), {l: occs[l].component.bRepBodies.count for l in LINKS})
    base_mat = lib_material(app)
    if not check('library plastic material found', base_mat is not None):
        return
    pla = nd.materials.addByCopy(base_mat, 'PLA (RL)')
    set_density(pla, PLA, next(r for r in recs if r['mat'] == 'pla')['new'])
    servo_vol = {}
    for r in recs:
        if r['mat'] == 'servo':
            servo_vol[r['group']] = servo_vol.get(r['group'], 0) + r['vol']
    smat = nd.materials.addByCopy(base_mat, 'Servo DS-843MG 8.5g (RL)')
    set_density(smat, SERVO_G / (sum(servo_vol.values()) / len(servo_vol)), next(r for r in recs if r['mat'] == 'servo')['new'])
    emats = {}
    for r in recs:
        if r['mat'] == 'elec' and r['group'] not in emats:
            vol = sum(x['vol'] for x in recs if x['group'] == r['group'])
            target = ELEC_G[r['group'].rsplit(':', 1)[0]]
            emats[r['group']] = m = nd.materials.addByCopy(base_mat, ('%s %gg (RL)' % (r['group'], target))[:60])
            set_density(m, target / vol, r['new'])
    for r in recs:
        r['new'].material = pla if r['mat'] == 'pla' else (smat if r['mat'] == 'servo' else emats[r['group']])
    _res['link_mass_g'] = {l: round(occs[l].physicalProperties.mass * 1000, 1) for l in LINKS}

    def P(x, y, z):
        p = adsk.core.Point3D.create(x, y, z)
        p.transformBy(M)
        return p

    def V(x, y, z):
        v = adsk.core.Vector3D.create(x, y, z)
        v.transformBy(M)
        return v
    joints = []
    for leg in LEGS:
        end, ax = leg[0], MAP['abduction_axes'][leg]                          # hip-servo spline axis (source frame)
        tx = occs[leg + '_thigh'].physicalProperties.centerOfMass.y + MID_X   # back to source x
        cx = occs[leg + '_calf'].physicalProperties.centerOfMass.y + MID_X
        joints += [(leg + '_hip_joint', leg + '_hip', 'base', P(*ax['point_cm']), V(*ax['dir']), LIMIT['hip']),
                   (leg + '_thigh_joint', leg + '_thigh', leg + '_hip', P(tx, *HIP[end]), V(1, 0, 0), LIMIT['thigh']),
                   (leg + '_calf_joint', leg + '_calf', leg + '_thigh', P(cx, *KNEE[end]), V(1, 0, 0), LIMIT['calf'])]
    for jn, child, parent, p, v, lim in joints:
        sk, circ = pivot_circle(nroot, jn, p, v)
        ji = nroot.asBuiltJoints.createInput(occs[child], occs[parent],
                                             adsk.fusion.JointGeometry.createByCurve(circ, adsk.fusion.JointKeyPointTypes.CenterKeyPoint))
        ji.setAsRevoluteJointMotion(adsk.fusion.JointDirections.ZAxisJointDirection)
        j = nroot.asBuiltJoints.add(ji)
        j.name = jn
        rl = j.jointMotion.rotationLimits
        rl.isMinimumValueEnabled, rl.minimumValue = True, -lim
        rl.isMaximumValueEnabled, rl.maximumValue = True, lim
        rl.isRestValueEnabled, rl.restValue = True, 0.0
        sk.isVisible = False

    def far_vertices(occ):
        for bi in range(occ.bRepBodies.count):
            b = occ.bRepBodies.item(bi)
            if b.vertices.count:
                return [b.vertices.item(k).geometry for k in (0, b.vertices.count // 2, b.vertices.count - 1)]
        return []

    def radial(q, p, a):
        w = adsk.core.Vector3D.create(q.x - p.x, q.y - p.y, q.z - p.z)
        t = w.dotProduct(a)
        return math.sqrt(max(w.length ** 2 - t * t, 0.0)), t
    for jn, child, parent, p, v, lim in joints:
        j = nroot.asBuiltJoints.itemByName(jn)
        a = v.copy()
        a.normalize()
        before = [q.copy() for q in far_vertices(occs[child])]
        j.jointMotion.rotationValue = 0.2
        adsk.doEvents()
        after = [q.copy() for q in far_vertices(occs[child])]
        j.jointMotion.rotationValue = 0.0
        adsk.doEvents()
        back = [q.copy() for q in far_vertices(occs[child])]
        err = max([max(abs(radial(qb, p, a)[0] - radial(qa, p, a)[0]), abs(radial(qb, p, a)[1] - radial(qa, p, a)[1]))
                   for qb, qa in zip(before, after)] or [99.0])
        moved = any(qb.distanceTo(qa) > 1e-3 for qb, qa in zip(before, after))
        returned = all(qb.distanceTo(qc) < 1e-4 for qb, qc in zip(before, back))
        check(jn, err < 1e-4 and moved and returned, {'rigid_rotation_err_cm': round(err, 6)})
    check('12 joints', nroot.asBuiltJoints.count == 12, nroot.asBuiltJoints.count)


def _finish():
    new_doc, folder, src_doc = _res.pop('new_doc', None), _res.pop('folder', None), _res.pop('src_doc', None)
    ok = _res['checks'] and all(v['ok'] for v in _res['checks'].values()) and not _res.get('status', '').startswith(('ABORT', 'retry'))
    if new_doc and ok and not DRY_RUN:
        _res['saved'] = new_doc.saveAs(_res['new_name'], folder, 'RL link+joint model from %s v%s' % (SOURCE_NAME, _res.get('source_version')), '')
        _res['status'] = 'ok'
    elif new_doc:
        _res['status'] = 'ok (dry run, discarded)' if ok else 'FAILED CHECKS: new document discarded'
        new_doc.close(False)
    elif not _res.get('status'):
        _res['status'] = 'ABORT: a check failed before the new document was created'
    if src_doc:
        src_doc.close(False)  # drop the in-memory UNJOIN; the design copy stays as saved


def _write():  # result file appears only once, with a final status
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'build_link_model.json'), 'w') as f:
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
