#!/usr/bin/env python3
"""Set each leg's home (the CAD pose) by moving the leg instead of its servos: swing it (pendulum) or extend / retract
its knee until it matches the CAD pose, then save its pivot and gear servo angles as their homes in servo_calib.json.

    ssh -t fox-wifi 'cd ~/fox && venv/bin/python fox_home.py'      (quit fox_calib.py first; --dry-run: no hardware)

CAD pose, side view, angles from straight down (+ = toward the front), legs in vertical planes:
    front: femur (hip pivot -> knee) -20.5 deg (knee back),     shin (knee -> ankle) +4.5 deg
    rear:  femur +30.5 deg (knee forward),                       shin -44.5 deg
Both motions are fixed servo ratios, so they do the same whatever the current calibration (N = gear_ratio):
    swing a (+ = foot forward): pivot -a, gear +N a. The knee angle stays; the whole leg turns about the hip.
    extend r mm (+ = longer):   both servos, from the leg IK (fox_pi.leg_ik radial): the ankle ends on the same line
                                from the hip axis (where the crank gear turns), so the leg keeps its angle and only its
                                length changes. At home the front legs are ~97 % straight: they lengthen only ~2 mm
                                (the IK keeps 15 deg of knee bend), the rear ones ~30 mm; both shorten freely. A longer
                                step than the leg can take goes as far as it can.
                                It runs as a swing, then the pivot alone: the gear servo can't turn its gear against the
                                femur (the gears skip teeth, and held in mesh they stall), so it only ever turns with
                                the femur (a swing), and the knee bends by the pivot alone (fox_leg_modes.py p / P).
Gear-servo play compensation: the gear path has 10-15 deg of servo play, which swallowed small moves. When a gear
servo reverses, it is commanded half the play (--gear-play, total, deg) extra in the new direction so the shin moves at
once. The angles shown and saved are the play's centre, which is what the sim calls the target. Servos move at most
SPEED deg/s (a jump makes the servo push its stall torque at once).
Nothing moves until you pick a leg: its hip, pivot and gear servos jump to their saved homes. A leg you leave keeps
holding. A move that would take a servo past 0-180 deg is refused.
"""
import argparse
import json
import math
import os
import sys
import termios
import time
import tty

import fox_pi
from fox_calib import Servos
from fox_pi import CALIB_FILE, CHANNEL, DIRECTION, LEGS, NEUTRAL_DEG, leg_fk, leg_ik, leg_reach

HELP = """ 1 2 3 4  pick FL / FR / RL / RR (its servos jump to their homes)
 w / s    swing the leg forward / back 1 deg (W / S: 5 deg)
 e / d    leg 5 mm longer / shorter at the same angle (E / D: 15 mm): a swing, then the pivot alone
 h        save this leg's pivot and gear angles as their homes  space   limp this leg     q   quit (all limp)"""
KEYS = {"w": ("swing", 1.0), "s": ("swing", -1.0), "W": ("swing", 5.0), "S": ("swing", -5.0),
        "e": ("extend", 5.0), "d": ("extend", -5.0), "E": ("extend", 15.0), "D": ("extend", -15.0)}
SPEED, DT = 25.0, 0.02                                   # deg/s per servo (fox_leg_modes: 5-10 deg in 0.4 s worked)


def deltas(leg, motion, amount, n, pivot=0.0, gear=0.0):
    """(pivot, gear) servo changes in the policy's + sense (deg): a swing of `amount` deg, or from the current pose
    (pivot, gear deg from home) an extension of `amount` mm along the hip-ankle line (None past the leg's reach)."""
    if motion == "swing":
        return -amount, n * amount
    if amount > 0:
        amount = min(amount, leg_reach(leg, math.radians(pivot), math.radians(gear)))
        if amount < 0.5:
            return None
    p, g = leg_ik(leg, amount, math.radians(pivot), math.radians(gear), radial=True)
    return math.degrees(p) - pivot, math.degrees(g) - gear


def split(dl, n):
    """An extension (pivot, gear servo deltas) as a swing that does all of its gear-servo motion (pivot -a, gear +n a:
    femur and crank gear turn together), then the pivot alone (the gear holds)."""
    dp, dg = dl
    return [(-dg / n, dg), (dp + dg / n, 0.0)]


def check(n):
    """A swing keeps the knee angle; an extension keeps the leg's angle (hip -> ankle) and changes its length by r."""
    for leg in LEGS:
        l1, l2, a1, a2 = fox_pi.LEG_GEOM[leg[0]]
        angle = lambda p, g: (a1 - math.radians(p), a2 + math.radians(g) / n)   # noqa: E731  femur, shin from down
        f, sh = angle(*deltas(leg, "swing", 5.0, n))
        assert abs((f - sh) - (a1 - a2)) < 1e-12, "swing keeps the knee"
        x0, z0 = leg_fk(leg, 0.0, 0.0)
        for r in (-10.0, 1.0):
            x, z = leg_fk(leg, *(math.radians(v) for v in deltas(leg, "extend", r, n)))
            assert abs(math.atan2(x, -z) - math.atan2(x0, -z0)) < 1e-9, "extension keeps the leg's angle"
            assert abs(math.hypot(x, z) - math.hypot(x0, z0) - r) < 1e-6, "extension changes the length by r"
            (sp, sg), (pp, pg) = split(deltas(leg, "extend", r, n), n)
            assert pg == 0.0 and abs(sp + sg / n) < 1e-12, "the gear servo moves only in the swing"
            assert abs(sp + pp - deltas(leg, "extend", r, n)[0]) < 1e-12, "the parts add up"


def save_homes(updates):
    """Write the given homes into servo_calib.json (read just before, so other entries and tools' edits survive)."""
    data = {}
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            data = json.load(f)
    data.setdefault("home_deg", {}).update(updates)
    with open(CALIB_FILE + ".tmp", "w") as f:
        json.dump(data, f, indent=1)
    os.replace(CALIB_FILE + ".tmp", CALIB_FILE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no hardware")
    ap.add_argument("--gear-play", type=float, default=12.0,
                    help="gear servo play to take up on a reversal (total deg, 0 = off)")
    args = ap.parse_args()
    n = fox_pi.GEAR_RATIO                                  # servo_calib.json's gear_ratio (fox_calib.py --gear-ratio)
    check(n)
    hw, tty_in = Servos(args.dry_run), sys.stdin.isatty()
    now, leg = {}, None                                   # servo -> target deg (the play's centre; holding servos only)
    comp = {}                                             # gear servo -> play taken up in its last direction (deg)

    sent = {}                                             # servo -> last angle sent (deg, play compensation in)

    def go(name, deg):
        sent[name] = min(180.0, max(0.0, deg + comp.get(name, 0.0)))
        hw.move(CHANNEL[name], sent[name])
        now[name] = deg

    def ramp(targets):
        """Move servos (name -> deg) together, the farthest at SPEED deg/s."""
        end = {s: min(180.0, max(0.0, d + comp.get(s, 0.0))) for s, d in targets.items()}
        start = dict((s, sent[s]) for s in targets)
        k_max = max(1, math.ceil(max(abs(end[s] - start[s]) for s in targets) / (SPEED * DT)))
        for k in range(1, k_max + 1):
            for s in targets:
                hw.move(CHANNEL[s], start[s] + (end[s] - start[s]) * k / k_max)
            time.sleep(DT)
        sent.update(end)
        now.update(targets)

    def status(what):
        p, g = leg + "_pivot", leg + "_gear"
        print("%-28s %s: pivot %.1f deg (home %.1f, %+.1f)   gear %.1f deg (home %.1f, %+.1f)" % (
            what, leg, now[p], NEUTRAL_DEG[p], now[p] - NEUTRAL_DEG[p],
            now[g], NEUTRAL_DEG[g], now[g] - NEUTRAL_DEG[g]), flush=True)

    old = termios.tcgetattr(sys.stdin) if tty_in else None
    if tty_in:
        tty.setcbreak(sys.stdin)
    print("%s\ngears %g:1, homes from %s%s. Support the robot; nothing moves until you pick a leg."
          % (HELP, n, CALIB_FILE, "   [DRY RUN]" if args.dry_run else ""), flush=True)
    try:
        while True:
            c = sys.stdin.read(1)
            if c in ("q", ""):
                break
            if c in "1234":
                leg = LEGS["1234".index(c)]
                for s in ("hip", "pivot", "gear"):
                    name = "%s_%s" % (leg, s)
                    if name not in now:
                        go(name, NEUTRAL_DEG[name])
                status("picked")
            elif leg is None:
                print("pick a leg first (1-4)", flush=True)
            elif c in KEYS:
                motion, amount = KEYS[c]
                p, g = leg + "_pivot", leg + "_gear"
                pm, gm = ((now[x] - NEUTRAL_DEG[x]) * DIRECTION[x] for x in (p, g))      # policy + sense
                short = motion == "extend" and amount > leg_reach(leg, math.radians(pm), math.radians(gm))
                dl = deltas(leg, motion, amount, n, pm, gm)
                if dl is None:
                    print("refused: the leg can't get longer here (it keeps 15 deg of knee bend)", flush=True)
                    continue
                plan, tp, tg = [], now[p], now[g]
                for dp, dg in (split(dl, n) if motion == "extend" else [dl]):
                    tp, tg = tp + DIRECTION[p] * dp, tg + DIRECTION[g] * dg
                    plan.append((round(tp, 2), round(tg, 2)))
                if not all(0.0 <= v <= 180.0 for t in plan for v in t):
                    print("refused: pivot %.1f / gear %.1f deg would leave 0-180" % (tp, tg), flush=True)
                    continue
                for tp, tg in plan:
                    if abs(tg - now[g]) > 1e-6:                   # the gear servo moves: take up its play that way
                        comp[g] = math.copysign(args.gear_play / 2.0, tg - now[g])
                    ramp({p: tp, g: tg})
                status("swing %+.0f deg" % amount if motion == "swing" else "leg %+.0f mm" % amount)
                if short:
                    print("  (only as far as it goes: the leg keeps 15 deg of knee bend)", flush=True)
            elif c == "h":
                upd = {leg + "_pivot": now[leg + "_pivot"], leg + "_gear": now[leg + "_gear"]}
                save_homes(upd)
                NEUTRAL_DEG.update(upd)
                status("homes saved")
            elif c == " ":
                for s in ("hip", "pivot", "gear"):
                    name = "%s_%s" % (leg, s)
                    hw.limp(CHANNEL[name])
                    now.pop(name, None)
                    comp.pop(name, None)
                    sent.pop(name, None)
                print("%s limp (pick it again to hold it at its home)" % leg, flush=True)
                leg = None
    finally:
        hw.close()
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("all limp", flush=True)


if __name__ == "__main__":
    main()
