#!/usr/bin/env python3
"""Set each leg's home (the CAD pose) by moving the leg instead of its servos: swing it (pendulum) or extend / retract
its knee until it matches the CAD pose, then save its pivot and gear servo angles as their homes in servo_calib.json.

    ssh -t fox-wifi 'cd ~/fox && venv/bin/python fox_home.py'      (quit fox_calib.py first; --dry-run: no hardware)

CAD pose, side view, angles from straight down (+ = toward the front), legs in vertical planes:
    front: femur (hip pivot -> knee) -20.5 deg (knee back),     shin (knee -> ankle) +4.5 deg
    rear:  femur +30.5 deg (knee forward),                       shin -44.5 deg
Both motions are fixed servo ratios, so they do the same whatever the current calibration (N = gear_ratio):
    swing a (+ = foot forward): pivot -a, gear +N a. The knee angle stays; the whole leg turns about the hip.
    knee  k (+ = straighter):   the pivot servo alone, +-k (the sign flips front / rear, whose knees bend opposite
                                ways), the gear servo holding: the femur turns and the shin keeps its angle (p / P in
                                fox_leg_modes.py; moving the gear servo too was lost in its play, 10-15 deg).
Nothing moves until you pick a leg: its hip, pivot and gear servos jump to their saved homes. A leg you leave keeps
holding. A move that would take a servo past 0-180 deg is refused.
"""
import argparse
import json
import math
import os
import sys
import termios
import tty

import fox_pi
from fox_calib import Servos
from fox_pi import CALIB_FILE, CHANNEL, DIRECTION, LEGS, NEUTRAL_DEG, leg_fk

HELP = """ 1 2 3 4  pick FL / FR / RL / RR (its servos jump to their homes)
 w / s    swing the leg forward / back 1 deg (W / S: 5 deg)    e / d   extend / retract the knee 1 deg (E / D: 5 deg)
 h        save this leg's pivot and gear angles as their homes  space   limp this leg     q   quit (all limp)"""
KEYS = {"w": ("swing", 1.0), "s": ("swing", -1.0), "W": ("swing", 5.0), "S": ("swing", -5.0),
        "e": ("knee", 1.0), "d": ("knee", -1.0), "E": ("knee", 5.0), "D": ("knee", -5.0)}


def deltas(leg, motion, deg, n):
    """(pivot, gear) servo targets in the policy's + sense (deg) for a swing or a knee change of `deg`."""
    if motion == "swing":
        return -deg, n * deg
    return (deg if leg[0] == "R" else -deg), 0.0      # rear knees point forward, front knees back


def check(n):
    """A swing keeps the knee angle; a knee change keeps the shin's angle and lengthens the leg when + (leg_fk)."""
    for leg in LEGS:
        l1, l2, a1, a2 = fox_pi.LEG_GEOM[leg[0]]
        angle = lambda p, g: (a1 - math.radians(p), a2 + math.radians(g) / n)   # noqa: E731  femur, shin from down
        f, sh = angle(*deltas(leg, "swing", 5.0, n))
        assert abs((f - sh) - (a1 - a2)) < 1e-12, "swing keeps the knee"
        f, sh = angle(*deltas(leg, "knee", 5.0, n))
        assert abs(sh - a2) < 1e-12 and abs(abs(f - sh) - abs(a1 - a2)) > 0.08, \
            "knee: the femur turns, the shin stays"
        x0, z0 = leg_fk(leg, 0.0, 0.0)
        x, z = leg_fk(leg, *(math.radians(v) for v in deltas(leg, "knee", 5.0, n)))
        assert math.hypot(x, z) > math.hypot(x0, z0), "knee + = longer leg"


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
    args = ap.parse_args()
    n = fox_pi.GEAR_RATIO                                  # servo_calib.json's gear_ratio (fox_calib.py --gear-ratio)
    check(n)
    hw, tty_in = Servos(args.dry_run), sys.stdin.isatty()
    now, leg = {}, None                                   # servo -> deg sent (holding servos only)

    def go(name, deg):
        hw.move(CHANNEL[name], deg)
        now[name] = deg

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
                motion, deg = KEYS[c]
                dp, dg = deltas(leg, motion, deg, n)
                p, g = leg + "_pivot", leg + "_gear"
                tp, tg = now[p] + DIRECTION[p] * dp, now[g] + DIRECTION[g] * dg
                if not (0.0 <= tp <= 180.0 and 0.0 <= tg <= 180.0):
                    print("refused: pivot %.1f / gear %.1f deg would leave 0-180" % (tp, tg), flush=True)
                    continue
                go(p, round(tp, 2))
                go(g, round(tg, 2))
                status("%s %+.0f deg" % ("swing" if motion == "swing" else "knee", deg))
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
                print("%s limp (pick it again to hold it at its home)" % leg, flush=True)
                leg = None
    finally:
        hw.close()
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("all limp", flush=True)


if __name__ == "__main__":
    main()
