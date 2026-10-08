#!/usr/bin/env python3
"""One leg's pendulum and extension, commanded the way Claude's servo-role model says (try it on RR).

    ssh -t fox-wifi 'cd ~/fox && venv/bin/python fox_leg_modes.py --ratio 1'    (quit fox_calib.py first; --dry-run: no hardware)

The model: the middle (pivot) servo sets the femur's angle and the pinion (gear) servo sets the shin's (through the crank,
at 1/ratio of the servo angle), each HOLDING while the other moves. From home (servo_calib.json home_deg, the CAD stance):
  pendulum a (+ = foot forward): middle -a and pinion +ratio*a together. The knee angle stays; the leg turns about the hip.
  extension dh (+ = longer): middle and pinion together, from fox_pi.leg_ik. The ankle moves straight down.
  one servo alone, the other holding (to compare): middle alone turns the femur and the shin keeps its angle; pinion alone
  turns the shin about the knee and the femur stays. The model says neither is a pendulum.
Only the leg's hip, middle and pinion servos get pulses: support the robot with that foot free. Nothing moves until h,
which jumps them to home (hobby servos can't report where they are). Moves are ramped over 0.4 s; a move that would take
a servo past its range is refused.
"""
import argparse
import math
import sys
import termios
import time
import tty

from fox_calib import Servos
from fox_pi import CHANNEL, DEG_PER_RAD, DIRECTION, LEGS, LIMIT_DEG, NEUTRAL_DEG, leg_fk, leg_ik, leg_reach

STEP_DEG, STEP_MM, RAMP_S, DT = 5.0, 5.0, 0.4, 0.02
KEYS = {"w": ("swing", 1), "s": ("swing", -1), "e": ("ext", 1), "d": ("ext", -1),
        "P": ("pivot", 1), "p": ("pivot", -1), "G": ("gear", 1), "g": ("gear", -1)}
HELP = """ w / s  pendulum: leg swings forward / back 5 deg     e / d  extension: foot 5 mm down / up (leg longer / shorter)
 P / p  middle (pivot) servo alone +/-5 deg             G / g  pinion (gear) servo alone +/-5 deg
 h  home (the first h jumps the leg's servos there)    q  quit (all limp)"""


def targets(leg, n, m):
    """(hip, pivot, gear) in rad from home: the extension from the 1:1 IK (its gear = the shin's turn), the pendulum
    (the whole leg turned by m["swing"]), then the single-servo offsets (servo deg)."""
    piv, shin = leg_ik(leg, m["ext"]) if m["ext"] else (0.0, 0.0)
    a = math.radians(m["swing"])
    return [0.0, piv - a + math.radians(m["pivot"]), n * (shin + a) + math.radians(m["gear"])]


def predict(leg, n, piv, gear):
    """What the model expects, both servos holding their targets: ankle mm forward / up from home, knee bend change (deg)."""
    x0, z0 = leg_fk(leg, 0.0, 0.0)
    x, z = leg_fk(leg, piv, gear / n)
    return x - x0, z - z0, math.degrees(-piv - gear / n)


def servo_deg(name, rad):
    return NEUTRAL_DEG[name] + DIRECTION[name] * DEG_PER_RAD * rad


def in_range(name, deg):
    lo, hi = LIMIT_DEG[name]
    return max(lo, 0.0) <= deg <= min(hi, 180.0)


def check(leg, n):
    """The commands do what the model says: the pendulum keeps the knee and the hip-ankle distance, the extension keeps
    the ankle's fore-aft position, the pinion alone leaves the femur."""
    zero = dict.fromkeys(("swing", "ext", "pivot", "gear"), 0.0)
    x0, z0 = leg_fk(leg, 0.0, 0.0)
    dx, dz, dk = predict(leg, n, *targets(leg, n, dict(zero, swing=10.0))[1:])
    assert abs(dk) < 1e-9 and abs(math.hypot(x0 + dx, z0 + dz) - math.hypot(x0, z0)) < 1e-6, "pendulum"
    dx, dz, dk = predict(leg, n, *targets(leg, n, dict(zero, ext=10.0))[1:])
    assert abs(dx) < 1e-6 and abs(dz + 10.0) < 1e-6, "extension"
    assert targets(leg, n, dict(zero, gear=5.0))[1] == 0.0, "pinion alone"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=float, required=True, help="pinion turns per crank turn: 1 = 12T:12T gears, 2 = v7 12T:24T")
    ap.add_argument("--leg", default="RR", choices=LEGS)
    ap.add_argument("--dry-run", action="store_true", help="no hardware")
    args = ap.parse_args()
    leg, n = args.leg, args.ratio
    check(leg, n)
    names = ["%s_%s" % (leg, s) for s in ("hip", "pivot", "gear")]
    hw, tty_in = Servos(args.dry_run), sys.stdin.isatty()
    m, now = dict.fromkeys(("swing", "ext", "pivot", "gear"), 0.0), None      # now: targets sent (rad), None = limp

    def send(t):
        for name, v in zip(names, t):
            hw.move(CHANNEL[name], servo_deg(name, v))

    def go(t):
        nonlocal now
        k_max = max(1, int(RAMP_S / DT))
        for k in range(1, k_max + 1):
            send([a + (b - a) * k / k_max for a, b in zip(now, t)])
            time.sleep(DT)
        now = t

    def status(what):
        dx, dz, dk = predict(leg, n, *now[1:])
        print("%-26s servo deg %s | model: ankle %+5.1f mm fwd %+5.1f mm up, knee bend %+5.1f deg"
              "   [pendulum %+.0f deg, extension %+.0f mm, middle alone %+.0f, pinion alone %+.0f]" % (
                  what, " ".join("%s %.1f" % (s.split("_")[1], servo_deg(s, v)) for s, v in zip(names, now)), dx, dz, dk,
                  m["swing"], m["ext"], m["pivot"], m["gear"]), flush=True)

    old = termios.tcgetattr(sys.stdin) if tty_in else None
    if tty_in:
        tty.setcbreak(sys.stdin)
    print("%s\n%s, gears %g:1, home = %s%s. Support the robot with %s's foot free; nothing moves until h." % (
        HELP, leg, n, ", ".join("%s %.1f" % (s.split("_")[1], NEUTRAL_DEG[s]) for s in names),
        "   [DRY RUN]" if args.dry_run else "", leg), flush=True)
    try:
        while True:
            c = sys.stdin.read(1)
            if c in ("q", ""):
                break
            if c == "h":
                m = dict.fromkeys(m, 0.0)
                if now is None:
                    now = [0.0] * 3
                    send(now)
                    status("jumped to home, holding")
                else:
                    go([0.0] * 3)
                    status("home")
            elif c in KEYS:
                if now is None:
                    print("press h first: it jumps %s's servos to home" % leg, flush=True)
                    continue
                key, sign = KEYS[c]
                trial = dict(m)
                trial[key] += sign * (STEP_MM if key == "ext" else STEP_DEG)
                t = targets(leg, n, trial)
                bad = ["%s %.1f deg" % (s, servo_deg(s, v)) for s, v in zip(names, t) if not in_range(s, servo_deg(s, v))]
                if trial["ext"] > leg_reach(leg):
                    bad.append("leg nearly straight")
                if bad:
                    print("refused (%s): %s" % (c, ", ".join(bad)), flush=True)
                    continue
                m = trial
                go(t)
                status({"swing": "pendulum", "ext": "extension", "pivot": "middle alone", "gear": "pinion alone"}[key]
                       + (" +" if sign > 0 else " -"))
    finally:
        hw.close()
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("all limp", flush=True)


if __name__ == "__main__":
    main()
