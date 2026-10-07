#!/usr/bin/env python3
"""Calibrate the Fox's servos by hand over SSH: drive any servo to an angle, and record its home (the CAD stance), safe
range, direction and channel in servo_calib.json, which fox_pi.py reads at start.

    ssh -t fox-wifi 'cd ~/fox && venv/bin/python fox_calib.py'        (-t: it needs a terminal;  --dry-run: no hardware)

All servos start limp and nothing moves until you press a move key. A servo's first move goes to its saved home angle
(hobby servos can't report where they are, so that first move is a jump). Every change to home / limits / direction /
channel is saved immediately.
"""
import argparse
import json
import os
import sys
import termios
import tty

from fox_pi import CALIB_FILE, CHANNEL, DIRECTION, NEUTRAL_DEG, PLUS, SERVOS

HELP = """ up/down  select servo       left/right  -/+1 deg     A / D  -/+5 deg      g  go to an angle      c  go to home
 h  home = this angle        < / >  min / max = this angle   r  flip direction   #  change channel
 space  limp this servo      L  limp all                q  quit (all limp)
 direction: +1 if right-arrow (larger angle) moves the joint the way "+ means" says, else -1"""


class Servos:
    """PCA9685 channels as angles, 0-180 deg over 500-2500 us like fox_pi.py; limp = no pulses."""

    def __init__(self, dry):
        self.dry, self.used = dry, {}
        if not dry:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
            self.pca = PCA9685(busio.I2C(board.SCL, board.SDA), address=0x40)
            self.pca.frequency = 50

    def move(self, ch, deg):
        if self.dry:
            return
        if ch not in self.used:
            from adafruit_motor import servo
            self.used[ch] = servo.Servo(self.pca.channels[ch], min_pulse=500, max_pulse=2500)
        self.used[ch].angle = deg

    def limp(self, ch):
        if ch in self.used:
            self.used[ch].fraction = None

    def close(self):
        for ch in list(self.used):
            self.limp(ch)
        if not self.dry:
            self.pca.deinit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no hardware")
    args = ap.parse_args()
    tty_in = sys.stdin.isatty()
    saved = {}
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            saved = json.load(f)
    limits = {k: list(v) for k, v in saved.get("limit_deg", {}).items() if k in SERVOS}   # only the ones you set
    stamp = os.path.getmtime(CALIB_FILE) if os.path.exists(CALIB_FILE) else None       # to spot edits made meanwhile
    now = {n: None for n in SERVOS}                                                       # None = limp
    hw, sel = Servos(args.dry_run), 0

    def lim(n):
        return limits.get(n, [NEUTRAL_DEG[n] - 50.0, NEUTRAL_DEG[n] + 50.0])

    def save():
        """Write the file, unless something else changed it since this session loaded or saved it: then the session's
        old copy would overwrite that change, so refuse (the caller says to restart)."""
        nonlocal stamp
        if (os.path.getmtime(CALIB_FILE) if os.path.exists(CALIB_FILE) else None) != stamp:
            return False
        data = {}
        if os.path.exists(CALIB_FILE):                   # keep entries this tool doesn't edit (imu_level_gravity)
            with open(CALIB_FILE) as f:
                data = json.load(f)
        data.update({"channel": CHANNEL, "home_deg": NEUTRAL_DEG, "direction": DIRECTION, "limit_deg": limits})
        tmp = CALIB_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, CALIB_FILE)
        stamp = os.path.getmtime(CALIB_FILE)
        return True

    stale = "NOT SAVED: servo_calib.json was changed outside this session. Quit (q) and restart to load it"

    def go(n, deg):
        deg = min(180.0, max(0.0, round(float(deg), 1)))
        hw.move(CHANNEL[n], deg)
        now[n] = deg
        lo, hi = lim(n)
        return "%s (ch %d) -> %.1f deg%s" % (n, CHANNEL[n], deg, "" if lo <= deg <= hi else "  OUTSIDE its range %.1f-%.1f" % (lo, hi))

    def draw(status):
        if not tty_in:                                   # piped keys (tests): status lines only
            if status:
                print(status, flush=True)
            return
        rows = ["\x1b[2J\x1b[H Fox servo calibration -> %s%s" % (CALIB_FILE, "   [DRY RUN]" if args.dry_run else ""), "",
                "   servo      ch   home   now     range (*=set)   dir  + means"]
        for i, n in enumerate(SERVOS):
            lo, hi = lim(n)
            rows.append("%s %-9s %3d  %5.1f  %-6s  %5.1f-%5.1f %s   %+d   %s" % (
                ">" if i == sel else " ", n, CHANNEL[n], NEUTRAL_DEG[n], "limp" if now[n] is None else "%.1f" % now[n],
                lo, hi, "*" if n in limits else " ", DIRECTION[n], PLUS[n.split("_")[1]]))
        rows += ["", " " + status, "", HELP]
        sys.stdout.write("\n".join(rows) + "\n")
        sys.stdout.flush()

    def key():
        c = sys.stdin.read(1)
        if c == "\x1b":
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(sys.stdin.read(2), "esc")
        return c

    def ask(prompt):
        buf = ""
        while True:
            draw(prompt + buf + "_")
            c = sys.stdin.read(1)
            if c in ("\n", "\r", ""):
                return buf
            if c == "\x1b":
                return None
            if c in ("\x7f", "\b"):
                buf = buf[:-1]
            elif c in "0123456789.-":
                buf += c

    old = termios.tcgetattr(sys.stdin) if tty_in else None
    if tty_in:
        tty.setcbreak(sys.stdin)
    msg = "all servos limp; nothing moves until you press a move key"
    try:
        while True:
            draw(msg)
            k = key()
            n = SERVOS[sel]
            if k in ("q", ""):
                break
            if k in ("up", "k", "down", "j"):
                sel = (sel + (-1 if k in ("up", "k") else 1)) % len(SERVOS)
                msg = ""
            elif k in ("left", "right", "a", "d", "A", "D"):
                step = {"left": -1, "a": -1, "right": 1, "d": 1, "A": -5, "D": 5}[k]
                msg = go(n, NEUTRAL_DEG[n]) + "  (first move: home)" if now[n] is None else go(n, now[n] + step)
            elif k == "g":
                v = ask("%s: go to angle (0-180, Enter; Esc cancels): " % n)
                try:
                    msg = go(n, float(v)) if v else "cancelled"
                except ValueError:
                    msg = "not a number: %r" % v
            elif k == "c":
                msg = go(n, NEUTRAL_DEG[n])
            elif k == " ":
                hw.limp(CHANNEL[n])
                now[n], msg = None, "%s limp" % n
            elif k == "L":
                for m in SERVOS:
                    hw.limp(CHANNEL[m])
                    now[m] = None
                msg = "all limp"
            elif k in ("h", "<", ">"):
                if now[n] is None:
                    msg = "move %s first: these keys record the angle it is at" % n
                elif k == "h":
                    NEUTRAL_DEG[n] = now[n]
                    msg = "home of %s = %.1f deg (saved)" % (n, now[n]) if save() else stale
                else:
                    lo, hi = lim(n)
                    lo, hi = (now[n], max(hi, now[n])) if k == "<" else (min(lo, now[n]), now[n])
                    limits[n] = [lo, hi]
                    msg = "range of %s = %.1f-%.1f deg (saved)" % (n, lo, hi) if save() else stale
            elif k == "r":
                DIRECTION[n] = -DIRECTION[n]
                msg = "direction of %s = %+d (saved)" % (n, DIRECTION[n]) if save() else stale
            elif k == "#":
                v = ask("%s: channel (0-15, Enter; Esc cancels): " % n)
                if v and v.isdigit() and 0 <= int(v) <= 15:
                    hw.limp(CHANNEL[n])
                    CHANNEL[n], now[n] = int(v), None
                    others = [m for m in SERVOS if m != n and CHANNEL[m] == int(v)]
                    msg = ("channel of %s = %s (saved)%s" % (n, v, "  WARNING: also used by " + ", ".join(others) if others else "")
                           if save() else stale)
                else:
                    msg = "cancelled" if not v else "channel must be 0-15"
    finally:
        hw.close()
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("\nall servos limp; calibration in %s" % CALIB_FILE)


if __name__ == "__main__":
    main()
