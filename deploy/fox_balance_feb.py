#!/usr/bin/env python3
"""The February balance controller (~/robofox_balance_controller_v3.py, 2026-02-02) on today's robot.

As in February: BNO055 Euler angles with the upside-down remap, one PID per axis (Kp 1.0, Ki 0.1, Kd 0.5), dead zones
roll 0.5 / pitch 1.0 deg, corrections up to +-25 deg, 50 Hz, and the same leg motion: per leg the first channel moves
+c and the second -c (the whole-leg pendulum swing), with RL = roll + pitch, RR = -roll + pitch, FR = roll - pitch,
FL = -roll - pitch, each servo kept within +-45 deg of its centre.
Changed for today's robot:
  * the centres are today's home angles (servo_calib.json, set with fox_calib.py), not February's 90 / 60 deg with
    the second servo at 180 minus the first;
  * the target attitude is captured over the first --capture s (hold the robot steady the way it should stay) instead
    of February's fixed roll 1.1 / pitch -0.4 deg, which also absorbs the re-mounted IMU's tilt;
  * the rear hips hold their home angle (February left the hips unpowered);
  * 45 deg off the target for 3 readings counts as a fall and goes limp, like Ctrl-C.

    ssh -t fox-wifi 'cd ~/fox && venv/bin/python fox_balance_feb.py'      (--dry-run: no hardware)
"""
import argparse
import time

import fox_pi as fp

LEG_CHANNELS = {"RL": (0, 1), "RR": (3, 4), "FR": (6, 7), "FL": (9, 10)}   # February's gear1, gear2 of each leg
HIP_CHANNELS = (2, 5)                                                       # rear hips (the front ones aren't connected)
KP, KI, KD = 1.0, 0.1, 0.5
DEAD_ZONE_ROLL, DEAD_ZONE_PITCH, MAX_CORRECTION, SPAN, FALL = 0.5, 1.0, 25.0, 45.0, 45.0   # deg
LOOP_PERIOD = 1.0 / 50


class PIDController:
    """February's PID, unchanged."""

    def __init__(self, kp, ki, kd, output_min, output_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_min, self.output_max = output_min, output_max
        self.integral, self.prev_error, self.prev_time = 0.0, 0.0, time.time()

    def update(self, error, current_time):
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.001
        self.integral += error * dt
        output = self.kp * error + self.ki * self.integral + self.kd * (error - self.prev_error) / dt
        output = max(self.output_min, min(self.output_max, output))
        if output == self.output_max or output == self.output_min:   # anti-windup
            self.integral -= error * dt
        self.prev_error, self.prev_time = error, current_time
        return output


def name_of(ch):
    return next(n for n in fp.SERVOS if fp.CHANNEL[n] == ch)


def centre(ch):
    return fp.NEUTRAL_DEG[name_of(ch)]


def read_angles(imu):
    """February's read_imu: (roll, pitch) in deg from the Euler angles, IMU mounted upside down; None on a bad read."""
    try:
        euler = imu.euler
    except OSError:
        return None
    if euler is None or None in euler:
        return None
    heading, roll, pitch = euler
    roll, pitch = -roll, pitch - 180
    if pitch < -180:
        pitch += 360
    return roll, pitch


def leg_commands(roll_correction, pitch_correction):
    """February's apply_balance_correction: {channel: servo deg} around today's centres."""
    corrections = {"RL": roll_correction + pitch_correction, "RR": -roll_correction + pitch_correction,
                   "FR": roll_correction - pitch_correction, "FL": -roll_correction - pitch_correction}
    out = {}
    for leg, c in corrections.items():
        ch1, ch2 = LEG_CHANNELS[leg]
        out[ch1] = min(centre(ch1) + SPAN, max(centre(ch1) - SPAN, centre(ch1) + c))
        out[ch2] = min(centre(ch2) + SPAN, max(centre(ch2) - SPAN, centre(ch2) - c))   # opposite direction
    return out


class FakeImu:
    euler = (0.0, 0.0, 180.0)   # level, upside down


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=float, default=10.0, help="seconds to capture the target attitude")
    ap.add_argument("--seconds", type=float, help="stop after this long and keep holding (default: until Ctrl-C)")
    ap.add_argument("--dry-run", action="store_true", help="no hardware")
    a = ap.parse_args()
    robot = fp.FakeRobot() if a.dry_run else fp.Robot()
    imu = FakeImu() if a.dry_run else robot.imu

    def write(commands):
        for ch, deg in commands.items():
            deg = min(180.0, max(0.0, deg))
            if not a.dry_run:
                robot.servos[fp.SERVOS.index(name_of(ch))].angle = deg

    finished = False
    try:
        write({ch: centre(ch) for ch in HIP_CHANNELS + sum(LEG_CHANNELS.values(), ())})
        print("servos at their centres (today's homes); capturing the target for %.1f s: hold the robot steady"
              % a.capture, flush=True)
        samples, end = [], time.time() + a.capture
        while time.time() < end:
            r = read_angles(imu)
            if r:
                samples.append(r)
            time.sleep(LOOP_PERIOD)
        if not samples:
            raise RuntimeError("no IMU reading during the capture")
        target_roll = sum(s[0] for s in samples) / len(samples)
        target_pitch = sum(s[1] for s in samples) / len(samples)
        print("target roll %+.2f deg, pitch %+.2f deg (%d readings); balancing, Ctrl-C = limp"
              % (target_roll, target_pitch, len(samples)), flush=True)
        pid_roll = PIDController(KP, KI, KD, -MAX_CORRECTION, MAX_CORRECTION)
        pid_pitch = PIDController(KP, KI, KD, -MAX_CORRECTION, MAX_CORRECTION)
        start, loops, off = time.time(), 0, 0
        while a.seconds is None or time.time() - start < a.seconds:
            loop_start = time.time()
            r = read_angles(imu)
            if r is None:
                time.sleep(LOOP_PERIOD)
                continue
            roll, pitch = r
            roll_error, pitch_error = target_roll - roll, target_pitch - pitch
            off = off + 1 if max(abs(roll_error), abs(pitch_error)) > FALL else 0
            if off >= 3:
                print("fallen over (%.0f deg off the target): stopping" % max(abs(roll_error), abs(pitch_error)))
                break
            if abs(roll_error) < DEAD_ZONE_ROLL:
                roll_error = 0
            if abs(pitch_error) < DEAD_ZONE_PITCH:
                pitch_error = 0
            now = time.time()
            roll_correction = pid_roll.update(roll_error, now)
            pitch_correction = pid_pitch.update(pitch_error, now)
            write(leg_commands(roll_correction, pitch_correction))
            loops += 1
            if loops % 10 == 0:
                print("Roll: %6.2f (err %5.2f, corr %5.2f) | Pitch: %6.2f (err %5.2f, corr %5.2f)"
                      % (roll, roll_error, roll_correction, pitch, pitch_error, pitch_correction), flush=True)
            time.sleep(max(0.0, LOOP_PERIOD - (time.time() - loop_start)))
        finished = a.seconds is not None and time.time() - start >= a.seconds
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if finished:
            print("time up: holding the last pose")
        else:
            robot.limp()
            print("servos released (limp)")


if __name__ == "__main__":
    main()
