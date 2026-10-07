#!/usr/bin/env python3
"""Run a trained Fox policy (policy.onnx, e.g. policies/fox_flat_blind_v3) on the robot.

Raspberry Pi + BNO055 + PCA9685 (0x40) on I2C, the stack of ~/robofox_leg_imu_calibration.py. 50 Hz loop:
IMU -> the policy's observation -> policy.onnx -> 12 servo angles -> PCA9685.

    pip install numpy onnxruntime adafruit-circuitpython-pca9685 adafruit-circuitpython-bno055 adafruit-circuitpython-motor
    python3 fox_pi.py policy.onnx --check-imu       # 1. IMU axes: tilt / turn the robot, compare with the printout
    python3 fox_pi.py policy.onnx --wiggle FL_hip   # 2. robot held in the air: which channel, which way is + (only that
                                                    #    servo is powered: centre, +10 deg, centre, -10 deg, centre, x2)
    python3 fox_pi.py policy.onnx                   # 3. walk. W/S A/D Q/E step the command, space = stop, Ctrl-C = limp
    python3 fox_pi.py policy.onnx --dry-run         # anywhere: fake hardware, 3 s, prints the servo angles

Calibrate CHANNEL / NEUTRAL_DEG / DIRECTION / IMU_TO_BODY below first. Servo target 0 is the CAD stance: legs in vertical
planes, femur (hip pivot -> knee) -20.5 deg front / +30.5 deg rear from straight down (+ = knee forward), shin (knee ->
ankle) +4.5 deg front / -44.5 deg rear. The servos jump to the stance at start: hold the robot or use a stand.
FoxPolicy (no hardware) is the input/output contract; tools/check_policy_io.py runs it in Isaac Lab against the sim.
"""
import argparse
import collections
import math
import select
import sys
import termios
import time
import tty

import numpy as np

# ---------------------------------------------------------------------------- calibration (measure on the robot)
LEGS = ("FL", "FR", "RL", "RR")
SERVOS = ["%s_%s" % (leg, s) for s in ("hip", "pivot", "gear") for leg in LEGS]     # the policy's output order
# PCA9685 channels. Leg blocks from ~/robofox_fourbar_kinematics.py (RL 0-2, RR 3-5, FR 6-8, FL 9-11). Inside a block,
# GUESSED from the January script's "gear1"/"gear2" pair: servo 1 = gear (pinion, sets the shin), servo 2 = pivot (on the
# femur), servo 3 = hip. Check with --wiggle
CHANNEL = {"RL_gear": 0, "RL_pivot": 1, "RL_hip": 2, "RR_gear": 3, "RR_pivot": 4, "RR_hip": 5,
           "FR_gear": 6, "FR_pivot": 7, "FR_hip": 8, "FL_gear": 9, "FL_pivot": 10, "FL_hip": 11}
NEUTRAL_DEG = {name: 90.0 for name in SERVOS}   # adafruit servo angle (0..180) that puts the joint at the CAD stance
DIRECTION = {name: 1 for name in SERVOS}        # -1 where a larger servo angle turns the joint against the policy's +
DEG_PER_RAD = 180.0 / math.pi                   # 500-2500 us = 180 deg (adafruit_motor); correct if the travel differs
LIMIT_DEG = {name: (NEUTRAL_DEG[name] - 50.0, NEUTRAL_DEG[name] + 50.0) for name in SERVOS}   # never command beyond
IMU_TO_BODY = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]   # BNO055 axes -> body (x fwd, y left, z up): upside-down mount,
#   verified with --check-imu 2026-10-07 (nose down x +0.99, left side down y +1.00, CCW turn +93 deg, gyro in rad/s)
PLUS = {"hip": "the foot moves to the robot's LEFT", "pivot": "the femur turns, knee moving BACK; the shin keeps its angle",
        "gear": "the shin turns, foot moving FORWARD; the femur stays"}   # policy + for each servo (fox_cfg.py servo map)

DT, MAX_TARGET = 0.02, 0.8   # 50 Hz like training; |target| clamp in rad (training stays well inside)


class FoxPolicy:
    """policy.onnx's contract, checked in Isaac Lab by tools/check_policy_io.py. Observation (105 floats): per term the
    last 5 readings, oldest first: gyro (rad/s, body), gravity direction (unit, (0, 0, -1) level), command (vx, vy m/s,
    yaw rad/s), its own previous output as sent (12). Output: servo targets = 0.25 * output (rad from the stance), SERVOS
    order. For straight commands (no sideways, no turning) the hip outputs are set to 0: the hips hold the stance."""

    H, SCALE = 5, 0.25

    def __init__(self, infer):
        self.infer = infer

    def reset(self, gyro, gravity, command):
        self.hist = [collections.deque([np.array(x, np.float32)] * self.H, maxlen=self.H)    # copies: callers may
                     for x in (gyro, gravity, command, np.zeros(12))]                              # reuse their buffers
        return self._act(command)

    def step(self, gyro, gravity, command):
        for h, x in zip(self.hist, (gyro, gravity, command, self.action)):
            h.append(np.array(x, np.float32))
        return self._act(command)

    def _act(self, command):
        self.obs = np.concatenate([np.concatenate(h) for h in self.hist])[None]
        self.action = np.array(self.infer(self.obs), np.float32).reshape(12)     # as sent: what the policy observes next
        if not hips_free(command):
            self.action[:4] = 0.0
        return self.SCALE * self.action


def hips_free(command):
    """Sideways or turning asked for; otherwise the hips hold the stance (= isaaclab/fox_mdp.py hips_free)."""
    return abs(command[1]) > 1e-3 or abs(command[2]) > 1e-3


def onnx_infer(path):
    """policy.onnx as obs (1, 105) -> action (12,): onnxruntime on the Pi, onnx's reference evaluator elsewhere."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(path)
        name = session.get_inputs()[0].name
        return lambda obs: session.run(None, {name: obs})[0][0]
    except ImportError:
        import onnx
        from onnx.reference import ReferenceEvaluator
        model = onnx.load(path)
        evaluator, name = ReferenceEvaluator(model), model.graph.input[0].name
        return lambda obs: evaluator.run(None, {name: obs})[0][0]


def servo_deg(name, rad):
    lo, hi = LIMIT_DEG[name]
    return min(hi, max(lo, NEUTRAL_DEG[name] + DIRECTION[name] * DEG_PER_RAD * rad))


class Robot:
    """BNO055 + PCA9685 through Adafruit Blinka, as in ~/robofox_leg_imu_calibration.py."""

    def __init__(self):
        import adafruit_bno055
        import board
        import busio
        from adafruit_motor import servo
        from adafruit_pca9685 import PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_bno055.BNO055_I2C(i2c)
        self.imu.mode = adafruit_bno055.IMUPLUS_MODE    # gyro + accelerometer fusion: no magnetometer near the servo motors
        self.pca = PCA9685(i2c, address=0x40)
        self.pca.frequency = 50
        self.servos = [servo.Servo(self.pca.channels[CHANNEL[n]], min_pulse=500, max_pulse=2500) for n in SERVOS]
        self.rot = np.array(IMU_TO_BODY, float)

    def read(self):
        """(gyro rad/s, gravity unit vector pointing down), body frame; None when the BNO055 drops a reading."""
        try:
            gyro, grav = self.imu.gyro, self.imu.gravity
        except OSError:                                 # BNO055 clock stretching vs the Pi's I2C: retry
            return None
        if None in gyro or None in grav or not any(grav):
            return None
        down = -(self.rot @ np.array(grav, float))      # the BNO055 reports gravity pointing up (+z when level)
        return self.rot @ np.array(gyro, float), down / np.linalg.norm(down)

    def write(self, targets):
        # ponytail: 12 single-channel I2C writes (~1 ms each); batch them if the loop overruns
        for s, name, t in zip(self.servos, SERVOS, targets):
            s.angle = servo_deg(name, t)

    def write_one(self, name, target):
        """Drive one servo; the others get no pulses (limp)."""
        self.servos[SERVOS.index(name)].angle = servo_deg(name, target)

    def limp(self):
        for s in self.servos:
            s.fraction = None
        self.pca.deinit()


class FakeRobot:
    """--dry-run: level and still, prints the servo angles once a second."""

    def __init__(self):
        self.k = 0

    def read(self):
        return np.zeros(3), np.array([0.0, 0.0, -1.0])

    def write(self, targets):
        self.k += 1
        if self.k % 50 == 0:
            print("servo deg:", " ".join("%s %.1f" % (n, servo_deg(n, t)) for n, t in zip(SERVOS, targets)))

    def write_one(self, name, target):
        print("  (fake) %s -> %.1f deg" % (name, servo_deg(name, target)))

    def limp(self):
        pass


class Keys:
    """Keyboard over SSH: W/S, A/D, Q/E step the command (the sim's keys), space or L = stop."""

    STEP = {"w": (0.1, 0, 0), "s": (-0.1, 0, 0), "a": (0, 0.1, 0), "d": (0, -0.1, 0), "q": (0, 0, 0.5), "e": (0, 0, -0.5)}
    LIMIT = np.array([0.3, 0.2, 1.0])   # the training command ranges

    def __init__(self):
        self.cmd, self.saved = np.zeros(3), None
        if sys.stdin.isatty():
            self.saved = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)

    def poll(self):
        while self.saved and select.select([sys.stdin], [], [], 0)[0]:
            c = sys.stdin.read(1).lower()
            if c in " l":
                self.cmd[:] = 0.0
            elif c in self.STEP:
                self.cmd = np.clip(self.cmd + self.STEP[c], -self.LIMIT, self.LIMIT)
            print("command vx %+.2f m/s  vy %+.2f m/s  yaw %+.2f rad/s" % tuple(self.cmd), flush=True)
        return self.cmd

    def close(self):
        if self.saved:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)


def read_retry(robot):
    for _ in range(25):                  # the BNO055 reads zeros for ~0.1 s after a mode change; I2C glitches
        r = robot.read()
        if r is not None:
            return r
        time.sleep(0.02)
    raise RuntimeError("no IMU reading for 0.5 s")


def walk(robot, policy, steps=None):
    keys, overruns, k = Keys(), 0, 0
    try:
        targets = policy.reset(*read_retry(robot), keys.cmd)
        tick = time.perf_counter()
        while steps is None or k < steps:
            robot.write(np.clip(targets, -MAX_TARGET, MAX_TARGET))
            tick += DT
            late = tick - time.perf_counter()
            if late > 0:
                time.sleep(late)
            else:
                overruns += 1
            gyro, down = read_retry(robot)
            if down[2] > -0.5:                          # tilted past 60 deg: fallen over
                print("tilted over: stopping")
                break
            targets = policy.step(gyro, down, keys.poll())
            k += 1
    finally:
        keys.close()
        robot.limp()
        print("%d steps, %d over the 20 ms budget" % (k, overruns))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help="policy.onnx (its .onnx.data next to it)")
    p.add_argument("--dry-run", action="store_true", help="fake hardware, 3 s")
    p.add_argument("--check-imu", action="store_true", help="print body-frame gyro and gravity")
    p.add_argument("--wiggle", choices=SERVOS, help="move one servo +/- around its centre (the others stay limp)")
    p.add_argument("--amplitude", type=float, default=10.0, help="--wiggle size in deg (57 for timing a servo on video)")
    p.add_argument("--set", metavar="SERVO=DEG", help="move one servo straight to DEG from its centre, hold 2 s, limp")
    a = p.parse_args()
    if a.set and (a.set.split("=")[0] not in SERVOS or a.set.count("=") != 1):
        p.error("--set needs SERVO=DEG with SERVO one of %s" % ", ".join(SERVOS))
    robot = FakeRobot() if a.dry_run else Robot()
    try:
        if a.check_imu:
            print("expect: level -> gravity (0, 0, -1); nose down -> x > 0; left side down -> y > 0; turn left -> gyro z > 0")
            while True:
                gyro, down = read_retry(robot)
                print("gyro %+.2f %+.2f %+.2f rad/s   gravity %+.2f %+.2f %+.2f" % (*gyro, *down), flush=True)
                time.sleep(0.2)
        if a.set:
            name, deg = a.set.split("=")
            print("%s on channel %d -> %.1f deg (centre %.1f %+.0f), hold 2 s, then limp"
                  % (name, CHANNEL[name], servo_deg(name, math.radians(float(deg))), NEUTRAL_DEG[name], float(deg)), flush=True)
            robot.write_one(name, math.radians(float(deg)))
            time.sleep(2.0)
            return
        if a.wiggle:
            amp = math.radians(a.amplitude)
            print("%s on channel %d, centre %.1f deg, +/-%.0f deg; + means: %s"
                  % (a.wiggle, CHANNEL[a.wiggle], NEUTRAL_DEG[a.wiggle], a.amplitude, PLUS[a.wiggle.split("_")[1]]), flush=True)
            for t in (0.0, amp, 0.0, -amp) * 2 + (0.0,):
                print("  %-6s %.1f deg  (t = %.3f s)" % ({0.0: "centre"}.get(t, "+" if t > 0 else "-"), servo_deg(a.wiggle, t),
                                                          time.perf_counter()), flush=True)
                robot.write_one(a.wiggle, t)
                time.sleep(1.0)
            return
    except KeyboardInterrupt:
        return
    finally:
        if a.check_imu or a.wiggle or a.set:
            robot.limp()
    walk(robot, FoxPolicy(onnx_infer(a.policy)), steps=150 if a.dry_run else None)


if __name__ == "__main__":
    main()
