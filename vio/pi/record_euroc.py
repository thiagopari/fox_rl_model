#!/usr/bin/env python3
"""Record the Fox Pi's camera + BNO055 in the EuRoC layout: OUT/mav0/cam0/data/<ns>.png + data.csv, OUT/mav0/imu0/data.csv.

NEVER touches the PCA9685 or any servo. Do not run it while deploy/fox_pi.py runs: it switches the BNO055 from the
policy's IMUPLUS fusion mode to raw ACCGYRO mode (bandwidths below) for ~200 Hz polling, and back to IMUPLUS at exit.

    python3 record_euroc.py OUT [--seconds 60] [--fps 30] [--size 640x480] [--exposure-us 4000 --gain 4]
    python3 record_euroc.py OUT --no-camera --seconds 7200          # IMU only: static log for an Allan-variance fit

IMU axes are the raw BNO055 axes (no IMU_TO_BODY remap): Kalibr estimates T_cam_imu. Both streams are stamped on
CLOCK_MONOTONIC: camera = libcamera SensorTimestamp, IMU = midpoint of the I2C burst read. The summary prints the
camera's arrival latency (should be a few tens of ms; huge or negative = different clocks) and the IMU rate, gaps and
repeated samples (a repeat means polling outran the sensor's output rate: lower IMU_HZ or raise the bandwidths).
"""
import argparse
import os
import struct
import threading
import time

IMU_HZ = 200
ACC_LSB, GYR_LSB = 1 / 100, 0.001090830782496456   # m/s^2 and rad/s per LSB (BNO055 default units, = adafruit_bno055)


def imu_loop(out, stop, stats):
    import adafruit_bno055
    import board
    import busio
    imu = adafruit_bno055.BNO055_I2C(busio.I2C(board.SCL, board.SDA))
    imu.mode = adafruit_bno055.ACCGYRO_MODE
    imu.accel_range, imu.accel_bandwidth = adafruit_bno055.ACCEL_4G, adafruit_bno055.ACCEL_125HZ
    imu.gyro_range, imu.gyro_bandwidth = adafruit_bno055.GYRO_500_DPS, adafruit_bno055.GYRO_116HZ
    buf, last = bytearray(18), None
    period, nxt = 10**9 // IMU_HZ, time.monotonic_ns()
    try:
        with open(os.path.join(out, "data.csv"), "w") as f:
            f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
            while not stop.is_set():
                nxt += period
                time.sleep(max(0, nxt - time.monotonic_ns()) / 1e9)
                t0 = time.monotonic_ns()
                try:                                     # one burst: accel 0x08, mag 0x0E (unused), gyro 0x14
                    with imu.i2c_device as dev:
                        dev.write_then_readinto(bytes([0x08]), buf)
                except OSError:                          # BNO055 clock stretching vs the Pi's I2C: drop the sample
                    stats["errors"] += 1
                    continue
                t = (t0 + time.monotonic_ns()) // 2
                v = struct.unpack("<9h", buf)
                stats["repeats"] += v == last
                last = v
                stats["t"].append(t)
                f.write(f"{t},{v[6] * GYR_LSB:.6f},{v[7] * GYR_LSB:.6f},{v[8] * GYR_LSB:.6f},"
                        f"{v[0] * ACC_LSB:.4f},{v[1] * ACC_LSB:.4f},{v[2] * ACC_LSB:.4f}\n")
    finally:
        imu.mode = adafruit_bno055.IMUPLUS_MODE         # what deploy/fox_pi.py expects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--exposure-us", type=int, help="fixed exposure (auto-exposure off); short = less motion blur")
    ap.add_argument("--gain", type=float, help="fixed analogue gain, with --exposure-us")
    ap.add_argument("--no-camera", action="store_true")
    a = ap.parse_args()
    mav = os.path.join(a.out, "mav0")
    os.makedirs(os.path.join(mav, "imu0"), exist_ok=True)
    stop, stats = threading.Event(), {"t": [], "errors": 0, "repeats": 0}
    th = threading.Thread(target=imu_loop, args=(os.path.join(mav, "imu0"), stop, stats))
    th.start()
    lat, frames = [], 0
    try:
        if a.no_camera:
            time.sleep(a.seconds)
        else:
            from picamera2 import Picamera2
            from PIL import Image
            w, h = (int(x) for x in a.size.split("x"))
            cam_dir = os.path.join(mav, "cam0", "data")
            os.makedirs(cam_dir, exist_ok=True)
            controls = {"FrameDurationLimits": (int(1e6 / a.fps),) * 2}
            if a.exposure_us:
                controls.update(AeEnable=False, ExposureTime=a.exposure_us, **({"AnalogueGain": a.gain} if a.gain else {}))
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"size": (w, h), "format": "YUV420"}, controls=controls))
            cam.start()
            end = time.monotonic() + a.seconds
            with open(os.path.join(mav, "cam0", "data.csv"), "w") as f:
                f.write("#timestamp [ns],filename\n")
                while time.monotonic() < end:
                    req = cam.capture_request()
                    ts = req.get_metadata()["SensorTimestamp"]
                    lat.append(time.monotonic_ns() - ts)
                    gray = req.make_array("main")[:h, :w].copy()      # Y plane = grayscale (mono and colour sensors)
                    req.release()
                    Image.fromarray(gray).save(os.path.join(cam_dir, f"{ts}.png"), compress_level=1)
                    f.write(f"{ts},{ts}.png\n")
                    frames += 1
            cam.stop()
    finally:
        stop.set()
        th.join()
    t = stats["t"]
    gaps = [b - c for b, c in zip(t[1:], t[:-1])]
    if gaps:
        gaps.sort()
        print(f"IMU: {len(t)} samples, {1e9 * (len(t) - 1) / (t[-1] - t[0]):.1f} Hz, median dt {gaps[len(gaps) // 2] / 1e6:.2f} ms, "
              f"max gap {gaps[-1] / 1e6:.1f} ms, {stats['errors']} I2C errors, {stats['repeats']} repeated samples")
    if lat:
        lat.sort()
        print(f"camera: {frames} frames, arrival latency median {lat[len(lat) // 2] / 1e6:.1f} ms "
              f"(min {lat[0] / 1e6:.1f}, max {lat[-1] / 1e6:.1f})")


if __name__ == "__main__":
    main()
