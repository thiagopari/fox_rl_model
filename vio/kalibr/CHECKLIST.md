# Calibration session checklist (Fox nose camera + BNO055)

Nothing here moves the robot: no `deploy/` code, servos unpowered, `deploy/fox_pi.py` not running.

## Before
- [ ] Camera mounted rigidly on the nose, on the same rigid part as the BNO055 (any flex between them breaks `T_cam_imu`);
      M12 focus set at ~1 m and locked (lock nut or a drop of glue).
- [ ] OV9281: `camera_auto_detect=0` and `dtoverlay=ov9281` (Arducam module: `dtoverlay=ov9281,arducam`) in
      `/boot/firmware/config.txt`; `rpicam-hello --list-cameras` lists it.
- [ ] `picamera2` importable from the Python that runs `vio/pi/record_euroc.py` (it is an apt package,
      `python3-picamera2`; a venv needs `--system-site-packages`).
- [ ] AprilGrid printed on A3 (`aprilgrid.yaml`), glued flat to a rigid board; measure one tag's edge and set `tagSize`.
- [ ] Smoke test: `python3 record_euroc.py /tmp/t --seconds 10 --exposure-us 4000`. Check the summary: IMU near 200 Hz,
      few I2C errors / repeated samples, camera arrival latency a few tens of ms (huge or negative = clock mismatch).

## Recordings (`record_euroc.py OUT ...`)
1. **IMU noise** (do first, or overnight): `--no-camera --seconds 7200`, robot still on a table, after ~10 min warm-up.
   Fit the Allan variance and replace the four values in `imu.yaml`.
2. **Camera intrinsics**, 60–90 s: robot still, move the board slowly through the whole image (corners and edges too),
   tilted up to ~45°, 0.3–1 m away. Short fixed exposure (`--exposure-us 3000`–`5000`), bright even light.
3. **Camera–IMU**, 60–90 s, twice: board fixed, move the ROBOT by hand. 2–3 s still at the start and end; in between,
   brisk but smooth rotations about all three axes and translations along all three, board always in view.

## Processing (laptop, Kalibr's Docker image, ROS 1)
- Bag per recording with `kalibr_bagcreater --folder DIR --output-bag X.bag`; it expects `DIR/cam0/*.png` and
  `DIR/imu0.csv`: symlink `mav0/cam0/data` to `DIR/cam0`, copy `mav0/imu0/data.csv` to `DIR/imu0.csv` (check that its
  header line is skipped).
- `kalibr_calibrate_cameras --target aprilgrid.yaml --models pinhole-radtan --topics /cam0/image_raw --bag cam.bag`
- `kalibr_calibrate_imu_camera --target aprilgrid.yaml --imu imu.yaml --cams camchain-cam.yaml --bag camimu.bag`

## Accept when (rules of thumb)
- Reprojection error RMS well under 1 px, residuals without structure across the image.
- `T_cam_imu` rotation within a few degrees of the CAD prior in `camchain_template.yaml`; the two camera–IMU runs agree.
- `timeshift_cam_imu` small and the same in both runs (it absorbs the stamp offset between exposure and I2C read).
- Then copy the results into OpenVINS' `kalibr_imucam_chain.yaml` (`T_imu_cam` = inverse of `T_cam_imu`) and
  `kalibr_imu_chain.yaml`.
