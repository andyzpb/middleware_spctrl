# EM Trajectory Tracking Protocol

## Goal

Compare two tip-tracking controllers using an EM sensor fixed near the probe tip:

1. Pure EM PID
2. IK + EM PID

The reference trajectory, measured tip position, and tracking error are all defined in the EM field frame.

## Before the experiment

1. Place the Aurora field generator in a fixed position.
2. Keep the probe, EM sensor, and field generator away from large metal objects and strong electromagnetic noise sources.
3. Attach the EM sensor close to the probe tip.
4. Open NDI Toolbox and confirm that the Aurora system sees the tool and returns stable tracking data.
5. Close NDI Toolbox before running the Python script if it keeps the serial port open.
6. Install dependencies:

```bash
pip install -r em_tracking_requirements_simple.txt
````

## Aurora read test

Edit `test_aurora_read_simple.py`:

```python
SERIAL_PORT = "COM7"  # replace with your Aurora SCU port
```

Then run:

```bash
python test_aurora_read_simple.py
```

Move the EM sensor by hand. The printed `p_F_mm` values should change smoothly.

## Main script configuration

Edit `em_trajectory_tracking_aurora_simple.py`:

```python
DXL_PORT = "COM10"
AURORA_SERIAL_PORT = "COM7"
TRAJ_SHAPE = "circle"  # or "square"
EM_TRAJ_RADIUS_MM = 0.25
EM_TRAJ_SIDE_MM = 0.50
```

Start with a small circle first. Do not start with a large square.

## Automatic calibration steps

The main script performs these steps:

1. Return the robot to home.
2. Move to a safe center pose using the simple PCC IK.
3. Average the EM sensor position at this pose. This becomes the trajectory center in the EM frame.
4. Move a small step in robot +x and record the EM displacement. This defines `ex_F`.
5. Move a small step in robot +y and record the EM displacement. This sets the sign of `ey_F`.
6. Move a small step in robot +z and record the EM displacement. This defines `ez_F`.
7. Build an orthonormal EM-frame trajectory basis: `ex_F`, `ey_F`, `ez_F`.
8. Measure the local EM Jacobian around the center pose for pure EM PID.

## Experiment sequence

The script then runs:

1. `pure_em_pid`
2. Return home
3. `ik_plus_em_pid`
4. Return home

For both experiments:

* The reference path is defined in the EM frame.
* The measured position is the EM sensor position in the EM frame.
* The tracking error is `p_ref_F - p_tip_F`.

## Output files

The script writes:

* `em_trajectory_tracking_log.csv`
* `em_trajectory_tracking_summary.csv`

Use the summary file to compare:

* `rms_error_mm`
* `mean_error_mm`
* `max_error_mm`
* `rms_ex_mm`
* `rms_ey_mm`
* `rms_ez_mm`
* `num_rate_limited`
* `num_pid_saturated`
* `num_ik_unreachable_cmd`

## Recommended first trials

Run in this order:

1. Circle, radius 0.25 mm
2. Circle, radius 0.50 mm
3. Square, side 0.50 mm
4. Square, side 0.80 mm

Stop increasing the trajectory size if the error grows quickly, commands saturate, or the probe motion looks unsafe.
