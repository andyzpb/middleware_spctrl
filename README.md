# Continuum Robot Control Middleware

Cross-platform control middleware for a continuum robot driven by three
DYNAMIXEL XL430-W250 actuators through a ROBOTIS U2D2.

The middleware is native Python and keeps the same command surface on macOS,
Linux, and Windows. ROS2 and EM PID can sit on top of the same safety core
later.

## Motor Contract

| Role | ID | Required mode | Safe command |
|---|---:|---|---|
| translation | 8 | Position mode by default | raw tick for now |
| rotation | 12 | Position mode by default | raw tick for now |
| bending | 18 | Position Control Mode only | -25 to +25 deg from home |

ID018 bending is the protected axis:

- Operating Mode(11): `3`
- Home tick: `1652`
- Min Position Limit(52): `1368`
- Max Position Limit(48): `1936`
- Logical range: `-25..+25` degrees from home

The tick range is the authority. The logical bend angle is measured relative to
the current ID018 home position.

## Layout

```text
continuum_control/
  core.py          shared config, bending map, SafetyGate
  config.py        YAML config loader
  serial_ports.py  cross-platform U2D2 port discovery
  dxl_agent.py     only layer allowed to use dynamixel_sdk
  cli.py           continuumctl command-line interface
  em_core.py       EM sample validation and relative pose math
  aurora_agent.py  only layer allowed to use sksurgery NDITracker
  em_cli.py        emctl command-line interface
  calibrate_em.py  rotation/bending EM calibration sweep

config/
  robot.yaml       shared hardware contract
  em.yaml          Aurora sensor-role mapping

tests/
  test_*.py        runtime-independent self-checks
```

## Setup

```bash
python -m pip install -r requirements-control.txt
```

U2D2 does not power the motors. Use an external DYNAMIXEL-compatible power
supply before running hardware commands.

## Motor CLI

Use `--port auto` for automatic U2D2 discovery, or pass an explicit port:

- macOS: `/dev/cu.usbserial-*` or `/dev/tty.usbserial-*`
- Linux: `/dev/ttyUSB*` or `/dev/ttyACM*`
- Windows: `COM*`

```bash
python -m continuum_control.cli --config config/robot.yaml --port auto status
python -m continuum_control.cli --config config/robot.yaml --port auto arm
python -m continuum_control.cli --config config/robot.yaml --port auto home
python -m continuum_control.cli --config config/robot.yaml --port auto jog --bending-deg 5
python -m continuum_control.cli --config config/robot.yaml --port auto disarm
```

For safety, `home` and `jog` run the arming validation first. If ID018 is not in
Position Control Mode, its limits cannot be set, or its present position is
outside `1368..1936`, the command fails before motion.

## Aurora EM CLI

EM sensors are role-based. The code reads roles such as `tip`, `base`, and
`aux`; `config/em.yaml` maps those roles to the current NDI tool indices.

```yaml
sensors:
  - name: tip
    role: tip
    tool_index: 1
  - name: base
    role: base
    tool_index: 0
```

When a third sensor is installed, add it as `aux`:

```yaml
  - name: aux
    role: aux
    tool_index: 2
```

The default `1/0` values are the current lab configuration, not a hard-coded
assumption. If NDI returns a different order, edit `config/em.yaml`.

```bash
python -m continuum_control.em_cli --config config/em.yaml status
python -m continuum_control.em_cli --config config/em.yaml scan --max-index 16
python -m continuum_control.em_cli --config config/em.yaml read --samples 1
python -m continuum_control.em_cli --config config/em.yaml pair --samples 1
```

Use `scan` before the sensor roles are known. It prints raw Aurora tracking
indices, for example `index=0 valid=1 x_mm=...`; move one physical sensor at a
time and assign the moving index to `tip`, `base`, or `aux` in `config/em.yaml`.
`read` checks the configured roles. `pair` requires live `tip` and `base` roles
and prints the tip position in the base sensor frame.

## Rotation/Bending EM Calibration

The calibration sweep moves only ID012 rotation and ID018 bending. ID018 is
validated as XL430-W250 in Position Control Mode before motion. The default grid
is rotation offsets `-100, 0, +100` ticks and bending `-10, -5, 0, +5, +10`
degrees, with one second settle and one second EM sampling at `sample_hz`.

```bash
python -m continuum_control.calibrate_em \
  --robot-config config/robot.yaml \
  --em-config config/em.yaml \
  --motor-port /dev/cu.usbserial-FT4TFQFO \
  --output em_rotation_bending_calibration.csv
```

On completion or EM abort, the sweep returns ID012 and ID018 to home. ID018 is
left torque-enabled at `1652` so the structure does not relax away from center.

## Safety Behavior

- Operator commands outside the configured range fail fast.
- Controller commands may be clamped only through the shared SafetyGate.
- ID018 goals are never published unless ID018 is confirmed in Position Control
  Mode.
- Arming is two-phase: all motors are configured and checked before any motor is
  torque-enabled.
- If automatic serial discovery finds multiple U2D2-like ports, it fails and
  asks for an explicit port.

## Test

```bash
python -m unittest discover -s tests -v
python -m py_compile continuum_control/*.py
```

Current coverage checks:

- ID018 bend mapping: `-25 -> 1368`, `0 -> 1652`, `25 -> 1936`
- unsafe operator jog is rejected before loading hardware
- ambiguous serial-port discovery fails
- ID018 mode, min/max limits, and present-position checks happen before torque on
- CLI command names are consistent across platforms
- EM config rejects unknown roles, duplicate tool indices, and more than three
  sensors
- EM relative pose math is tested without Aurora hardware

## Recommended Hardware Commissioning

1. Connect only ID018 first.
2. Confirm external power and U2D2 wiring.
3. Run `status` with an explicit port.
4. Run `arm`; verify mode `3`, limits `1368..1936`, and no hardware error.
5. Run a small bending jog, for example `--bending-deg 2`.
6. Run `home`.
7. Add ID008 and ID012 after ID018 is proven safe.
