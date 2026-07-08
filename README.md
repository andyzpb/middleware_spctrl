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
| bending | 18 | Position Control Mode only | -20 to +20 continuum deg |

ID018 bending is the protected axis:

- Operating Mode(11): `3`
- Home tick: `1536`
- Min Position Limit(52): `1195`
- Max Position Limit(48): `1877`
- Logical range: `-20..+20` continuum degrees

The tick range is the authority. The logical bend angle is a robot calibration,
not the XL430 horn angle.

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
outside `1195..1877`, the command fails before motion.

## Aurora EM CLI

EM sensors are role-based. The code reads roles such as `tip`, `base`, and
`aux`; `config/em.yaml` maps those roles to the current NDI tool indices.

```yaml
sensors:
  - name: tip
    role: tip
    tool_index: 10
  - name: base
    role: base
    tool_index: 11
```

When a third sensor is installed, add it as `aux`:

```yaml
  - name: aux
    role: aux
    tool_index: 12
```

The default `10/11` values are lab configuration, not a hard-coded
assumption. If NDI returns a different order, edit `config/em.yaml`.

```bash
python -m continuum_control.em_cli --config config/em.yaml status
python -m continuum_control.em_cli --config config/em.yaml read --samples 1
python -m continuum_control.em_cli --config config/em.yaml pair --samples 1
```

`read` allows tip-only smoke testing. `pair` requires live `tip` and `base`
roles and prints the tip position in the base sensor frame.

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

- ID018 bend mapping: `-20 -> 1195`, `0 -> 1536`, `20 -> 1877`
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
4. Run `arm`; verify mode `3`, limits `1195..1877`, and no hardware error.
5. Run a small bending jog, for example `--bending-deg 2`.
6. Run `home`.
7. Add ID008 and ID012 after ID018 is proven safe.
