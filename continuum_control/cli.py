from __future__ import annotations

import argparse
from dataclasses import replace
import sys

from .config import ConfigError, load_robot_config
from .core import Command, SafetyError, SafetyGate
from .dxl_agent import DxlAgent, DxlError
from .serial_ports import SerialPortError, resolve_serial_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="continuumctl")
    parser.add_argument("--config", default="config/robot.yaml")
    parser.add_argument("--port", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("arm")
    subparsers.add_parser("disarm")
    subparsers.add_parser("home")
    jog = subparsers.add_parser("jog")
    jog.add_argument("--bending-deg", type=float, required=True)
    return parser


def _print_status(rows) -> None:
    for row in rows:
        print(
            " ".join(
                [
                    f"id={row.dxl_id}",
                    f"mode={row.operating_mode}",
                    f"torque={int(row.torque_enabled)}",
                    f"position={row.present_position}",
                    f"hardware_error={row.hardware_error_status}",
                    f"voltage_raw={row.input_voltage_raw}",
                    f"temperature_c={row.temperature_c}",
                ]
            )
        )


def main(argv: list[str] | None = None, agent_factory=DxlAgent) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    agent = None

    try:
        config = load_robot_config(args.config)
        port = resolve_serial_port(args.port or config.serial_port)
        config = replace(config, serial_port=port)
        if args.command == "jog":
            SafetyGate(config).command_to_goals(Command(bending_deg=args.bending_deg, source="operator"))
        agent = agent_factory(config)

        if args.command == "status":
            agent.open()
            _print_status(agent.status())
        elif args.command == "arm":
            agent.arm()
            print("armed")
        elif args.command == "disarm":
            agent.open()
            agent.disarm()
            print("disarmed")
        elif args.command == "home":
            agent.arm()
            agent.home()
            print("home command sent")
        elif args.command == "jog":
            agent.arm()
            agent.jog_bending(args.bending_deg)
            print("jog command sent")
        else:
            parser.error(f"unknown command {args.command}")
        return 0
    except (ConfigError, SerialPortError, DxlError, SafetyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
