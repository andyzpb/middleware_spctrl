from __future__ import annotations

import argparse
import sys
import time

from .aurora_agent import AuroraAgent, AuroraError
from .config import ConfigError, EMConfig, load_em_config
from .em_core import EMError, EMSample, fresh_samples, relative_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emctl")
    parser.add_argument("--config", default="config/em.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    read = subparsers.add_parser("read")
    read.add_argument("--samples", type=int, default=1)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--samples", type=int, default=1)
    return parser


def main(argv: list[str] | None = None, config_loader=load_em_config, agent_factory=AuroraAgent) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    agent = None

    try:
        config = config_loader(args.config)
        if args.command == "status":
            _print_config(config)
            return 0

        if args.samples < 1:
            raise EMError("samples must be >= 1")
        if args.command == "pair":
            config.sensor_by_role("tip")
            config.sensor_by_role("base")

        agent = agent_factory(config)
        agent.start()
        for _ in range(args.samples):
            samples = agent.read_samples()
            if args.command == "read":
                _print_samples(samples)
            elif args.command == "pair":
                pair = fresh_samples(samples, ("tip", "base"), time.monotonic(), config.timeout_s)
                rel = relative_transform(pair["base"], pair["tip"])
                print(
                    " ".join(
                        [
                            "tip_in_base",
                            f"x_mm={rel[0][3]:.3f}",
                            f"y_mm={rel[1][3]:.3f}",
                            f"z_mm={rel[2][3]:.3f}",
                        ]
                    )
                )
            else:
                parser.error(f"unknown command {args.command}")
        return 0
    except (ConfigError, EMError, AuroraError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if agent is not None:
            agent.close()


def _print_config(config: EMConfig) -> None:
    print(
        " ".join(
            [
                f"serial_port={config.serial_port}",
                f"ports_to_probe={config.ports_to_probe}",
                f"timeout_s={config.timeout_s}",
                f"sample_hz={config.sample_hz}",
            ]
        )
    )
    for sensor in config.sensors:
        print(f"role={sensor.role} name={sensor.name} tool_index={sensor.tool_index}")


def _print_samples(samples: dict[str, EMSample]) -> None:
    for sample in samples.values():
        if not sample.valid:
            print(
                " ".join(
                    [
                        f"role={sample.role}",
                        f"name={sample.name}",
                        f"tool_index={sample.tool_index}",
                        "valid=0",
                        f"error={sample.error or 'invalid'}",
                    ]
                )
            )
            continue
        x, y, z = sample.position_mm
        print(
            " ".join(
                [
                    f"role={sample.role}",
                    f"name={sample.name}",
                    f"tool_index={sample.tool_index}",
                    "valid=1",
                    f"x_mm={x:.3f}",
                    f"y_mm={y:.3f}",
                    f"z_mm={z:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
