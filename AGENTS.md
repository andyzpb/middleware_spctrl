# AGENTS.md

## Repository Rules

- Fail fast. Do not hide invalid state, missing inputs, unsafe commands, or broken assumptions behind silent fallbacks.
- Keep tests and production code separate. Test stubs, dry-run harnesses, fake ROS messages, and analysis scripts must not be mixed into runtime control paths.
- Do not add meaningless `try`/`except` blocks. Catch only specific exceptions when the code can recover or add useful context before re-raising.
- Prefer explicit validation over defensive guessing. If a required calibration, topic, config entry, or HDF5 dataset is missing, raise a clear error.
- Keep hardware interfaces narrow. `ros_control.py` should use ROS topics and the configured motor mapping contract, not direct hardware SDK calls.
- Preserve experiment logs. HDF5 fields and attrs should describe units, controller choices, trajectory settings, safety limits, and abort reasons clearly.
