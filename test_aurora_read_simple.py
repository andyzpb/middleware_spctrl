#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_aurora_read_simple.py

Minimal NDI Aurora read test using scikit-surgerynditracker.
Run this before running the robot controller.
"""

import time
import numpy as np
from sksurgerynditracker.nditracker import NDITracker

# Set this to the Aurora SCU serial port, for example "COM7".
# If None, the library probes the first PORTS_TO_PROBE serial ports.
SERIAL_PORT = None
PORTS_TO_PROBE = 20
TOOL_INDEX = 0


def main():
    settings = {
        "tracker type": "aurora",
        "verbose": True,
    }
    if SERIAL_PORT:
        settings["serial port"] = SERIAL_PORT
    else:
        settings["ports to probe"] = PORTS_TO_PROBE

    tracker = NDITracker(settings)

    try:
        tracker.start_tracking()
        print("Tracking started. Move the EM sensor by hand.")

        for i in range(200):
            port_handles, timestamps, frame_numbers, tracking, quality = tracker.get_frame()

            if tracking is None or len(tracking) <= TOOL_INDEX:
                print(f"{i:03d}: no tracking data")
                time.sleep(0.05)
                continue

            mat = np.asarray(tracking[TOOL_INDEX], dtype=float)
            if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
                print(f"{i:03d}: invalid transform")
                time.sleep(0.05)
                continue

            p = mat[:3, 3]
            q = None
            if quality is not None and len(quality) > TOOL_INDEX:
                q = quality[TOOL_INDEX]

            print(f"{i:03d}: p_F_mm = [{p[0]:8.3f}, {p[1]:8.3f}, {p[2]:8.3f}], quality = {q}")
            time.sleep(0.05)

    finally:
        try:
            tracker.stop_tracking()
        except Exception:
            pass
        try:
            tracker.close()
        except Exception:
            pass
        print("Closed tracker.")


if __name__ == "__main__":
    main()