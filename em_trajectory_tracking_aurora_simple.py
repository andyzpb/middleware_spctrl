#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
em_trajectory_tracking_pid_baseline_vs_ik_jacobian.py

Two EM-tip trajectory tracking experiments for a 3-DOF probe:

Experiment A: em_pid_no_jacobian_baseline
    Baseline controller with EM feedback and Cartesian PID only.
    No IK feedforward is used during tracking.
    No measured, dynamic, or model Jacobian is inverted.

    p_ref_F - p_tip_F
        -> Cartesian PID in the EM field frame
        -> transform PID correction to the calibrated local robot frame
        -> direct heuristic motor-space increments
        -> q_cmd = q_prev + dq_pid

Experiment B: ik_jacobian_em_pid_closed_loop
    The reference point is transformed to the local robot/PCC base frame.
    PCC IK provides feedforward motor command.
    EM feedback provides a Cartesian PID residual correction.
    The analytical PCC/IK model Jacobian maps this residual correction to motor space.

    p_ref_F
        -> EM-to-base transform
        -> PCC IK
        -> q_ff

    p_ref_F - p_tip_F
        -> Cartesian PID
        -> analytical PCC/IK Jacobian inverse
        -> dq_fb

    Final:
        q_cmd = q_ff + dq_fb

Notes:
- The EM sensor must be fixed near the probe tip.
- Both experiments use the EM sensor for feedback/logging.
- Experiment A deliberately does not use IK or any Jacobian during the tracking loop.
- Experiment B uses the simple PCC analytical Jacobian, not the measured dynamic Jacobian.
- The PCC IK/Jacobian is a simple constant-curvature approximation, not a validated model.
- Start with a very small circle or square.
"""

import csv
import math
import time
from pathlib import Path

import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupBulkWrite,
    PacketHandler,
    PortHandler,
    DXL_LOBYTE,
    DXL_HIBYTE,
    DXL_LOWORD,
    DXL_HIWORD,
)

try:
    from sksurgerynditracker.nditracker import NDITracker
except Exception:
    NDITracker = None


# ============================================================
# USER CONFIG
# ============================================================

DXL_PORT = "COM10"
DXL_BAUD = 1_000_000
DXL_PROTOCOL = 2.0

# Set this to the Aurora SCU serial port, for example "COM7".
# If None, scikit-surgerynditracker will try the ports listed below.
AURORA_SERIAL_PORT = None
AURORA_PORTS_TO_PROBE = 20
AURORA_TOOL_INDEX = 0

LOG_CSV_PATH = Path(__file__).with_name("em_trajectory_tracking_pid_baseline_vs_ik_jacobian_log.csv")
SUMMARY_CSV_PATH = Path(__file__).with_name("em_trajectory_tracking_pid_baseline_vs_ik_jacobian_summary.csv")

RUN_EXPERIMENTS = [
    "em_pid_no_jacobian_baseline",
    "ik_jacobian_em_pid_closed_loop",
]

TRAJ_SHAPE = "circle"       # "circle" or "square"
TRAJECTORY_DURATION_S = 30.0
SAMPLE_HZ = 20.0
NUM_CYCLES = 1

# Keep the first trials small.
EM_TRAJ_RADIUS_MM = 0.25
EM_TRAJ_SIDE_MM = 0.50

PAUSE_BETWEEN_STEPS = True


# ============================================================
# DYNAMIXEL IDS AND CONTROL TABLE
# ============================================================

DXL_TRANS = 8
DXL_ROT = 12
DXL_BEND = 10
DXL_IDS = [DXL_TRANS, DXL_ROT, DXL_BEND]

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

TORQUE_ON = 1
TORQUE_OFF = 0
LEN_4 = 4

DECIMAL_PER_REV = 4096
DECIMAL_PER_DEG = DECIMAL_PER_REV / 360.0

EXT_MIN = -1_048_575
EXT_MAX = 1_048_575
POS_MIN = 0
POS_MAX = 4095


# ============================================================
# HARDWARE CALIBRATION
# ============================================================

SCREW_LEAD_MM_PER_REV = 1.027210

Z_HOME_MM = 2.1
Z_MAX_MM = 13.1
Z_SAFE_MIN_MM = 2.1
Z_SAFE_MAX_MM = 10.0

R0_DECIMAL = 2048
ROT_MIN_DEG = -179.0
ROT_MAX_DEG = 179.0
ROT_SAFE_MIN_DEG = -120.0
ROT_SAFE_MAX_DEG = 120.0

BEND_HOME_DECIMAL = 1515
BEND_NEG_ONSET_DECIMAL = 1435
BEND_POS_ONSET_DECIMAL = 1595
BEND_NEG_45_DECIMAL = 1181
BEND_POS_45_DECIMAL = 1805

PROBE_BEND_MIN_DEG = -45.0
PROBE_BEND_MAX_DEG = 45.0
BEND_SAFE_MIN_DEG = -30.0
BEND_SAFE_MAX_DEG = 30.0
BEND_ZERO_EPS_DEG = 0.5

# Conservative motion profiles.
TRANS_PROFILE_VEL_RAW = 30
TRANS_PROFILE_ACCEL_RAW = 10
ROT_PROFILE_VEL_RAW = 22
ROT_PROFILE_ACCEL_RAW = 10
BEND_PROFILE_VEL_RAW = 13
BEND_PROFILE_ACCEL_RAW = 6

DECIMAL_TOL_TRANS = 20
DECIMAL_TOL_ROT = 10
DECIMAL_TOL_BEND = 10
MOVE_TIMEOUT_S = 20.0


# ============================================================
# SIMPLE PCC MODEL AND SAFE CENTER
# ============================================================

L_CC_MM = 6.0
IK_BEND_LIMIT_DEG = 28.0

# The center is intentionally offset from the centerline to avoid rotation jumps.
CENTER_X_B_MM = 1.25
CENTER_Y_B_MM = 0.00
CENTER_TIP_Z_B_MM = 12.5

# EM frame calibration moves around the safe center.
CALIB_X_STEP_MM = 0.20
CALIB_Y_STEP_MM = 0.20
CALIB_Z_STEP_MM = 0.40

# Local Jacobian identification steps.
JAC_Z_STEP_MM = 0.20
JAC_ROT_STEP_DEG = 2.0
JAC_BEND_STEP_DEG = 2.0
JAC_DAMPING = 0.02


# ============================================================
# PID AND SAFETY LIMITS
# ============================================================

CART_PID_KP = np.array([0.20, 0.20, 0.15], dtype=float)
CART_PID_KI = np.array([0.00, 0.00, 0.00], dtype=float)
CART_PID_KD = np.array([0.01, 0.01, 0.005], dtype=float)
CART_PID_INTEGRAL_LIMIT_MM_S = np.array([0.5, 0.5, 0.5], dtype=float)
CART_PID_OUTPUT_LIMIT_MM = np.array([0.25, 0.25, 0.20], dtype=float)

# Per-sample command limits.
MAX_DQ_PER_SAMPLE = {
    "z_mm": 0.04,
    "rot_deg": 0.8,
    "bend_deg": 0.8,
}

MAX_EM_ERROR_MM = 5.0
EM_AVERAGE_DURATION_S = 0.7
EM_AVERAGE_HZ = 40.0


# ============================================================
# NO-JACOBIAN BASELINE AND IK-JACOBIAN CONFIG
# ============================================================

# Experiment A maps the EM PID correction directly to actuator increments.
# These gains are intentionally conservative because no Jacobian is used.
# Input is a PID Cartesian correction in the calibrated local frame, unit mm.
NOJAC_PID_Z_MM_PER_MM = 1.00
NOJAC_PID_ROT_DEG_PER_MM = 16.0
NOJAC_PID_BEND_DEG_PER_MM = 14.0

# Experiment B uses a scaled analytical PCC/IK Jacobian inverse.
# Raw q = [z_mm, rot_deg, bend_deg]
# Scaled dq = raw dq / IK_JAC_Q_SCALE
IK_JAC_Q_SCALE = np.array([
    JAC_Z_STEP_MM,
    JAC_ROT_STEP_DEG,
    JAC_BEND_STEP_DEG,
], dtype=float)
IK_JAC_DAMPING = 0.02
JACOBIAN_SINGULAR_VALUE_EPS = 1e-12


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def u32_to_i32(u):
    return u - (1 << 32) if u >= (1 << 31) else u


def goal_bytes(pos):
    pos = int(pos)
    return [
        DXL_LOBYTE(DXL_LOWORD(pos)),
        DXL_HIBYTE(DXL_LOWORD(pos)),
        DXL_LOBYTE(DXL_HIWORD(pos)),
        DXL_HIBYTE(DXL_HIWORD(pos)),
    ]


def decimal_per_mm_from_lead(lead_mm_per_rev):
    if lead_mm_per_rev <= 0:
        raise ValueError("SCREW_LEAD_MM_PER_REV must be positive")
    return DECIMAL_PER_REV / lead_mm_per_rev


def deg_to_decimal(deg):
    return int(round(deg * DECIMAL_PER_DEG))


def decimal_to_deg(decimal_pos):
    return decimal_pos / DECIMAL_PER_DEG


def wrap_to_signed_180(angle_deg):
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and angle_deg > 0:
        return 180.0
    return wrapped


def safety_clamp_q(q):
    return {
        "z_mm": clamp(q["z_mm"], Z_SAFE_MIN_MM, Z_SAFE_MAX_MM),
        "rot_deg": clamp(wrap_to_signed_180(q["rot_deg"]), ROT_SAFE_MIN_DEG, ROT_SAFE_MAX_DEG),
        "bend_deg": clamp(q["bend_deg"], BEND_SAFE_MIN_DEG, BEND_SAFE_MAX_DEG),
    }


def vec_norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def normalize(v, name="vector"):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise RuntimeError(f"Cannot normalize near-zero {name}")
    return v / n


def rate_limit_q(q_des, q_prev):
    q_out = dict(q_des)
    limited = False

    for key, max_step in MAX_DQ_PER_SAMPLE.items():
        delta = q_out[key] - q_prev[key]
        if key == "rot_deg":
            delta = wrap_to_signed_180(q_out[key] - q_prev[key])
        if abs(delta) > max_step:
            q_out[key] = q_prev[key] + clamp(delta, -max_step, max_step)
            limited = True

    return safety_clamp_q(q_out), limited


def q_dict_to_vec(q):
    return np.array([
        q["z_mm"],
        q["rot_deg"],
        q["bend_deg"],
    ], dtype=float)


def q_vec_delta(q_now, q_prev):
    dq = q_dict_to_vec(q_now) - q_dict_to_vec(q_prev)
    dq[1] = wrap_to_signed_180(dq[1])
    return dq


def dq_vec_to_dict_add(q_base, dq):
    return {
        "z_mm": q_base["z_mm"] + float(dq[0]),
        "rot_deg": q_base["rot_deg"] + float(dq[1]),
        "bend_deg": q_base["bend_deg"] + float(dq[2]),
    }


# ============================================================
# DYNAMIXEL HELPERS
# ============================================================

def write1(ph, port, dxl_id, addr, val):
    res, err = ph.write1ByteTxRx(port, dxl_id, addr, int(val))
    if res != COMM_SUCCESS:
        raise RuntimeError(f"[ID:{dxl_id}] write1 failed: {ph.getTxRxResult(res)}")
    if err != 0:
        raise RuntimeError(f"[ID:{dxl_id}] write1 packet error: {ph.getRxPacketError(err)}")


def write4(ph, port, dxl_id, addr, val):
    res, err = ph.write4ByteTxRx(port, dxl_id, addr, int(val))
    if res != COMM_SUCCESS:
        raise RuntimeError(f"[ID:{dxl_id}] write4 failed: {ph.getTxRxResult(res)}")
    if err != 0:
        raise RuntimeError(f"[ID:{dxl_id}] write4 packet error: {ph.getRxPacketError(err)}")


def read_present_pos(ph, port, dxl_id):
    u32, res, err = ph.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
    if res != COMM_SUCCESS:
        raise RuntimeError(f"[ID:{dxl_id}] read position failed: {ph.getTxRxResult(res)}")
    if err != 0:
        raise RuntimeError(f"[ID:{dxl_id}] read position packet error: {ph.getRxPacketError(err)}")
    return u32_to_i32(u32)


def set_operating_mode(ph, port, dxl_id, mode_val):
    write1(ph, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
    time.sleep(0.05)
    write1(ph, port, dxl_id, ADDR_OPERATING_MODE, mode_val)
    time.sleep(0.05)


def set_profile(ph, port, dxl_id, accel_raw, vel_raw):
    write1(ph, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
    time.sleep(0.02)
    write4(ph, port, dxl_id, ADDR_PROFILE_ACCELERATION, accel_raw)
    write4(ph, port, dxl_id, ADDR_PROFILE_VELOCITY, vel_raw)
    time.sleep(0.02)


def send_goals(gbw, ph, goals):
    for dxl_id, goal in goals.items():
        ok = gbw.addParam(dxl_id, ADDR_GOAL_POSITION, LEN_4, goal_bytes(int(goal)))
        if not ok:
            raise RuntimeError(f"BulkWrite addParam failed for ID {dxl_id}")

    res = gbw.txPacket()
    gbw.clearParam()

    if res != COMM_SUCCESS:
        raise RuntimeError(f"BulkWrite txPacket failed: {ph.getTxRxResult(res)}")


# ============================================================
# BEND CALIBRATION
# ============================================================

def probe_deg_to_bend_decimal(probe_deg):
    probe_deg = clamp(probe_deg, PROBE_BEND_MIN_DEG, PROBE_BEND_MAX_DEG)

    if abs(probe_deg) < BEND_ZERO_EPS_DEG:
        return BEND_HOME_DECIMAL

    if probe_deg > 0:
        dec = BEND_POS_ONSET_DECIMAL + probe_deg * (
            BEND_POS_45_DECIMAL - BEND_POS_ONSET_DECIMAL
        ) / 45.0
    else:
        dec = BEND_NEG_ONSET_DECIMAL + probe_deg * (
            BEND_NEG_ONSET_DECIMAL - BEND_NEG_45_DECIMAL
        ) / 45.0

    return int(round(clamp(dec, POS_MIN, POS_MAX)))


def bend_decimal_to_probe_deg_est(decimal):
    if BEND_NEG_ONSET_DECIMAL <= decimal <= BEND_POS_ONSET_DECIMAL:
        return 0.0

    if decimal > BEND_POS_ONSET_DECIMAL:
        return 45.0 * (decimal - BEND_POS_ONSET_DECIMAL) / (
            BEND_POS_45_DECIMAL - BEND_POS_ONSET_DECIMAL
        )

    return -45.0 * (BEND_NEG_ONSET_DECIMAL - decimal) / (
        BEND_NEG_ONSET_DECIMAL - BEND_NEG_45_DECIMAL
    )


# ============================================================
# SIMPLE PCC FK / IK
# ============================================================

def pcc_lateral_mm(theta_deg):
    theta = math.radians(abs(theta_deg))
    if theta < 1e-8:
        return 0.0
    return L_CC_MM * (1.0 - math.cos(theta)) / theta


def pcc_axial_mm(theta_deg):
    theta = math.radians(abs(theta_deg))
    if theta < 1e-8:
        return L_CC_MM
    return L_CC_MM * math.sin(theta) / theta


def theta_deg_from_lateral(rho_mm, theta_max_deg):
    rho_mm = max(0.0, float(rho_mm))
    rho_max = pcc_lateral_mm(theta_max_deg)
    reachable = True

    if rho_mm > rho_max:
        rho_mm = rho_max
        reachable = False

    if rho_mm < 1e-6:
        return 0.0, reachable

    lo = 0.0
    hi = math.radians(theta_max_deg)

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        rho_mid = L_CC_MM * (1.0 - math.cos(mid)) / mid if mid > 1e-9 else 0.0
        if rho_mid < rho_mm:
            lo = mid
        else:
            hi = mid

    return math.degrees(0.5 * (lo + hi)), reachable


def simple_pcc_ik(x_mm, y_mm, tip_z_mm):
    rho = math.hypot(x_mm, y_mm)
    bend_deg, reachable = theta_deg_from_lateral(rho, IK_BEND_LIMIT_DEG)
    rot_deg = wrap_to_signed_180(math.degrees(math.atan2(y_mm, x_mm)))
    z_mm = tip_z_mm - pcc_axial_mm(bend_deg)

    q = safety_clamp_q({
        "z_mm": z_mm,
        "rot_deg": rot_deg,
        "bend_deg": bend_deg,
    })

    return q, reachable


# ============================================================
# AURORA TRACKER
# ============================================================

class AuroraTracker:
    """Minimal wrapper around scikit-surgerynditracker for NDI Aurora."""

    def __init__(self, serial_port=None, tool_index=0):
        if NDITracker is None:
            raise RuntimeError(
                "scikit-surgerynditracker is not installed. Run: pip install scikit-surgerynditracker"
            )

        settings = {
            "tracker type": "aurora",
            "verbose": False,
        }
        if serial_port:
            settings["serial port"] = serial_port
        else:
            settings["ports to probe"] = AURORA_PORTS_TO_PROBE

        self.tool_index = int(tool_index)
        self.tracker = NDITracker(settings)
        self.started = False

    def start(self):
        self.tracker.start_tracking()
        self.started = True

    def stop(self):
        if self.started:
            try:
                self.tracker.stop_tracking()
            except Exception:
                pass
        try:
            self.tracker.close()
        except Exception:
            pass

    def read_position_F(self, timeout_s=0.2):
        """Return the selected tool position in the EM field frame, in mm."""
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            try:
                port_handles, timestamps, frame_numbers, tracking, quality = self.tracker.get_frame()
            except Exception:
                time.sleep(0.01)
                continue

            if tracking is None or len(tracking) <= self.tool_index:
                time.sleep(0.01)
                continue

            mat = np.asarray(tracking[self.tool_index], dtype=float)
            if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
                time.sleep(0.01)
                continue

            p = mat[:3, 3].astype(float)
            if not np.all(np.isfinite(p)):
                time.sleep(0.01)
                continue

            q = None
            if quality is not None and len(quality) > self.tool_index:
                q = quality[self.tool_index]

            return {
                "valid": True,
                "timestamp": time.time(),
                "p_F": p,
                "quality": q,
            }

        return {
            "valid": False,
            "timestamp": time.time(),
            "p_F": None,
            "quality": None,
        }


def average_em_position_F(em, duration_s=EM_AVERAGE_DURATION_S, sample_hz=EM_AVERAGE_HZ):
    pts = []
    t0 = time.time()

    while time.time() - t0 < duration_s:
        sample = em.read_position_F(timeout_s=0.1)
        if sample["valid"]:
            pts.append(sample["p_F"])
        time.sleep(1.0 / sample_hz)

    if len(pts) < 5:
        raise RuntimeError("Not enough valid EM samples")

    return np.mean(np.vstack(pts), axis=0)


# ============================================================
# PID
# ============================================================

class CartesianPID:
    """PID in the EM field frame."""

    def __init__(self):
        self.integral = np.zeros(3)
        self.prev_error = None

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error = None

    def update(self, p_ref_F, p_tip_F, dt_s):
        if dt_s <= 1e-6:
            dt_s = 1e-6

        e = np.asarray(p_ref_F, dtype=float) - np.asarray(p_tip_F, dtype=float)
        self.integral += e * dt_s
        self.integral = np.clip(
            self.integral,
            -CART_PID_INTEGRAL_LIMIT_MM_S,
            CART_PID_INTEGRAL_LIMIT_MM_S,
        )

        if self.prev_error is None:
            de = np.zeros(3)
        else:
            de = (e - self.prev_error) / dt_s

        u = CART_PID_KP * e + CART_PID_KI * self.integral + CART_PID_KD * de
        u_clamped = np.clip(u, -CART_PID_OUTPUT_LIMIT_MM, CART_PID_OUTPUT_LIMIT_MM)
        saturated = bool(np.any(np.abs(u - u_clamped) > 1e-12))

        self.prev_error = e.copy()
        return u_clamped, e, saturated


# ============================================================
# MAIN ROBOT CLASS
# ============================================================

class Robot:
    """Small wrapper for the three Dynamixel motors."""

    def __init__(self):
        self.port = PortHandler(DXL_PORT)
        self.ph = PacketHandler(DXL_PROTOCOL)
        self.gbw = GroupBulkWrite(self.port, self.ph)
        self.z_ref_decimal = None
        self.dpm = decimal_per_mm_from_lead(SCREW_LEAD_MM_PER_REV)

    def open(self):
        if not self.port.openPort():
            raise RuntimeError("Failed to open Dynamixel port")
        if not self.port.setBaudRate(DXL_BAUD):
            raise RuntimeError("Failed to set Dynamixel baudrate")

        set_operating_mode(self.ph, self.port, DXL_TRANS, 4)  # Extended Position mode
        set_operating_mode(self.ph, self.port, DXL_ROT, 3)    # Position mode
        set_operating_mode(self.ph, self.port, DXL_BEND, 3)   # Position mode

        set_profile(self.ph, self.port, DXL_TRANS, TRANS_PROFILE_ACCEL_RAW, TRANS_PROFILE_VEL_RAW)
        set_profile(self.ph, self.port, DXL_ROT, ROT_PROFILE_ACCEL_RAW, ROT_PROFILE_VEL_RAW)
        set_profile(self.ph, self.port, DXL_BEND, BEND_PROFILE_ACCEL_RAW, BEND_PROFILE_VEL_RAW)

        for dxl_id in DXL_IDS:
            write1(self.ph, self.port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

        time.sleep(0.5)
        self.z_ref_decimal = read_present_pos(self.ph, self.port, DXL_TRANS)

        print(f"z_ref_decimal = {self.z_ref_decimal} at Z_HOME_MM = {Z_HOME_MM:.3f} mm")
        print(f"translation decimals/mm = {self.dpm:.3f}")

    def close(self):
        try:
            for dxl_id in DXL_IDS:
                write1(self.ph, self.port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        except Exception:
            pass
        try:
            self.port.closePort()
        except Exception:
            pass

    def z_mm_to_ticks(self, z_mm):
        z_mm = clamp(z_mm, Z_HOME_MM, Z_MAX_MM)
        goal = self.z_ref_decimal + int(round((z_mm - Z_HOME_MM) * self.dpm))
        if goal < EXT_MIN or goal > EXT_MAX:
            raise RuntimeError(f"Translation goal {goal} outside extended range")
        return int(goal)

    def ticks_to_z_mm(self, ticks):
        return Z_HOME_MM + (ticks - self.z_ref_decimal) / self.dpm

    def rot_deg_to_ticks(self, rot_deg):
        rot_deg = clamp(wrap_to_signed_180(rot_deg), ROT_MIN_DEG, ROT_MAX_DEG)
        goal = R0_DECIMAL + deg_to_decimal(rot_deg)
        if goal < POS_MIN or goal > POS_MAX:
            raise RuntimeError(f"Rotation goal {goal} outside position range")
        return int(goal)

    def ticks_to_rot_deg(self, ticks):
        return decimal_to_deg(ticks - R0_DECIMAL)

    def q_to_goals(self, q):
        q = safety_clamp_q(q)
        return {
            DXL_TRANS: self.z_mm_to_ticks(q["z_mm"]),
            DXL_ROT: self.rot_deg_to_ticks(q["rot_deg"]),
            DXL_BEND: probe_deg_to_bend_decimal(q["bend_deg"]),
        }

    def read_q(self):
        z_ticks = read_present_pos(self.ph, self.port, DXL_TRANS)
        r_ticks = read_present_pos(self.ph, self.port, DXL_ROT)
        b_ticks = read_present_pos(self.ph, self.port, DXL_BEND)
        return {
            "z_mm": self.ticks_to_z_mm(z_ticks),
            "rot_deg": self.ticks_to_rot_deg(r_ticks),
            "bend_deg": bend_decimal_to_probe_deg_est(b_ticks),
            "z_ticks": z_ticks,
            "rot_ticks": r_ticks,
            "bend_ticks": b_ticks,
        }

    def send_q(self, q):
        goals = self.q_to_goals(q)
        send_goals(self.gbw, self.ph, goals)
        return goals

    def move_to_q_wait(self, q, label="move"):
        q = safety_clamp_q(q)
        goals = self.send_q(q)
        t0 = time.time()

        while time.time() - t0 < MOVE_TIMEOUT_S:
            state = self.read_q()
            ok = (
                abs(goals[DXL_TRANS] - state["z_ticks"]) <= DECIMAL_TOL_TRANS
                and abs(goals[DXL_ROT] - state["rot_ticks"]) <= DECIMAL_TOL_ROT
                and abs(goals[DXL_BEND] - state["bend_ticks"]) <= DECIMAL_TOL_BEND
            )
            if ok:
                return True
            time.sleep(0.03)

        print(f"[WARN] Timeout while waiting for {label}")
        return False

    def return_home(self):
        print("Returning home...")
        state = self.read_q()
        self.move_to_q_wait({"z_mm": state["z_mm"], "rot_deg": state["rot_deg"], "bend_deg": 0.0}, "bend home")
        state = self.read_q()
        self.move_to_q_wait({"z_mm": state["z_mm"], "rot_deg": 0.0, "bend_deg": 0.0}, "rotation home")
        self.move_to_q_wait({"z_mm": Z_HOME_MM, "rot_deg": 0.0, "bend_deg": 0.0}, "z home")


# ============================================================
# EM FRAME CALIBRATION
# ============================================================

def p_F_to_B(p_F, calib):
    """Convert EM field-frame point to the local robot/PCC base frame."""
    v = np.asarray(p_F, dtype=float) - calib["center_F"]
    return {
        "x_mm": calib["center_B"][0] + float(np.dot(v, calib["ex_F"])),
        "y_mm": calib["center_B"][1] + float(np.dot(v, calib["ey_F"])),
        "tip_z_mm": calib["center_B"][2] + float(np.dot(v, calib["ez_F"])),
    }


def make_p_F(center_F, ex_F, ey_F, ez_F, x_mm=0.0, y_mm=0.0, z_mm=0.0):
    return center_F + x_mm * ex_F + y_mm * ey_F + z_mm * ez_F


def move_to_B(robot, x_mm, y_mm, tip_z_mm, label):
    q, reachable = simple_pcc_ik(x_mm, y_mm, tip_z_mm)
    if not reachable:
        print(f"[WARN] IK target was clamped for {label}")
    robot.move_to_q_wait(q, label)
    time.sleep(0.3)
    return q


def calibrate_em_frame(robot, em):
    """
    Define the trajectory frame inside the EM field frame.

    The EM trajectory center is the measured sensor position at a safe PCC center.
    ex_F, ey_F, ez_F are robot-like axes expressed in the EM field frame.
    """
    center_B = np.array([CENTER_X_B_MM, CENTER_Y_B_MM, CENTER_TIP_Z_B_MM], dtype=float)

    print("\nEM frame calibration:")
    q_center = move_to_B(robot, center_B[0], center_B[1], center_B[2], "center pose")
    center_F = average_em_position_F(em)
    print(f"  center_F = {center_F}")

    move_to_B(robot, center_B[0] + CALIB_X_STEP_MM, center_B[1], center_B[2], "+x calibration")
    px_F = average_em_position_F(em)

    move_to_B(robot, center_B[0], center_B[1] + CALIB_Y_STEP_MM, center_B[2], "+y calibration")
    py_F = average_em_position_F(em)

    move_to_B(robot, center_B[0], center_B[1], center_B[2] + CALIB_Z_STEP_MM, "+z calibration")
    pz_F = average_em_position_F(em)

    robot.move_to_q_wait(q_center, "back to center")
    time.sleep(0.3)

    ex_F = normalize(px_F - center_F, "ex_F")
    ez_raw = normalize(pz_F - center_F, "ez_raw")
    ey_F = normalize(np.cross(ez_raw, ex_F), "ey_F")

    # Choose the sign of ey_F to match the measured +y move.
    if np.dot(ey_F, py_F - center_F) < 0:
        ey_F = -ey_F

    # Recompute ez_F to make the axes right-handed.
    ez_F = normalize(np.cross(ex_F, ey_F), "ez_F")
    if np.dot(ez_F, pz_F - center_F) < 0:
        ez_F = -ez_F
        ey_F = -ey_F

    print(f"  ex_F = {ex_F}")
    print(f"  ey_F = {ey_F}")
    print(f"  ez_F = {ez_F}")

    return {
        "center_B": center_B,
        "center_F": center_F,
        "ex_F": ex_F,
        "ey_F": ey_F,
        "ez_F": ez_F,
        "q_center": q_center,
    }


# ============================================================
# TRAJECTORY GENERATOR
# ============================================================

def em_trajectory_ref(t_s, total_time_s, calib):
    phase = (t_s / total_time_s * NUM_CYCLES) % 1.0
    center_F = calib["center_F"]
    ex_F = calib["ex_F"]
    ey_F = calib["ey_F"]
    ez_F = calib["ez_F"]

    if TRAJ_SHAPE.lower() == "circle":
        angle = 2.0 * math.pi * phase
        a = EM_TRAJ_RADIUS_MM * math.cos(angle)
        b = EM_TRAJ_RADIUS_MM * math.sin(angle)
    elif TRAJ_SHAPE.lower() == "square":
        side = EM_TRAJ_SIDE_MM
        h = side / 2.0
        s = 4.0 * phase
        if s < 1.0:
            a = -h + side * s
            b = -h
        elif s < 2.0:
            a = h
            b = -h + side * (s - 1.0)
        elif s < 3.0:
            a = h - side * (s - 2.0)
            b = h
        else:
            a = -h
            b = h - side * (s - 3.0)
    else:
        raise RuntimeError("TRAJ_SHAPE must be 'circle' or 'square'")

    p_ref_F = make_p_F(center_F, ex_F, ey_F, ez_F, a, b, 0.0)
    return p_ref_F, phase, a, b


# ============================================================
# MODEL JACOBIAN AND CONTROLLER HELPERS
# ============================================================

def delta_F_to_B(v_F, calib):
    """Resolve an EM-frame displacement vector in the calibrated local robot frame."""
    v_F = np.asarray(v_F, dtype=float)
    return np.array([
        float(np.dot(v_F, calib["ex_F"])),
        float(np.dot(v_F, calib["ey_F"])),
        float(np.dot(v_F, calib["ez_F"])),
    ], dtype=float)


def rotation_B_to_F(calib):
    """Columns are local robot-frame unit axes expressed in the EM field frame."""
    return np.column_stack([
        np.asarray(calib["ex_F"], dtype=float),
        np.asarray(calib["ey_F"], dtype=float),
        np.asarray(calib["ez_F"], dtype=float),
    ])


def signed_pcc_terms(bend_deg):
    """
    Signed PCC lateral/axial terms and derivatives with respect to bend angle in radians.

    Model:
        rho(theta) = L * (1 - cos(theta)) / theta
        zeta(theta) = L * sin(theta) / theta

    rho is signed, so negative bend gives lateral motion in the opposite bending direction.
    """
    theta = math.radians(float(bend_deg))

    if abs(theta) < 1e-6:
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2

        rho = L_CC_MM * (0.5 * theta - theta3 / 24.0)
        axial = L_CC_MM * (1.0 - theta2 / 6.0 + theta4 / 120.0)
        drho_dtheta = L_CC_MM * (0.5 - theta2 / 8.0 + theta4 / 144.0)
        daxial_dtheta = L_CC_MM * (-theta / 3.0 + theta3 / 30.0)
        return rho, axial, drho_dtheta, daxial_dtheta

    rho = L_CC_MM * (1.0 - math.cos(theta)) / theta
    axial = L_CC_MM * math.sin(theta) / theta
    drho_dtheta = L_CC_MM * (theta * math.sin(theta) - (1.0 - math.cos(theta))) / (theta * theta)
    daxial_dtheta = L_CC_MM * (theta * math.cos(theta) - math.sin(theta)) / (theta * theta)
    return rho, axial, drho_dtheta, daxial_dtheta


def simple_pcc_fk_B(q):
    """Analytical PCC tip position in the local robot/PCC base frame."""
    q = safety_clamp_q(q)
    phi = math.radians(wrap_to_signed_180(q["rot_deg"]))
    rho, axial, _, _ = signed_pcc_terms(q["bend_deg"])

    return np.array([
        rho * math.cos(phi),
        rho * math.sin(phi),
        q["z_mm"] + axial,
    ], dtype=float)


def simple_pcc_jacobian_B(q):
    """
    Analytical PCC Jacobian in the local robot/PCC base frame.

    Columns match raw motor coordinates:
        [z_mm, rot_deg, bend_deg]
    Rows are local robot/PCC frame displacement:
        [x_B_mm, y_B_mm, tip_z_B_mm]
    """
    q = safety_clamp_q(q)
    phi = math.radians(wrap_to_signed_180(q["rot_deg"]))
    rho, _, drho_dtheta, daxial_dtheta = signed_pcc_terms(q["bend_deg"])

    c = math.cos(phi)
    s = math.sin(phi)
    deg_to_rad = math.pi / 180.0

    col_z = np.array([0.0, 0.0, 1.0], dtype=float)
    col_rot = np.array([
        -rho * s * deg_to_rad,
        rho * c * deg_to_rad,
        0.0,
    ], dtype=float)
    col_bend = np.array([
        drho_dtheta * c * deg_to_rad,
        drho_dtheta * s * deg_to_rad,
        daxial_dtheta * deg_to_rad,
    ], dtype=float)

    return np.column_stack([col_z, col_rot, col_bend])


def simple_pcc_jacobian_F(q, calib):
    """Analytical PCC Jacobian expressed in the EM field frame."""
    return rotation_B_to_F(calib) @ simple_pcc_jacobian_B(q)


def jacobian_correction_to_dq(J_raw, u_F, q_scale=IK_JAC_Q_SCALE, damping=IK_JAC_DAMPING):
    """
    Convert an EM-frame Cartesian correction to raw motor-coordinate correction.

    Uses scaled coordinates to avoid treating 1 mm and 1 degree as numerically equivalent.
    """
    J_raw = np.asarray(J_raw, dtype=float)
    u_F = np.asarray(u_F, dtype=float)
    q_scale = np.asarray(q_scale, dtype=float)

    J_scaled = J_raw @ np.diag(q_scale)
    J_pinv_scaled = J_scaled.T @ np.linalg.inv(
        J_scaled @ J_scaled.T + (damping ** 2) * np.eye(3)
    )
    ds = J_pinv_scaled @ u_F
    dq_raw = q_scale * ds
    return dq_raw


def jacobian_stats_scaled(J_raw, q_scale=IK_JAC_Q_SCALE):
    J_raw = np.asarray(J_raw, dtype=float)
    q_scale = np.asarray(q_scale, dtype=float)
    J_scaled = J_raw @ np.diag(q_scale)

    try:
        svals = np.linalg.svd(J_scaled, compute_uv=False)
        if len(svals) == 3 and svals[-1] > JACOBIAN_SINGULAR_VALUE_EPS:
            cond = float(svals[0] / svals[-1])
        else:
            cond = float("inf")
        return cond, svals
    except Exception:
        return float("nan"), np.array([np.nan, np.nan, np.nan], dtype=float)


def no_jacobian_pid_correction_to_dq(u_F, q_prev, calib):
    """
    Experiment-A direct PID-to-motor mapping with no Jacobian inverse.

    The PID output is first resolved in the calibrated local frame. Lateral correction is
    split into radial/tangential components relative to the current commanded rotation:
        radial      -> bend increment
        tangential  -> rotation increment
        local z     -> insertion increment
    """
    u_B = delta_F_to_B(u_F, calib)
    phi = math.radians(wrap_to_signed_180(q_prev["rot_deg"]))

    radial_mm = u_B[0] * math.cos(phi) + u_B[1] * math.sin(phi)
    tangential_mm = -u_B[0] * math.sin(phi) + u_B[1] * math.cos(phi)

    return np.array([
        NOJAC_PID_Z_MM_PER_MM * u_B[2],
        NOJAC_PID_ROT_DEG_PER_MM * tangential_mm,
        NOJAC_PID_BEND_DEG_PER_MM * radial_mm,
    ], dtype=float)


# ============================================================
# CONTROLLERS
# ============================================================

def ik_initial_q(p_ref_F, calib):
    p_B = p_F_to_B(p_ref_F, calib)
    q, reachable = simple_pcc_ik(p_B["x_mm"], p_B["y_mm"], p_B["tip_z_mm"])
    return q, reachable


def em_pid_no_jacobian_baseline_update(
    pid,
    p_ref_F,
    p_tip_F,
    dt_s,
    q_prev,
    calib,
):
    """
    Experiment A:
        EM feedback PID baseline with no IK feedforward and no Jacobian inverse.

        q_cmd = q_prev + direct_map(PID(p_ref_F - p_tip_F))
    """
    u_F, e_F, sat = pid.update(p_ref_F, p_tip_F, dt_s)
    dq_pid = no_jacobian_pid_correction_to_dq(u_F, q_prev, calib)

    q_des = dq_vec_to_dict_add(q_prev, dq_pid)
    q_cmd, rate_limited = rate_limit_q(
        safety_clamp_q(q_des),
        q_prev,
    )

    return q_cmd, {
        "u_F": u_F,
        "e_F": e_F,
        "pid_saturated": sat,
        "rate_limited": rate_limited,
        "ik_reachable_cmd": True,
        "ik_reachable_ff": True,
        "dq": dq_pid,
        "q_ff": {"z_mm": np.nan, "rot_deg": np.nan, "bend_deg": np.nan},
        "jacobian_mode": "none",
        "jacobian_used": False,
        "jacobian_cond_scaled": np.nan,
        "jacobian_svals_scaled": np.array([np.nan, np.nan, np.nan], dtype=float),
        "J_raw": np.full((3, 3), np.nan, dtype=float),
    }


def ik_jacobian_em_pid_closed_loop_update(
    pid,
    p_ref_F,
    p_tip_F,
    dt_s,
    q_prev,
    calib,
):
    """
    Experiment B:
        IK feedforward + EM-PID residual correction through analytical PCC/IK Jacobian.

        q_cmd = q_ff + J_ik(q_ff)^+ PID(error)
    """
    # 1. PCC IK feedforward.
    p_ref_B = p_F_to_B(p_ref_F, calib)
    q_ff, reachable_ff = simple_pcc_ik(
        p_ref_B["x_mm"],
        p_ref_B["y_mm"],
        p_ref_B["tip_z_mm"],
    )

    # 2. EM feedback PID.
    u_F, e_F, sat = pid.update(p_ref_F, p_tip_F, dt_s)

    # 3. Analytical IK/PCC Jacobian residual correction.
    J_raw = simple_pcc_jacobian_F(q_ff, calib)
    dq_fb = jacobian_correction_to_dq(J_raw, u_F)
    jacobian_cond, jacobian_svals = jacobian_stats_scaled(J_raw)

    # 4. Add feedback correction on top of IK feedforward.
    q_des = dq_vec_to_dict_add(q_ff, dq_fb)
    q_cmd, rate_limited = rate_limit_q(
        safety_clamp_q(q_des),
        q_prev,
    )

    return q_cmd, {
        "u_F": u_F,
        "e_F": e_F,
        "pid_saturated": sat,
        "rate_limited": rate_limited,
        "ik_reachable_cmd": bool(reachable_ff),
        "ik_reachable_ff": bool(reachable_ff),
        "dq": dq_fb,
        "q_ff": q_ff,
        "jacobian_mode": "analytical_pcc_ik",
        "jacobian_used": True,
        "jacobian_cond_scaled": jacobian_cond,
        "jacobian_svals_scaled": jacobian_svals,
        "J_raw": J_raw,
    }


# ============================================================
# EXPERIMENT LOOP
# ============================================================

def run_experiment(robot, em, calib, experiment_name, writer):
    print("\n" + "=" * 70)
    print(f"Running experiment: {experiment_name}")
    print("=" * 70)

    p_ref0_F, _, _, _ = em_trajectory_ref(0.0, TRAJECTORY_DURATION_S, calib)

    if experiment_name == "em_pid_no_jacobian_baseline":
        # The baseline starts from the calibrated safe center and then relies on EM PID only.
        q0 = safety_clamp_q(calib["q_center"])
        reachable = True
    elif experiment_name == "ik_jacobian_em_pid_closed_loop":
        q0, reachable = ik_initial_q(p_ref0_F, calib)
        if not reachable:
            print("[WARN] Initial IK target was clamped")
    else:
        raise RuntimeError(f"Unknown experiment: {experiment_name}")

    robot.move_to_q_wait(q0, f"{experiment_name} initial pose")
    time.sleep(1.0)

    pid = CartesianPID()
    pid.reset()

    q_cmd_prev = q0.copy()

    metrics = {
        "errors": [],
        "ex": [],
        "ey": [],
        "ez": [],
        "num_samples": 0,
        "num_invalid_em": 0,
        "num_large_error": 0,
        "num_rate_limited": 0,
        "num_pid_saturated": 0,
        "num_ik_unreachable_cmd": 0,
        "num_jacobian_used_samples": 0,
    }

    t_start = time.monotonic()
    t_prev = t_start
    sample_idx = 0
    dt_nominal = 1.0 / SAMPLE_HZ

    while True:
        now = time.monotonic()
        t_s = now - t_start
        if t_s > TRAJECTORY_DURATION_S:
            break

        dt_s = now - t_prev
        t_prev = now
        if dt_s <= 1e-6 or dt_s > 1.0:
            dt_s = dt_nominal

        p_ref_F, phase, local_a_mm, local_b_mm = em_trajectory_ref(
            t_s,
            TRAJECTORY_DURATION_S,
            calib,
        )

        em_sample = em.read_position_F(timeout_s=0.1)
        if not em_sample["valid"]:
            metrics["num_invalid_em"] += 1
            print("[WARN] Invalid EM sample; holding previous command")
            time.sleep(dt_nominal)
            continue

        p_tip_F = em_sample["p_F"]

        raw_error = np.asarray(p_ref_F) - np.asarray(p_tip_F)
        raw_error_norm = vec_norm(raw_error)

        if raw_error_norm > MAX_EM_ERROR_MM:
            metrics["num_large_error"] += 1
            print(f"[WARN] Large EM error {raw_error_norm:.3f} mm; holding previous command")
            time.sleep(dt_nominal)
            continue

        if experiment_name == "em_pid_no_jacobian_baseline":
            q_cmd, dbg = em_pid_no_jacobian_baseline_update(
                pid,
                p_ref_F,
                p_tip_F,
                dt_s,
                q_cmd_prev,
                calib,
            )
        elif experiment_name == "ik_jacobian_em_pid_closed_loop":
            q_cmd, dbg = ik_jacobian_em_pid_closed_loop_update(
                pid,
                p_ref_F,
                p_tip_F,
                dt_s,
                q_cmd_prev,
                calib,
            )
        else:
            raise RuntimeError(f"Unknown experiment: {experiment_name}")

        goals = robot.send_q(q_cmd)
        q_cmd_prev = q_cmd.copy()

        # Read actual q after sending, for logging.
        q_actual = robot.read_q()

        e_F = dbg["e_F"]
        e_norm = vec_norm(e_F)

        metrics["errors"].append(e_norm)
        metrics["ex"].append(float(e_F[0]))
        metrics["ey"].append(float(e_F[1]))
        metrics["ez"].append(float(e_F[2]))
        metrics["num_samples"] += 1
        metrics["num_rate_limited"] += int(dbg["rate_limited"])
        metrics["num_pid_saturated"] += int(dbg["pid_saturated"])
        metrics["num_ik_unreachable_cmd"] += int(not dbg["ik_reachable_cmd"])
        metrics["num_jacobian_used_samples"] += int(dbg["jacobian_used"])

        q_ff = dbg.get("q_ff")
        if q_ff is None:
            q_ff = {"z_mm": np.nan, "rot_deg": np.nan, "bend_deg": np.nan}

        dq_fb = dbg["dq"]
        J_raw = dbg.get("J_raw", np.full((3, 3), np.nan, dtype=float))
        jacobian_svals = dbg.get(
            "jacobian_svals_scaled",
            np.array([np.nan, np.nan, np.nan], dtype=float),
        )

        writer.writerow({
            "experiment": experiment_name,
            "timestamp_unix": time.time(),
            "t_s": t_s,
            "dt_s": dt_s,
            "traj_shape": TRAJ_SHAPE,
            "phase": phase,
            "local_a_mm": local_a_mm,
            "local_b_mm": local_b_mm,

            "x_ref_F_mm": float(p_ref_F[0]),
            "y_ref_F_mm": float(p_ref_F[1]),
            "z_ref_F_mm": float(p_ref_F[2]),
            "x_tip_F_mm": float(p_tip_F[0]),
            "y_tip_F_mm": float(p_tip_F[1]),
            "z_tip_F_mm": float(p_tip_F[2]),
            "ex_F_mm": float(e_F[0]),
            "ey_F_mm": float(e_F[1]),
            "ez_F_mm": float(e_F[2]),
            "e_norm_F_mm": e_norm,

            "pid_u_x_F_mm": float(dbg["u_F"][0]),
            "pid_u_y_F_mm": float(dbg["u_F"][1]),
            "pid_u_z_F_mm": float(dbg["u_F"][2]),
            "pid_saturated": int(dbg["pid_saturated"]),
            "rate_limited": int(dbg["rate_limited"]),
            "ik_reachable_cmd": int(dbg["ik_reachable_cmd"]),

            "q_ff_z_mm": q_ff["z_mm"],
            "q_ff_rot_deg": q_ff["rot_deg"],
            "q_ff_bend_deg": q_ff["bend_deg"],

            "dq_fb_z_mm": float(dq_fb[0]),
            "dq_fb_rot_deg": float(dq_fb[1]),
            "dq_fb_bend_deg": float(dq_fb[2]),

            "q_cmd_z_mm": q_cmd["z_mm"],
            "q_cmd_rot_deg": q_cmd["rot_deg"],
            "q_cmd_bend_deg": q_cmd["bend_deg"],
            "q_actual_z_mm": q_actual["z_mm"],
            "q_actual_rot_deg": q_actual["rot_deg"],
            "q_actual_bend_deg_est": q_actual["bend_deg"],

            "z_goal_decimal": goals[DXL_TRANS],
            "rot_goal_decimal": goals[DXL_ROT],
            "bend_goal_decimal": goals[DXL_BEND],
            "z_actual_decimal": q_actual["z_ticks"],
            "rot_actual_decimal": q_actual["rot_ticks"],
            "bend_actual_decimal": q_actual["bend_ticks"],

            "jacobian_mode": dbg["jacobian_mode"],
            "jacobian_used": int(dbg["jacobian_used"]),
            "jacobian_cond_scaled": dbg["jacobian_cond_scaled"],
            "jacobian_s1_scaled": float(jacobian_svals[0]),
            "jacobian_s2_scaled": float(jacobian_svals[1]),
            "jacobian_s3_scaled": float(jacobian_svals[2]),

            "J00_raw": float(J_raw[0, 0]),
            "J01_raw": float(J_raw[0, 1]),
            "J02_raw": float(J_raw[0, 2]),
            "J10_raw": float(J_raw[1, 0]),
            "J11_raw": float(J_raw[1, 1]),
            "J12_raw": float(J_raw[1, 2]),
            "J20_raw": float(J_raw[2, 0]),
            "J21_raw": float(J_raw[2, 1]),
            "J22_raw": float(J_raw[2, 2]),

            "em_quality": em_sample["quality"],
        })

        if sample_idx % max(1, int(SAMPLE_HZ / 2.0)) == 0:
            print(
                f"t={t_s:6.2f}s | e={e_norm:6.3f} mm | "
                f"cmd=({q_cmd['z_mm']:.2f}, {q_cmd['rot_deg']:.2f}, {q_cmd['bend_deg']:.2f}) | "
                f"J={dbg['jacobian_mode']}"
            )

        sample_idx += 1
        next_time = t_start + sample_idx * dt_nominal
        sleep_s = next_time - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"Finished: {experiment_name}")
    print(f"Jacobian-used samples: {metrics['num_jacobian_used_samples']}")
    return metrics


def summarize_metrics(experiment_name, metrics):
    err = np.asarray(metrics["errors"], dtype=float)
    ex = np.asarray(metrics["ex"], dtype=float)
    ey = np.asarray(metrics["ey"], dtype=float)
    ez = np.asarray(metrics["ez"], dtype=float)

    if len(err) == 0:
        return {
            "experiment": experiment_name,
            "num_samples": 0,
            "rms_error_mm": np.nan,
            "mean_error_mm": np.nan,
            "max_error_mm": np.nan,
            "rms_ex_mm": np.nan,
            "rms_ey_mm": np.nan,
            "rms_ez_mm": np.nan,
            "num_invalid_em": metrics["num_invalid_em"],
            "num_large_error": metrics["num_large_error"],
            "num_rate_limited": metrics["num_rate_limited"],
            "num_pid_saturated": metrics["num_pid_saturated"],
            "num_ik_unreachable_cmd": metrics["num_ik_unreachable_cmd"],
            "num_jacobian_used_samples": metrics["num_jacobian_used_samples"],
        }

    return {
        "experiment": experiment_name,
        "num_samples": len(err),
        "rms_error_mm": float(np.sqrt(np.mean(err ** 2))),
        "mean_error_mm": float(np.mean(err)),
        "max_error_mm": float(np.max(err)),
        "rms_ex_mm": float(np.sqrt(np.mean(ex ** 2))),
        "rms_ey_mm": float(np.sqrt(np.mean(ey ** 2))),
        "rms_ez_mm": float(np.sqrt(np.mean(ez ** 2))),
        "num_invalid_em": metrics["num_invalid_em"],
        "num_large_error": metrics["num_large_error"],
        "num_rate_limited": metrics["num_rate_limited"],
        "num_pid_saturated": metrics["num_pid_saturated"],
        "num_ik_unreachable_cmd": metrics["num_ik_unreachable_cmd"],
        "num_jacobian_used_samples": metrics["num_jacobian_used_samples"],
    }


# ============================================================
# MAIN
# ============================================================

def maybe_pause(message):
    if PAUSE_BETWEEN_STEPS:
        input(message)


def main():
    print("=" * 70)
    print("EM trajectory tracking: EM PID baseline vs IK Jacobian + EM PID")
    print("=" * 70)
    print(f"Dynamixel port: {DXL_PORT}")
    print(f"Aurora serial port: {AURORA_SERIAL_PORT}")
    print(f"Trajectory: {TRAJ_SHAPE}")
    print(f"Log: {LOG_CSV_PATH}")
    print(f"Summary: {SUMMARY_CSV_PATH}")
    maybe_pause("Check hardware safety, then press Enter to start...")

    robot = Robot()
    em = None

    fieldnames = [
        "experiment", "timestamp_unix", "t_s", "dt_s", "traj_shape", "phase", "local_a_mm", "local_b_mm",

        "x_ref_F_mm", "y_ref_F_mm", "z_ref_F_mm",
        "x_tip_F_mm", "y_tip_F_mm", "z_tip_F_mm",
        "ex_F_mm", "ey_F_mm", "ez_F_mm", "e_norm_F_mm",

        "pid_u_x_F_mm", "pid_u_y_F_mm", "pid_u_z_F_mm",
        "pid_saturated", "rate_limited", "ik_reachable_cmd",

        "q_ff_z_mm", "q_ff_rot_deg", "q_ff_bend_deg",

        "dq_fb_z_mm", "dq_fb_rot_deg", "dq_fb_bend_deg",

        "q_cmd_z_mm", "q_cmd_rot_deg", "q_cmd_bend_deg",
        "q_actual_z_mm", "q_actual_rot_deg", "q_actual_bend_deg_est",

        "z_goal_decimal", "rot_goal_decimal", "bend_goal_decimal",
        "z_actual_decimal", "rot_actual_decimal", "bend_actual_decimal",

        "jacobian_mode", "jacobian_used",
        "jacobian_cond_scaled", "jacobian_s1_scaled", "jacobian_s2_scaled", "jacobian_s3_scaled",

        "J00_raw", "J01_raw", "J02_raw",
        "J10_raw", "J11_raw", "J12_raw",
        "J20_raw", "J21_raw", "J22_raw",

        "em_quality",
    ]

    summary_rows = []

    try:
        robot.open()
        robot.return_home()

        print("\nStarting Aurora tracker...")
        em = AuroraTracker(serial_port=AURORA_SERIAL_PORT, tool_index=AURORA_TOOL_INDEX)
        em.start()
        time.sleep(1.0)

        sample = em.read_position_F(timeout_s=1.0)
        if not sample["valid"]:
            raise RuntimeError("No valid Aurora sample. Check NDI Toolbox, sensor visibility, and serial port.")
        print(f"First EM sample: {sample['p_F']}")

        maybe_pause("Press Enter to run EM frame calibration...")
        calib = calibrate_em_frame(robot, em)

        print("\nSkipping measured dynamic Jacobian identification.")
        print("Experiment A uses no Jacobian; Experiment B uses the analytical PCC/IK Jacobian.")

        with LOG_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for experiment_name in RUN_EXPERIMENTS:
                maybe_pause(f"Press Enter to run {experiment_name}...")
                metrics = run_experiment(
                    robot,
                    em,
                    calib,
                    experiment_name,
                    writer,
                )
                f.flush()

                summary = summarize_metrics(experiment_name, metrics)
                summary_rows.append(summary)

                print("Summary:")
                for k, v in summary.items():
                    print(f"  {k}: {v}")

                robot.return_home()

        if summary_rows:
            with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as fsum:
                writer = csv.DictWriter(fsum, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        print("\nAll experiments finished.")
        print(f"Log saved to: {LOG_CSV_PATH}")
        print(f"Summary saved to: {SUMMARY_CSV_PATH}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        try:
            robot.return_home()
        except Exception as e:
            print(f"[WARN] Could not return home: {e}")

    finally:
        print("\nClosing hardware...")
        if em is not None:
            em.stop()
        robot.close()
        print("Done.")


if __name__ == "__main__":
    main()
