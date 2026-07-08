#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
three_experiments_circle_tracking.py

Three experiment modes on the same hardware driver:

Experiment 1:
    open_loop_command_circle
    q_ref(t) = [z_mm, rot_deg, bend_deg]
    q_cmd(t) = q_ref(t)

Experiment 2:
    joint_pid_command_circle
    q_ref(t) = [z_mm, rot_deg, bend_deg]
    q_actual(t) from servo present position
    q_cmd(t) = q_ref(t) + PID(q_ref - q_actual)

Experiment 3:
    ik_joint_pid_cartesian_circle
    Cartesian circular reference p_ref(t) in a conservative reachable region
    simple PCC IK -> q_ref(t) = [z_mm, rot_deg, bend_deg]
    q_cmd(t) = q_ref(t) + joint-space PID(q_ref - q_actual)

Important:
- This is still a hardware servo-level script.
- The IK here is a lightweight PCC constant-curvature approximation, not the full Drake planner.
- Do NOT directly treat this as a validated Cartesian controller until you compare with camera / SmartProbe pose feedback.
- The third experiment only uses IK to generate a feasible feedforward reference. The PID feedback is still servo-position feedback.
"""

import csv
import math
import time
from pathlib import Path

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupBulkRead,
    GroupBulkWrite,
    PacketHandler,
    PortHandler,
    DXL_LOBYTE,
    DXL_HIBYTE,
    DXL_LOWORD,
    DXL_HIWORD,
)

# ============================================================
# USER CONFIG
# ============================================================

PORT = "COM10"
BAUD = 1_000_000
PROTOCOL = 2.0

DXL_TRANS = 8     # translation motor, Extended Position mode
DXL_ROT   = 12    # rotation motor, Position mode
DXL_BEND  = 10    # bending motor, Position mode

LOG_CSV_PATH = Path(__file__).with_name("three_experiments_tracking_log.csv")

# Which experiments to run, in order.
RUN_EXPERIMENTS = [
    "open_loop_command_circle",
    "joint_pid_command_circle",
    "ik_joint_pid_cartesian_circle",
]

PAUSE_BETWEEN_EXPERIMENTS = True

# ============================================================
# HARDWARE CALIBRATION
# ============================================================

SCREW_LEAD_MM_PER_REV = 1.027210

Z_HOME_MM = 2.1
Z_MAX_MM  = 13.1

# Conservative insertion range used by experiments.
Z_SAFE_MIN_MM = 2.1
Z_SAFE_MAX_MM = 10.0

# Common insertion depth for command-space circle experiments.
CIRCLE_Z_MM = 6.5

# Rotation.
R0_DECIMAL = 2048

ROT_MIN_DEG = -179.0
ROT_MAX_DEG =  179.0

# More conservative experimental rotation range.
ROT_SAFE_MIN_DEG = -120.0
ROT_SAFE_MAX_DEG =  120.0

# Bend calibration from your description:
#   deadband [1435, 1595]
#   true neutral / home = 1515
#   +45 deg at 1805
#   -45 deg at 1181
BEND_HOME_DECIMAL = 1515

BEND_NEG_ONSET_DECIMAL = 1435
BEND_POS_ONSET_DECIMAL = 1595

BEND_NEG_45_DECIMAL = 1181
BEND_POS_45_DECIMAL = 1805

PROBE_BEND_MIN_DEG = -45.0
PROBE_BEND_MAX_DEG =  45.0

# Conservative experimental bend range.
BEND_SAFE_MIN_DEG = -30.0
BEND_SAFE_MAX_DEG =  30.0

BEND_ZERO_EPS_PROBE_DEG = 0.5

# Hysteresis compensation when returning home.
HYST_COMP_PROBE_DEG = 5.0
HYST_COMP_DWELL_S = 1.0

# ============================================================
# EXPERIMENT TIMING
# ============================================================

TRAJECTORY_DURATION_S = 30.0
SAMPLE_HZ = 20.0
NUM_CYCLES = 1

# ============================================================
# EXPERIMENT 1 / 2: COMMAND-SPACE CIRCLE
# ============================================================

# This circle is in [rotation_deg, bend_deg] command space.
# It is intentionally kept away from rotation +/-180 and bend +/-45.
COMMAND_ROT_CENTER_DEG = 0.0
COMMAND_ROT_RADIUS_DEG = 60.0

COMMAND_BEND_CENTER_DEG = 0.0
COMMAND_BEND_RADIUS_DEG = 22.0

# ============================================================
# EXPERIMENT 3: CARTESIAN CIRCLE + SIMPLE PCC IK
# ============================================================

# PCC arc length used by the planner report: Lcc = 0.006 m.
# Use mm here because the hardware script works mostly in mm.
L_CC_MM = 6.0

# Conservative bend cap for IK-generated references.
IK_BEND_LIMIT_DEG = 28.0

# The Cartesian circle is intentionally offset from the centerline.
# Reason:
#   A circle centered at x=0,y=0 would require phi to sweep 0..360 deg,
#   which is unsafe for the rotation servo in Position mode.
#
# This offset circle remains in a small phi window, roughly +/-15 deg.
CART_CIRCLE_CENTER_X_MM = 1.25
CART_CIRCLE_CENTER_Y_MM = 0.00
CART_CIRCLE_RADIUS_MM   = 0.25

# Desired axial tip position is chosen so the insertion stays around CIRCLE_Z_MM.
# The script computes insertion z_mm = tip_z_des - PCC_axial_component.
# Keep this conservative.
CART_TIP_Z_DES_MM = CIRCLE_Z_MM + L_CC_MM

# ============================================================
# PID CONFIG
# ============================================================

# This is an outer PID on top of the Dynamixel internal servo position control.
# Keep gains conservative.
#
# q = [z_mm, rot_deg, bend_deg]
#
# If the probe oscillates or commands look too aggressive:
#   reduce Kp
#   set Ki = 0
#   set Kd = 0
#   reduce output limits
JOINT_PID_KP = [0.15, 0.20, 0.20]
JOINT_PID_KI = [0.00, 0.00, 0.00]
JOINT_PID_KD = [0.01, 0.01, 0.01]

JOINT_PID_INTEGRAL_LIMIT = [0.5, 10.0, 8.0]     # [mm*s, deg*s, deg*s]
JOINT_PID_OUTPUT_LIMIT   = [0.30, 5.0, 4.0]     # [mm, deg, deg]

# ============================================================
# MOTION PROFILES
# ============================================================

ROT_PROFILE_VEL_RAW    = 22
ROT_PROFILE_ACCEL_RAW  = 10

BEND_PROFILE_VEL_RAW   = 13
BEND_PROFILE_ACCEL_RAW = 6

# Optional translation profile. If your old setup already handles this well,
# these conservative values should still be fine.
TRANS_PROFILE_VEL_RAW    = 30
TRANS_PROFILE_ACCEL_RAW  = 10

# ============================================================
# TOLERANCES / TIMEOUTS
# ============================================================

DECIMAL_TOL_TRANS = 20
DECIMAL_TOL_ROT   = 10
DECIMAL_TOL_BEND  = 10

TIMEOUT_TRANS_S = 25.0
TIMEOUT_ROT_S   = 18.0
TIMEOUT_BEND_S  = 18.0

# ============================================================
# DYNAMIXEL CONTROL TABLE
# ============================================================

ADDR_OPERATING_MODE       = 11
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_TORQUE_ENABLE        = 64
ADDR_GOAL_POSITION        = 116
ADDR_PRESENT_POSITION     = 132

TORQUE_ON  = 1
TORQUE_OFF = 0

LEN_4 = 4

DECIMAL_PER_REV = 4096
DECIMAL_PER_DEG = DECIMAL_PER_REV / 360.0

EXT_MIN = -1_048_575
EXT_MAX =  1_048_575

POS_MIN = 0
POS_MAX = 4095


# ============================================================
# LOW-LEVEL HELPERS
# ============================================================

def u32_to_i32(u: int) -> int:
    return u - (1 << 32) if u >= (1 << 31) else u


def goal_bytes(pos: int):
    pos = int(pos)
    return [
        DXL_LOBYTE(DXL_LOWORD(pos)),
        DXL_HIBYTE(DXL_LOWORD(pos)),
        DXL_LOBYTE(DXL_HIWORD(pos)),
        DXL_HIBYTE(DXL_HIWORD(pos)),
    ]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_ext(goal_decimal: int, name: str) -> int:
    goal_decimal = int(goal_decimal)
    if goal_decimal < EXT_MIN or goal_decimal > EXT_MAX:
        raise RuntimeError(
            f"{name} goal {goal_decimal} outside Extended range [{EXT_MIN}, {EXT_MAX}]"
        )
    return goal_decimal


def clamp_pos(goal_decimal: int, name: str) -> int:
    goal_decimal = int(goal_decimal)
    if goal_decimal < POS_MIN or goal_decimal > POS_MAX:
        raise RuntimeError(
            f"{name} goal {goal_decimal} outside Position range [{POS_MIN}, {POS_MAX}]"
        )
    return goal_decimal


def deg_to_decimal(deg: float) -> int:
    return int(round(deg * DECIMAL_PER_DEG))


def decimal_to_deg(decimal_pos: int) -> float:
    return decimal_pos / DECIMAL_PER_DEG


def decimal_per_mm_from_lead(lead_mm_per_rev: float) -> float:
    if lead_mm_per_rev <= 0:
        raise ValueError("SCREW_LEAD_MM_PER_REV must be > 0")
    return DECIMAL_PER_REV / lead_mm_per_rev


def wrap_to_signed_180(angle_deg: float) -> float:
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and angle_deg > 0:
        return 180.0
    return wrapped


def angle_error_deg(target_deg: float, actual_deg: float) -> float:
    return wrap_to_signed_180(target_deg - actual_deg)


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


def read_present_pos(ph, port, dxl_id) -> int:
    u32, res, err = ph.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
    if res != COMM_SUCCESS:
        raise RuntimeError(f"[ID:{dxl_id}] read position failed: {ph.getTxRxResult(res)}")
    if err != 0:
        raise RuntimeError(f"[ID:{dxl_id}] read position packet error: {ph.getRxPacketError(err)}")
    return u32_to_i32(u32)


def set_operating_mode(ph, port, dxl_id, mode_val: int):
    write1(ph, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
    time.sleep(0.05)
    write1(ph, port, dxl_id, ADDR_OPERATING_MODE, mode_val)
    time.sleep(0.05)


def set_profile(ph, port, dxl_id, accel_raw: int, vel_raw: int):
    write1(ph, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
    time.sleep(0.02)
    write4(ph, port, dxl_id, ADDR_PROFILE_ACCELERATION, accel_raw)
    write4(ph, port, dxl_id, ADDR_PROFILE_VELOCITY, vel_raw)
    time.sleep(0.02)


def send_goals(gbw, ph, goals_dict):
    for dxl_id, goal in goals_dict.items():
        ok = gbw.addParam(
            dxl_id,
            ADDR_GOAL_POSITION,
            LEN_4,
            goal_bytes(int(goal)),
        )
        if not ok:
            raise RuntimeError(f"BulkWrite addParam failed for ID {dxl_id}")

    res = gbw.txPacket()
    gbw.clearParam()

    if res != COMM_SUCCESS:
        raise RuntimeError(f"BulkWrite txPacket failed: {ph.getTxRxResult(res)}")


def bulkread_positions(gbr, ids, ph=None, port=None, retries=3, retry_sleep=0.03):
    for _ in range(retries):
        res = gbr.txRxPacket()
        if res == COMM_SUCCESS:
            out = {}
            ok_all = True

            for dxl_id in ids:
                if not gbr.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_4):
                    ok_all = False
                    break

                u32 = gbr.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_4)
                out[dxl_id] = u32_to_i32(u32)

            if ok_all:
                return out

        time.sleep(retry_sleep)

    if ph is not None and port is not None:
        out = {}
        for dxl_id in ids:
            out[dxl_id] = read_present_pos(ph, port, dxl_id)
        print("[WARN] Bulk read failed; recovered using individual reads")
        return out

    return None


def wait_until_reached(gbr, ids_all, dxl_id, goal, tol_decimal, timeout_s, ph=None, port=None):
    t0 = time.time()

    while True:
        if time.time() - t0 > timeout_s:
            return False

        pos = bulkread_positions(gbr, ids_all, ph=ph, port=port)
        if pos is None:
            continue

        if abs(goal - pos[dxl_id]) <= tol_decimal:
            return True

        time.sleep(0.02)


# ============================================================
# BEND CALIBRATION
# ============================================================

def probe_deg_to_bend_decimal(probe_deg: float) -> int:
    """
    Convert target probe bending angle to bending motor decimal.

    Uses:
      deadband: [1435, 1595]
      home: 1515
      +45 deg at 1805
      -45 deg at 1181
    """
    probe_deg = clamp(probe_deg, PROBE_BEND_MIN_DEG, PROBE_BEND_MAX_DEG)

    if abs(probe_deg) < BEND_ZERO_EPS_PROBE_DEG:
        return BEND_HOME_DECIMAL

    if probe_deg > 0:
        dec = BEND_POS_ONSET_DECIMAL + probe_deg * (
            BEND_POS_45_DECIMAL - BEND_POS_ONSET_DECIMAL
        ) / 45.0
    else:
        dec = BEND_NEG_ONSET_DECIMAL + probe_deg * (
            BEND_NEG_ONSET_DECIMAL - BEND_NEG_45_DECIMAL
        ) / 45.0

    return clamp_pos(int(round(dec)), "Bending")


def bend_decimal_to_probe_deg_est(decimal: int) -> float:
    """
    Rough inverse estimate for logging and joint-space PID feedback.

    Deadband region returns 0.
    This is not true probe shape measurement.
    """
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
# SIMPLE PID
# ============================================================

class JointPID:
    """
    PID on q = [z_mm, rot_deg, bend_deg].

    This PID outputs a correction in the same units:
      [mm, deg, deg]

    It is intended as a conservative outer loop, not a replacement for the
    Dynamixel internal position controller.
    """

    def __init__(self, kp, ki, kd, integral_limit, output_limit):
        self.kp = list(kp)
        self.ki = list(ki)
        self.kd = list(kd)
        self.integral_limit = list(integral_limit)
        self.output_limit = list(output_limit)

        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = None

    def reset(self):
        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = None

    def update(self, q_ref, q_actual, dt):
        if dt <= 1e-6:
            dt = 1e-6

        err = [
            q_ref["z_mm"] - q_actual["z_mm"],
            angle_error_deg(q_ref["rot_deg"], q_actual["rot_deg"]),
            q_ref["bend_deg"] - q_actual["bend_deg"],
        ]

        if self.prev_error is None:
            derr = [0.0, 0.0, 0.0]
        else:
            derr = [
                (err[i] - self.prev_error[i]) / dt
                for i in range(3)
            ]

        out = [0.0, 0.0, 0.0]

        for i in range(3):
            self.integral[i] += err[i] * dt
            self.integral[i] = clamp(
                self.integral[i],
                -self.integral_limit[i],
                self.integral_limit[i],
            )

            out[i] = (
                self.kp[i] * err[i]
                + self.ki[i] * self.integral[i]
                + self.kd[i] * derr[i]
            )

            out[i] = clamp(out[i], -self.output_limit[i], self.output_limit[i])

        self.prev_error = err

        return {
            "z_mm": out[0],
            "rot_deg": out[1],
            "bend_deg": out[2],
        }


# ============================================================
# SIMPLE PCC FK / IK
# ============================================================

def pcc_lateral_mm_from_theta_deg(theta_deg: float) -> float:
    """
    Constant-curvature lateral displacement magnitude for a segment of length L_CC_MM.

    rho = L * (1 - cos(theta)) / theta

    theta in radians.
    """
    theta = math.radians(abs(theta_deg))
    if theta < 1e-8:
        return 0.0
    return L_CC_MM * (1.0 - math.cos(theta)) / theta


def pcc_axial_mm_from_theta_deg(theta_deg: float) -> float:
    """
    Constant-curvature axial component.

    z_axis = L * sin(theta) / theta
    """
    theta = math.radians(abs(theta_deg))
    if theta < 1e-8:
        return L_CC_MM
    return L_CC_MM * math.sin(theta) / theta


def theta_deg_from_lateral_rho_mm(rho_mm: float, theta_max_deg: float):
    """
    Invert rho = L * (1 - cos(theta)) / theta using bisection.

    Returns:
      theta_deg, reachable
    """
    rho_mm = max(0.0, rho_mm)

    rho_max = pcc_lateral_mm_from_theta_deg(theta_max_deg)
    reachable = True

    if rho_mm > rho_max:
        rho_mm = rho_max
        reachable = False

    if rho_mm < 1e-6:
        return 0.0, reachable

    lo = 0.0
    hi = math.radians(theta_max_deg)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid < 1e-9:
            rho_mid = 0.0
        else:
            rho_mid = L_CC_MM * (1.0 - math.cos(mid)) / mid

        if rho_mid < rho_mm:
            lo = mid
        else:
            hi = mid

    theta_rad = 0.5 * (lo + hi)
    return math.degrees(theta_rad), reachable


def simple_pcc_ik_cartesian_to_q_hw(x_mm, y_mm, tip_z_des_mm):
    """
    Simplified PCC IK.

    Desired Cartesian point:
      x_mm, y_mm: lateral tip location in base frame
      tip_z_des_mm: axial tip location

    Output:
      q_ref = [z_mm, rot_deg, bend_deg]

    Convention:
      - bend_deg is always positive here.
      - lateral direction is handled by rot_deg = atan2(y, x).
      - This avoids signed bend ambiguity for the first IK experiment.
    """
    rho_mm = math.sqrt(x_mm * x_mm + y_mm * y_mm)

    theta_deg, reachable = theta_deg_from_lateral_rho_mm(
        rho_mm,
        theta_max_deg=IK_BEND_LIMIT_DEG,
    )

    rot_deg = math.degrees(math.atan2(y_mm, x_mm))
    rot_deg = wrap_to_signed_180(rot_deg)

    axial_cc_mm = pcc_axial_mm_from_theta_deg(theta_deg)
    z_mm = tip_z_des_mm - axial_cc_mm

    q_ref = {
        "z_mm": z_mm,
        "rot_deg": rot_deg,
        "bend_deg": theta_deg,
    }

    q_ref = safety_clamp_q_hw(q_ref)

    return q_ref, reachable, theta_deg, rho_mm, axial_cc_mm


def q_hw_to_pcc_tip_estimate(q_hw):
    """
    Rough FK estimate from hardware command/state to Cartesian tip position.

    This is only for logging/comparison. It is not calibrated truth.
    """
    z_mm = q_hw["z_mm"]
    rot_deg = q_hw["rot_deg"]
    bend_deg = q_hw["bend_deg"]

    theta_abs_deg = abs(bend_deg)

    rho = pcc_lateral_mm_from_theta_deg(theta_abs_deg)
    axial = pcc_axial_mm_from_theta_deg(theta_abs_deg)

    if bend_deg >= 0:
        phi_deg = rot_deg
    else:
        # Negative bend can be approximated as positive bend in the opposite plane.
        phi_deg = rot_deg + 180.0

    phi = math.radians(phi_deg)

    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    z = z_mm + axial

    return {
        "x_mm": x,
        "y_mm": y,
        "tip_z_mm": z,
    }


# ============================================================
# TRAJECTORY GENERATORS
# ============================================================

def command_space_circle_ref(t_s: float, total_time_s: float):
    """
    Reference for experiments 1 and 2:
      z fixed,
      rotation and bend form a closed circle in command space.
    """
    omega = 2.0 * math.pi * NUM_CYCLES / total_time_s
    phase = omega * t_s

    z_ref = CIRCLE_Z_MM
    rot_ref = COMMAND_ROT_CENTER_DEG + COMMAND_ROT_RADIUS_DEG * math.cos(phase)
    bend_ref = COMMAND_BEND_CENTER_DEG + COMMAND_BEND_RADIUS_DEG * math.sin(phase)

    q_ref = {
        "z_mm": z_ref,
        "rot_deg": rot_ref,
        "bend_deg": bend_ref,
    }

    q_ref = safety_clamp_q_hw(q_ref)
    p_est = q_hw_to_pcc_tip_estimate(q_ref)

    return {
        "q_ref": q_ref,
        "p_ref": p_est,
        "ik_reachable": True,
        "ik_theta_deg": abs(bend_ref),
        "ik_rho_mm": pcc_lateral_mm_from_theta_deg(abs(bend_ref)),
        "ik_axial_mm": pcc_axial_mm_from_theta_deg(abs(bend_ref)),
    }


def ik_cartesian_circle_ref(t_s: float, total_time_s: float):
    """
    Reference for experiment 3:
      p_ref(t) is a small Cartesian circle in lateral x-y space,
      centered away from the centerline so phi does not need to rotate 360 deg.
    """
    omega = 2.0 * math.pi * NUM_CYCLES / total_time_s
    phase = omega * t_s

    x_ref = CART_CIRCLE_CENTER_X_MM + CART_CIRCLE_RADIUS_MM * math.cos(phase)
    y_ref = CART_CIRCLE_CENTER_Y_MM + CART_CIRCLE_RADIUS_MM * math.sin(phase)
    tip_z_ref = CART_TIP_Z_DES_MM

    q_ref, reachable, theta_deg, rho_mm, axial_mm = simple_pcc_ik_cartesian_to_q_hw(
        x_ref,
        y_ref,
        tip_z_ref,
    )

    return {
        "q_ref": q_ref,
        "p_ref": {
            "x_mm": x_ref,
            "y_mm": y_ref,
            "tip_z_mm": tip_z_ref,
        },
        "ik_reachable": reachable,
        "ik_theta_deg": theta_deg,
        "ik_rho_mm": rho_mm,
        "ik_axial_mm": axial_mm,
    }


# ============================================================
# SAFETY CLAMPS
# ============================================================

def safety_clamp_q_hw(q):
    """
    Clamp in physical units before conversion to servo decimals.
    """
    return {
        "z_mm": clamp(q["z_mm"], Z_SAFE_MIN_MM, Z_SAFE_MAX_MM),
        "rot_deg": clamp(
            wrap_to_signed_180(q["rot_deg"]),
            ROT_SAFE_MIN_DEG,
            ROT_SAFE_MAX_DEG,
        ),
        "bend_deg": clamp(
            q["bend_deg"],
            BEND_SAFE_MIN_DEG,
            BEND_SAFE_MAX_DEG,
        ),
    }


def validate_experiment_ranges():
    command_rot_min = COMMAND_ROT_CENTER_DEG - COMMAND_ROT_RADIUS_DEG
    command_rot_max = COMMAND_ROT_CENTER_DEG + COMMAND_ROT_RADIUS_DEG

    command_bend_min = COMMAND_BEND_CENTER_DEG - COMMAND_BEND_RADIUS_DEG
    command_bend_max = COMMAND_BEND_CENTER_DEG + COMMAND_BEND_RADIUS_DEG

    if command_rot_min < ROT_SAFE_MIN_DEG or command_rot_max > ROT_SAFE_MAX_DEG:
        raise RuntimeError(
            f"Command-space rotation circle [{command_rot_min}, {command_rot_max}] "
            f"exceeds safe range [{ROT_SAFE_MIN_DEG}, {ROT_SAFE_MAX_DEG}]"
        )

    if command_bend_min < BEND_SAFE_MIN_DEG or command_bend_max > BEND_SAFE_MAX_DEG:
        raise RuntimeError(
            f"Command-space bend circle [{command_bend_min}, {command_bend_max}] "
            f"exceeds safe range [{BEND_SAFE_MIN_DEG}, {BEND_SAFE_MAX_DEG}]"
        )

    if CIRCLE_Z_MM < Z_SAFE_MIN_MM or CIRCLE_Z_MM > Z_SAFE_MAX_MM:
        raise RuntimeError(
            f"CIRCLE_Z_MM={CIRCLE_Z_MM} outside safe range "
            f"[{Z_SAFE_MIN_MM}, {Z_SAFE_MAX_MM}]"
        )

    # Check IK circle samples conservatively.
    max_abs_rot = 0.0
    max_bend = 0.0
    min_z = 1e9
    max_z = -1e9

    for i in range(361):
        t = TRAJECTORY_DURATION_S * i / 360.0
        ref = ik_cartesian_circle_ref(t, TRAJECTORY_DURATION_S)
        q = ref["q_ref"]

        max_abs_rot = max(max_abs_rot, abs(q["rot_deg"]))
        max_bend = max(max_bend, abs(q["bend_deg"]))
        min_z = min(min_z, q["z_mm"])
        max_z = max(max_z, q["z_mm"])

    print("\nIK Cartesian circle safety preview:")
    print(f"  z range:        [{min_z:.3f}, {max_z:.3f}] mm")
    print(f"  max |rot|:      {max_abs_rot:.3f} deg")
    print(f"  max |bend|:     {max_bend:.3f} deg")

    if max_abs_rot > ROT_SAFE_MAX_DEG:
        raise RuntimeError("IK Cartesian circle exceeds rotation safe range")

    if max_bend > BEND_SAFE_MAX_DEG:
        raise RuntimeError("IK Cartesian circle exceeds bend safe range")

    if min_z < Z_SAFE_MIN_MM or max_z > Z_SAFE_MAX_MM:
        raise RuntimeError("IK Cartesian circle exceeds insertion safe range")


# ============================================================
# MAIN
# ============================================================

def main():
    validate_experiment_ranges()

    ids_all = [DXL_TRANS, DXL_ROT, DXL_BEND]

    print("=" * 80)
    print("THREE EXPERIMENTS: OPEN LOOP vs JOINT PID vs IK + JOINT PID")
    print("=" * 80)
    print(f"Port:              {PORT}")
    print(f"Baud:              {BAUD}")
    print(f"Translation ID:    {DXL_TRANS}")
    print(f"Rotation ID:       {DXL_ROT}")
    print(f"Bending ID:        {DXL_BEND}")
    print()
    print(f"Safe z range:      [{Z_SAFE_MIN_MM}, {Z_SAFE_MAX_MM}] mm")
    print(f"Safe rot range:    [{ROT_SAFE_MIN_DEG}, {ROT_SAFE_MAX_DEG}] deg")
    print(f"Safe bend range:   [{BEND_SAFE_MIN_DEG}, {BEND_SAFE_MAX_DEG}] deg")
    print()
    print(f"Duration each:     {TRAJECTORY_DURATION_S:.1f} s")
    print(f"Sample rate:       {SAMPLE_HZ:.1f} Hz")
    print(f"Cycles each:       {NUM_CYCLES}")
    print(f"Log file:          {LOG_CSV_PATH}")
    print()
    print("Experiments:")
    for name in RUN_EXPERIMENTS:
        print(f"  - {name}")
    print()
    print("Make sure the probe is physically safe and the translation reference is valid.")
    print(f"At startup, current translation decimal will be treated as Z_HOME_MM = {Z_HOME_MM:.3f} mm.")
    input("Press Enter to start hardware setup...")

    port = PortHandler(PORT)
    ph = PacketHandler(PROTOCOL)

    gbw = GroupBulkWrite(port, ph)
    gbr = GroupBulkRead(port, ph)

    last_good_positions = None

    def get_positions():
        nonlocal last_good_positions
        pos = bulkread_positions(gbr, ids_all, ph=ph, port=port)
        if pos is not None:
            last_good_positions = pos.copy()
            return pos

        if last_good_positions is not None:
            print("[WARN] Position read failed; using last known positions")
            return last_good_positions.copy()

        raise RuntimeError("Position read failed and no last known position available")

    def move_and_wait(goals, wait_items, label=""):
        send_goals(gbw, ph, goals)

        ok_all = True
        for dxl_id, goal, tol, timeout_s in wait_items:
            ok = wait_until_reached(
                gbr,
                ids_all,
                dxl_id,
                goal,
                tol,
                timeout_s,
                ph=ph,
                port=port,
            )
            ok_all = ok_all and ok

        if not ok_all:
            print(f"[WARN] move_and_wait timeout: {label}")

        return ok_all

    try:
        if not port.openPort():
            raise SystemExit("Failed to open port")

        if not port.setBaudRate(BAUD):
            raise SystemExit("Failed to set baudrate")

        # ------------------------------------------------------------
        # Motor setup
        # ------------------------------------------------------------
        print("\nSetting operating modes...")

        set_operating_mode(ph, port, DXL_TRANS, 4)   # Extended Position mode
        set_operating_mode(ph, port, DXL_ROT,   3)   # Position mode
        set_operating_mode(ph, port, DXL_BEND,  3)   # Position mode

        set_profile(ph, port, DXL_TRANS, TRANS_PROFILE_ACCEL_RAW, TRANS_PROFILE_VEL_RAW)
        set_profile(ph, port, DXL_ROT,   ROT_PROFILE_ACCEL_RAW,   ROT_PROFILE_VEL_RAW)
        set_profile(ph, port, DXL_BEND,  BEND_PROFILE_ACCEL_RAW,  BEND_PROFILE_VEL_RAW)

        for dxl_id in ids_all:
            ok = gbr.addParam(dxl_id, ADDR_PRESENT_POSITION, LEN_4)
            if not ok:
                raise RuntimeError(f"BulkRead addParam failed for ID {dxl_id}")

        print("Enabling torque...")
        for dxl_id in ids_all:
            write1(ph, port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

        time.sleep(0.5)

        # ------------------------------------------------------------
        # Snapshot references
        # ------------------------------------------------------------
        print("\nSnapshotting hardware references...")
        z_ref_decimal = read_present_pos(ph, port, DXL_TRANS)
        r0_decimal = R0_DECIMAL
        dpm = decimal_per_mm_from_lead(SCREW_LEAD_MM_PER_REV)

        print(f"z_ref_decimal: {z_ref_decimal}  (= {Z_HOME_MM:.3f} mm)")
        print(f"r0_decimal:    {r0_decimal}  (= 0 deg)")
        print(f"dec/mm:        {dpm:.3f}")

        def z_mm_to_ticks(z_mm: float) -> int:
            z_mm = clamp(z_mm, Z_HOME_MM, Z_MAX_MM)
            goal = z_ref_decimal + int(round((z_mm - Z_HOME_MM) * dpm))
            return clamp_ext(goal, "Translation")

        def ticks_to_z_mm(z_decimal: int) -> float:
            return Z_HOME_MM + (z_decimal - z_ref_decimal) / dpm

        def rot_deg_to_ticks(rot_deg: float) -> int:
            rot_deg = wrap_to_signed_180(rot_deg)
            rot_deg = clamp(rot_deg, ROT_MIN_DEG, ROT_MAX_DEG)
            goal = r0_decimal + deg_to_decimal(rot_deg)
            return clamp_pos(goal, "Rotation")

        def ticks_to_rot_deg(r_decimal: int) -> float:
            return decimal_to_deg(r_decimal - r0_decimal)

        def q_hw_to_goals(q):
            q = safety_clamp_q_hw(q)
            return {
                DXL_TRANS: z_mm_to_ticks(q["z_mm"]),
                DXL_ROT:   rot_deg_to_ticks(q["rot_deg"]),
                DXL_BEND:  probe_deg_to_bend_decimal(q["bend_deg"]),
            }

        def positions_to_q_hw(pos):
            return {
                "z_mm": ticks_to_z_mm(pos[DXL_TRANS]),
                "rot_deg": ticks_to_rot_deg(pos[DXL_ROT]),
                "bend_deg": bend_decimal_to_probe_deg_est(pos[DXL_BEND]),
            }

        def send_q_hw(q):
            goals = q_hw_to_goals(q)
            send_goals(gbw, ph, goals)
            return goals

        def move_to_q_hw_wait(q, label="move_to_q"):
            q = safety_clamp_q_hw(q)
            goals = q_hw_to_goals(q)

            return move_and_wait(
                goals,
                [
                    (DXL_TRANS, goals[DXL_TRANS], DECIMAL_TOL_TRANS, TIMEOUT_TRANS_S),
                    (DXL_ROT,   goals[DXL_ROT],   DECIMAL_TOL_ROT,   TIMEOUT_ROT_S),
                    (DXL_BEND,  goals[DXL_BEND],  DECIMAL_TOL_BEND,  TIMEOUT_BEND_S),
                ],
                label=label,
            )

        def bend_hysteresis_then_home():
            actual = get_positions()
            q_actual = positions_to_q_hw(actual)
            bend_now = q_actual["bend_deg"]

            if bend_now > BEND_ZERO_EPS_PROBE_DEG:
                tug_deg = -HYST_COMP_PROBE_DEG
            elif bend_now < -BEND_ZERO_EPS_PROBE_DEG:
                tug_deg = +HYST_COMP_PROBE_DEG
            else:
                tug_deg = 0.0

            if abs(tug_deg) > 0.0:
                q_tug = {
                    "z_mm": q_actual["z_mm"],
                    "rot_deg": q_actual["rot_deg"],
                    "bend_deg": tug_deg,
                }

                print(f"  Hysteresis tug: {tug_deg:+.1f} deg")
                move_to_q_hw_wait(q_tug, label="hysteresis tug")
                time.sleep(HYST_COMP_DWELL_S)

            actual = get_positions()
            q_actual = positions_to_q_hw(actual)

            q_home_bend = {
                "z_mm": q_actual["z_mm"],
                "rot_deg": q_actual["rot_deg"],
                "bend_deg": 0.0,
            }

            move_to_q_hw_wait(q_home_bend, label="bend home")

        def return_home():
            print("\nReturning home safely...")

            bend_hysteresis_then_home()

            actual = get_positions()
            q_actual = positions_to_q_hw(actual)

            q_rot_home = {
                "z_mm": q_actual["z_mm"],
                "rot_deg": 0.0,
                "bend_deg": 0.0,
            }
            move_to_q_hw_wait(q_rot_home, label="rotation home")

            q_z_home = {
                "z_mm": Z_HOME_MM,
                "rot_deg": 0.0,
                "bend_deg": 0.0,
            }
            move_to_q_hw_wait(q_z_home, label="translation home")

            final_pos = get_positions()
            final_q = positions_to_q_hw(final_pos)

            print("Home result:")
            print(f"  z_actual_mm:     {final_q['z_mm']:.3f}")
            print(f"  rot_actual_deg:  {final_q['rot_deg']:.3f}")
            print(f"  bend_est_deg:    {final_q['bend_deg']:.3f}")
            print(f"  decimals:        {final_pos}")

        # Validate rotation decimal range.
        rot_min_goal = rot_deg_to_ticks(ROT_MIN_DEG)
        rot_max_goal = rot_deg_to_ticks(ROT_MAX_DEG)

        if not (POS_MIN <= rot_min_goal <= POS_MAX and POS_MIN <= rot_max_goal <= POS_MAX):
            raise RuntimeError(
                f"R0_DECIMAL={R0_DECIMAL} is not centered enough for "
                f"[{ROT_MIN_DEG}, {ROT_MAX_DEG}] deg in Position mode."
            )

        # ------------------------------------------------------------
        # Go home before all experiments
        # ------------------------------------------------------------
        print("\nGoing to initial home pose...")
        move_to_q_hw_wait(
            {
                "z_mm": Z_HOME_MM,
                "rot_deg": 0.0,
                "bend_deg": 0.0,
            },
            label="initial home",
        )

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        fieldnames = [
            "experiment",
            "controller_mode",
            "timestamp_unix",
            "t_s",
            "dt_s",

            "z_ref_mm",
            "rot_ref_deg",
            "bend_ref_deg",

            "z_cmd_mm",
            "rot_cmd_deg",
            "bend_cmd_deg",

            "z_actual_mm",
            "rot_actual_deg",
            "bend_actual_deg_est",

            "z_pid_correction_mm",
            "rot_pid_correction_deg",
            "bend_pid_correction_deg",

            "z_goal_decimal",
            "rot_goal_decimal",
            "bend_goal_decimal",

            "z_actual_decimal",
            "rot_actual_decimal",
            "bend_actual_decimal",

            "x_ref_mm",
            "y_ref_mm",
            "tip_z_ref_mm",

            "x_est_actual_mm",
            "y_est_actual_mm",
            "tip_z_est_actual_mm",

            "ik_reachable",
            "ik_theta_deg",
            "ik_rho_mm",
            "ik_axial_mm",
        ]

        log_file = LOG_CSV_PATH.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        def run_tracking_experiment(experiment_name, ref_generator, use_joint_pid):
            print("\n" + "=" * 80)
            print(f"Starting experiment: {experiment_name}")
            print("=" * 80)

            # Move to initial reference before timed loop to avoid first-sample jump.
            ref0 = ref_generator(0.0, TRAJECTORY_DURATION_S)
            q0 = ref0["q_ref"]

            print("Moving to initial reference...")
            print(
                f"  q0: z={q0['z_mm']:.3f} mm, "
                f"rot={q0['rot_deg']:.3f} deg, "
                f"bend={q0['bend_deg']:.3f} deg"
            )

            move_to_q_hw_wait(q0, label=f"{experiment_name} initial reference")
            time.sleep(1.0)

            pid = JointPID(
                kp=JOINT_PID_KP,
                ki=JOINT_PID_KI,
                kd=JOINT_PID_KD,
                integral_limit=JOINT_PID_INTEGRAL_LIMIT,
                output_limit=JOINT_PID_OUTPUT_LIMIT,
            )
            pid.reset()

            dt_nominal = 1.0 / SAMPLE_HZ
            total_time_s = TRAJECTORY_DURATION_S

            t_start = time.monotonic()
            t_prev = t_start
            sample_idx = 0

            while True:
                now = time.monotonic()
                t_s = now - t_start

                if t_s > total_time_s:
                    break

                dt_s = now - t_prev
                t_prev = now

                if dt_s <= 0.0 or dt_s > 1.0:
                    dt_s = dt_nominal

                ref = ref_generator(t_s, total_time_s)
                q_ref = ref["q_ref"]

                actual_before = get_positions()
                q_actual_before = positions_to_q_hw(actual_before)

                if use_joint_pid:
                    correction = pid.update(q_ref, q_actual_before, dt_s)

                    q_cmd = {
                        "z_mm": q_ref["z_mm"] + correction["z_mm"],
                        "rot_deg": q_ref["rot_deg"] + correction["rot_deg"],
                        "bend_deg": q_ref["bend_deg"] + correction["bend_deg"],
                    }

                    q_cmd = safety_clamp_q_hw(q_cmd)
                else:
                    correction = {
                        "z_mm": 0.0,
                        "rot_deg": 0.0,
                        "bend_deg": 0.0,
                    }
                    q_cmd = safety_clamp_q_hw(q_ref)

                goals = send_q_hw(q_cmd)

                actual_after = get_positions()
                q_actual_after = positions_to_q_hw(actual_after)
                p_est_actual = q_hw_to_pcc_tip_estimate(q_actual_after)

                writer.writerow(
                    {
                        "experiment": experiment_name,
                        "controller_mode": "joint_pid" if use_joint_pid else "open_loop",
                        "timestamp_unix": time.time(),
                        "t_s": t_s,
                        "dt_s": dt_s,

                        "z_ref_mm": q_ref["z_mm"],
                        "rot_ref_deg": q_ref["rot_deg"],
                        "bend_ref_deg": q_ref["bend_deg"],

                        "z_cmd_mm": q_cmd["z_mm"],
                        "rot_cmd_deg": q_cmd["rot_deg"],
                        "bend_cmd_deg": q_cmd["bend_deg"],

                        "z_actual_mm": q_actual_after["z_mm"],
                        "rot_actual_deg": q_actual_after["rot_deg"],
                        "bend_actual_deg_est": q_actual_after["bend_deg"],

                        "z_pid_correction_mm": correction["z_mm"],
                        "rot_pid_correction_deg": correction["rot_deg"],
                        "bend_pid_correction_deg": correction["bend_deg"],

                        "z_goal_decimal": goals[DXL_TRANS],
                        "rot_goal_decimal": goals[DXL_ROT],
                        "bend_goal_decimal": goals[DXL_BEND],

                        "z_actual_decimal": actual_after[DXL_TRANS],
                        "rot_actual_decimal": actual_after[DXL_ROT],
                        "bend_actual_decimal": actual_after[DXL_BEND],

                        "x_ref_mm": ref["p_ref"]["x_mm"],
                        "y_ref_mm": ref["p_ref"]["y_mm"],
                        "tip_z_ref_mm": ref["p_ref"]["tip_z_mm"],

                        "x_est_actual_mm": p_est_actual["x_mm"],
                        "y_est_actual_mm": p_est_actual["y_mm"],
                        "tip_z_est_actual_mm": p_est_actual["tip_z_mm"],

                        "ik_reachable": ref["ik_reachable"],
                        "ik_theta_deg": ref["ik_theta_deg"],
                        "ik_rho_mm": ref["ik_rho_mm"],
                        "ik_axial_mm": ref["ik_axial_mm"],
                    }
                )
                log_file.flush()

                if sample_idx % max(1, int(SAMPLE_HZ / 2.0)) == 0:
                    print(
                        f"t={t_s:6.2f}s | "
                        f"ref=({q_ref['z_mm']:5.2f} mm, "
                        f"{q_ref['rot_deg']:7.2f} deg, "
                        f"{q_ref['bend_deg']:7.2f} deg) | "
                        f"cmd=({q_cmd['z_mm']:5.2f} mm, "
                        f"{q_cmd['rot_deg']:7.2f} deg, "
                        f"{q_cmd['bend_deg']:7.2f} deg) | "
                        f"act=({q_actual_after['z_mm']:5.2f} mm, "
                        f"{q_actual_after['rot_deg']:7.2f} deg, "
                        f"{q_actual_after['bend_deg']:7.2f} deg)"
                    )

                sample_idx += 1

                next_time = t_start + sample_idx * dt_nominal
                sleep_s = next_time - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)

            print(f"Experiment finished: {experiment_name}")

        # ------------------------------------------------------------
        # Run the three experiments
        # ------------------------------------------------------------
        for exp_name in RUN_EXPERIMENTS:
            if PAUSE_BETWEEN_EXPERIMENTS:
                print("\n" + "-" * 80)
                print(f"Next experiment: {exp_name}")
                input("Check the probe/camera. Press Enter to run this experiment...")

            if exp_name == "open_loop_command_circle":
                run_tracking_experiment(
                    experiment_name=exp_name,
                    ref_generator=command_space_circle_ref,
                    use_joint_pid=False,
                )

            elif exp_name == "joint_pid_command_circle":
                run_tracking_experiment(
                    experiment_name=exp_name,
                    ref_generator=command_space_circle_ref,
                    use_joint_pid=True,
                )

            elif exp_name == "ik_joint_pid_cartesian_circle":
                run_tracking_experiment(
                    experiment_name=exp_name,
                    ref_generator=ik_cartesian_circle_ref,
                    use_joint_pid=True,
                )

            else:
                raise RuntimeError(f"Unknown experiment name: {exp_name}")

            print(f"\nReturning to safe home after {exp_name}...")
            return_home()

        print("\n" + "=" * 80)
        print("All experiments finished.")
        print(f"Log saved to: {LOG_CSV_PATH}")
        print("=" * 80)

        log_file.close()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] KeyboardInterrupt received.")

        try:
            print("Attempting safe return home...")
            # These helpers only exist after reference setup.
            if "return_home" in locals():
                return_home()
        except Exception as e:
            print(f"[WARN] Could not return home cleanly after interrupt: {e}")

    finally:
        try:
            if "log_file" in locals() and not log_file.closed:
                log_file.close()
        except Exception:
            pass

        print("\nTorque off and closing port...")

        try:
            for dxl_id in ids_all:
                ph.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        except Exception:
            pass

        try:
            port.closePort()
        except Exception:
            pass

        print("Done.")


if __name__ == "__main__":
    main()