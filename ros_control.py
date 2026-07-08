import math
import os
import sys
import time

import h5py
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from dynamixel_sdk_custom_interfaces.msg import SyncSetPosition
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R


BASE_TOPIC = "/ndi/channel_11/pose"
TIP_TOPIC = "/ndi/channel_10/pose"
COMMAND_TOPIC = "/dynamixel/sync_set_position"

ROLE_ROT = "inner_rotation"
ROLE_BEND = "inner_bending"
ROLE_TRANS = "inner_translation"
ROLE_OUTER_BEND_1 = "outer_bending_1"
ROLE_OUTER_BEND_2 = "outer_bending_2"
ROLE_OUTER_ROT = "outer_rotation"

INNER_HOME_ROLES = (ROLE_TRANS, ROLE_ROT, ROLE_BEND)
FULL_HOME_ROLES = (
    ROLE_OUTER_BEND_1,
    ROLE_OUTER_BEND_2,
    ROLE_OUTER_ROT,
    ROLE_TRANS,
    ROLE_ROT,
    ROLE_BEND,
)
FULL_HOME_DURATION_S = 2.0

EXPERIMENTS = (
    "em_pid_no_jacobian_baseline",
    "ik_jacobian_em_pid_closed_loop",
)

TRAJ_SHAPE = "figure8_auto"
TRAJECTORY_DURATION_S = 30.0
SAMPLE_HZ = 20.0
NUM_CYCLES = 1
EM_TRAJ_RADIUS_MM = 0.25
EM_TRAJ_SIDE_MM = 0.50
FIG8_MIN_DURATION_S = 35.0
FIG8_MAX_DURATION_S = 90.0
FIG8_RATE_BUDGET_FRACTION = 0.8
FIG8_MIN_AX_MM = 0.05
FIG8_AY_RATIO = 1.0
FIG8_AZ_RATIO = 0.3
FIG8_AZ_MAX_MM = 1.0
FIG8_MIN_RHO_MM = 0.05
FIG8_MIN_XY_BBOX_MM = 0.98
FIG8_PLAN_SAMPLES = 180
FIG8_SEARCH_AX_STEPS = 90
FIG8_SEARCH_X0_STEPS = 12
FIG8_DURATION_STEP_S = 0.25
FIG8_Q_MARGIN = {
    "z_mm": 0.25,
    "rot_deg": 2.0,
    "bend_deg": 0.5,
}

CENTER_X_B_MM = 0.00
CENTER_Y_B_MM = 0.00
CENTER_TIP_Z_B_MM = 12.5
CALIB_X_STEP_MM = 0.10
CALIB_Y_STEP_MM = 0.15
CALIB_Z_STEP_MM = 0.40
CALIB_SAMPLE_DURATION_S = 0.70
MOVE_SETTLE_S = 2.00
EXPERIMENT_START_SETTLE_S = 5.0
EXPERIMENT_START_ERROR_MM = 0.35
EXPERIMENT_START_STABLE_SAMPLES = 5
EM_TIMEOUT_S = 0.50
EM_XY_ALIGNMENT_TOL_MM = 0.30
EM_XY_ALIGNMENT_STABLE_SAMPLES = 5
EM_XY_ALIGNMENT_TIMEOUT_S = 8.0
EM_XY_ALIGNMENT_GAIN = 0.70
EM_XY_ALIGNMENT_MAX_BEND_DEG = 18.0

L_CC_MM = 6.0
IK_BEND_LIMIT_DEG = 28.0

Z_SAFE_MIN_MM = 2.1
Z_SAFE_MAX_MM = 10.0
ROT_SAFE_MIN_DEG = -120.0
ROT_SAFE_MAX_DEG = 120.0
BEND_SAFE_MIN_DEG = -30.0
BEND_SAFE_MAX_DEG = 30.0

CART_PID_KP = np.array([0.35, 0.35, 0.25], dtype=float)
CART_PID_KI = np.array([0.00, 0.00, 0.00], dtype=float)
CART_PID_KD = np.array([0.015, 0.015, 0.008], dtype=float)
CART_PID_INTEGRAL_LIMIT_MM_S = np.array([0.5, 0.5, 0.5], dtype=float)
CART_PID_OUTPUT_LIMIT_MM = np.array([0.40, 0.40, 0.30], dtype=float)

PSEUDO_JOINT_PID_KP = np.array([0.22, 0.25, 0.25], dtype=float)
PSEUDO_JOINT_PID_KI = np.array([0.00, 0.00, 0.00], dtype=float)
PSEUDO_JOINT_PID_KD = np.array([0.015, 0.015, 0.015], dtype=float)
PSEUDO_JOINT_PID_INTEGRAL_LIMIT = np.array([0.4, 6.0, 6.0], dtype=float)
PSEUDO_JOINT_PID_OUTPUT_LIMIT = np.array([0.30, 3.0, 3.0], dtype=float)
POLAR_ROT_EPS_MM = 0.2

MAX_DQ_PER_SAMPLE = {
    "z_mm": 0.04,
    "rot_deg": 0.8,
    "bend_deg": 0.8,
}
MAX_EM_ERROR_MM = 5.0
ERROR_ABORT_SECONDS = 1.0
ABORT_NONE = 0
ABORT_LARGE_ERROR = 1
ABORT_INVALID_EM = 2
ABORT_START_NOT_SETTLED = 3

NOJAC_PID_Z_MM_PER_MM = 1.00
NOJAC_PID_ROT_DEG_PER_MM = 16.0
NOJAC_PID_BEND_DEG_PER_MM = 14.0
IK_JAC_Q_SCALE = np.array([CALIB_Z_STEP_MM, 2.0, 2.0], dtype=float)
IK_JAC_DAMPING = 0.02
JACOBIAN_SINGULAR_VALUE_EPS = 1e-12

SAMPLE_COLUMNS = [
    "experiment_id", "timestamp_unix", "t_s", "dt_s", "phase",
    "local_a_mm", "local_b_mm", "local_c_mm",
    "x_ref_mm", "y_ref_mm", "z_ref_mm",
    "x_tip_mm", "y_tip_mm", "z_tip_mm",
    "ex_mm", "ey_mm", "ez_mm", "e_norm_mm",
    "pid_u_x_mm", "pid_u_y_mm", "pid_u_z_mm",
    "pid_saturated", "rate_limited", "ik_reachable_cmd",
    "q_ref_z_mm", "q_ref_rot_deg", "q_ref_bend_deg",
    "q_tip_est_z_mm", "q_tip_est_rot_deg", "q_tip_est_bend_deg",
    "q_err_z_mm", "q_err_rot_deg", "q_err_bend_deg",
    "q_ff_z_mm", "q_ff_rot_deg", "q_ff_bend_deg",
    "dq_z_mm", "dq_rot_deg", "dq_bend_deg",
    "q_cmd_z_mm", "q_cmd_rot_deg", "q_cmd_bend_deg",
    "goal_tick_trans", "goal_tick_rot", "goal_tick_bend",
    "jacobian_used", "jacobian_cond_scaled",
    "jacobian_s1_scaled", "jacobian_s2_scaled", "jacobian_s3_scaled",
    "J00_raw", "J01_raw", "J02_raw",
    "J10_raw", "J11_raw", "J12_raw",
    "J20_raw", "J21_raw", "J22_raw",
]

SUMMARY_COLUMNS = [
    "experiment_id", "num_samples", "rms_error_mm", "mean_error_mm",
    "max_error_mm", "rms_ex_mm", "rms_ey_mm", "rms_ez_mm",
    "num_invalid_em", "num_large_error", "num_rate_limited",
    "num_pid_saturated", "num_ik_unreachable_cmd",
    "num_jacobian_used_samples", "aborted", "abort_code", "abort_t_s",
]


def load_rob_config():
    yaml_path = os.getenv("ROB_CENTRAL_CONFIG_PATH")
    if not yaml_path:
        yaml_path = os.path.join(
            get_package_share_directory("rob_central"),
            "config",
            "rob_config.yaml",
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def vec_norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def normalize(v, name="vector"):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise RuntimeError(f"Cannot normalize near-zero {name}")
    return v / n


def wrap_to_signed_180(angle_deg):
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and angle_deg > 0:
        return 180.0
    return wrapped


def safety_clamp_q(q):
    return {
        "z_mm": clamp(float(q["z_mm"]), Z_SAFE_MIN_MM, Z_SAFE_MAX_MM),
        "rot_deg": clamp(
            wrap_to_signed_180(float(q["rot_deg"])),
            ROT_SAFE_MIN_DEG,
            ROT_SAFE_MAX_DEG,
        ),
        "bend_deg": clamp(float(q["bend_deg"]), BEND_SAFE_MIN_DEG, BEND_SAFE_MAX_DEG),
    }


def q_dict_to_vec(q):
    return np.array([q["z_mm"], q["rot_deg"], q["bend_deg"]], dtype=float)


def dq_vec_to_dict_add(q_base, dq):
    return {
        "z_mm": q_base["z_mm"] + float(dq[0]),
        "rot_deg": q_base["rot_deg"] + float(dq[1]),
        "bend_deg": q_base["bend_deg"] + float(dq[2]),
    }


def q_step_abs(q_next, q_prev):
    return np.array([
        abs(float(q_next["z_mm"]) - float(q_prev["z_mm"])),
        abs(wrap_to_signed_180(float(q_next["rot_deg"]) - float(q_prev["rot_deg"]))),
        abs(float(q_next["bend_deg"]) - float(q_prev["bend_deg"])),
    ], dtype=float)


def rate_limit_q(q_des, q_prev):
    q_out = dict(q_des)
    limited = False
    for key, max_step in MAX_DQ_PER_SAMPLE.items():
        delta = q_out[key] - q_prev[key]
        if key == "rot_deg":
            delta = wrap_to_signed_180(delta)
        if abs(delta) > max_step:
            q_out[key] = q_prev[key] + clamp(delta, -max_step, max_step)
            limited = True
    return safety_clamp_q(q_out), limited


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


def simple_pcc_ik_raw(x_mm, y_mm, tip_z_mm):
    rho = math.hypot(x_mm, y_mm)
    bend_deg, reachable = theta_deg_from_lateral(rho, IK_BEND_LIMIT_DEG)
    rot_deg = wrap_to_signed_180(math.degrees(math.atan2(y_mm, x_mm)))
    z_mm = tip_z_mm - pcc_axial_mm(bend_deg)
    return {
        "z_mm": float(z_mm),
        "rot_deg": float(rot_deg),
        "bend_deg": float(bend_deg),
    }, reachable


def simple_pcc_ik(x_mm, y_mm, tip_z_mm):
    q, reachable = simple_pcc_ik_raw(x_mm, y_mm, tip_z_mm)
    return safety_clamp_q({
        "z_mm": q["z_mm"],
        "rot_deg": q["rot_deg"],
        "bend_deg": q["bend_deg"],
    }), reachable


def simple_pcc_fk_B(q):
    q = safety_clamp_q(q)
    phi = math.radians(wrap_to_signed_180(q["rot_deg"]))
    rho, axial, _, _ = signed_pcc_terms(q["bend_deg"])
    return np.array([
        rho * math.cos(phi),
        rho * math.sin(phi),
        q["z_mm"] + axial,
    ], dtype=float)


def point_to_polar_q(p_F, calib, fallback_rot_deg=0.0):
    p_B = p_F_to_B(p_F, calib)
    x_mm = p_B["x_mm"]
    y_mm = p_B["y_mm"]
    tip_z_mm = p_B["tip_z_mm"]
    rho = math.hypot(x_mm, y_mm)

    if rho < POLAR_ROT_EPS_MM:
        rot_deg = wrap_to_signed_180(fallback_rot_deg)
    else:
        rot_deg = wrap_to_signed_180(math.degrees(math.atan2(y_mm, x_mm)))

    bend_deg, reachable = theta_deg_from_lateral(rho, IK_BEND_LIMIT_DEG)
    z_mm = tip_z_mm - pcc_axial_mm(bend_deg)
    return safety_clamp_q({
        "z_mm": z_mm,
        "rot_deg": rot_deg,
        "bend_deg": bend_deg,
    }), reachable


def signed_pcc_terms(bend_deg):
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
    drho_dtheta = L_CC_MM * (
        theta * math.sin(theta) - (1.0 - math.cos(theta))
    ) / (theta * theta)
    daxial_dtheta = L_CC_MM * (
        theta * math.cos(theta) - math.sin(theta)
    ) / (theta * theta)
    return rho, axial, drho_dtheta, daxial_dtheta


def simple_pcc_jacobian_B(q):
    q = safety_clamp_q(q)
    phi = math.radians(wrap_to_signed_180(q["rot_deg"]))
    rho, _, drho_dtheta, daxial_dtheta = signed_pcc_terms(q["bend_deg"])
    c = math.cos(phi)
    s = math.sin(phi)
    deg_to_rad = math.pi / 180.0
    col_z = np.array([0.0, 0.0, 1.0], dtype=float)
    col_rot = np.array([-rho * s * deg_to_rad, rho * c * deg_to_rad, 0.0])
    col_bend = np.array([
        drho_dtheta * c * deg_to_rad,
        drho_dtheta * s * deg_to_rad,
        daxial_dtheta * deg_to_rad,
    ])
    return np.column_stack([col_z, col_rot, col_bend])


def rotation_B_to_F(calib):
    return np.column_stack([calib["ex_F"], calib["ey_F"], calib["ez_F"]])


def simple_pcc_jacobian_F(q, calib):
    return rotation_B_to_F(calib) @ simple_pcc_jacobian_B(q)


def p_F_to_B(p_F, calib):
    v = np.asarray(p_F, dtype=float) - calib["center_F"]
    return {
        "x_mm": calib["center_B"][0] + float(np.dot(v, calib["ex_F"])),
        "y_mm": calib["center_B"][1] + float(np.dot(v, calib["ey_F"])),
        "tip_z_mm": calib["center_B"][2] + float(np.dot(v, calib["ez_F"])),
    }


def p_B_to_F(p_B, calib):
    p_B = np.asarray(p_B, dtype=float)
    delta_B = p_B - np.asarray(calib["center_B"], dtype=float)
    return make_p_F(
        calib["center_F"],
        calib["ex_F"],
        calib["ey_F"],
        calib["ez_F"],
        delta_B[0],
        delta_B[1],
        delta_B[2],
    )


def make_p_F(center_F, ex_F, ey_F, ez_F, x_mm=0.0, y_mm=0.0, z_mm=0.0):
    return center_F + x_mm * ex_F + y_mm * ey_F + z_mm * ez_F


def delta_F_to_B(v_F, calib):
    v_F = np.asarray(v_F, dtype=float)
    return np.array([
        float(np.dot(v_F, calib["ex_F"])),
        float(np.dot(v_F, calib["ey_F"])),
        float(np.dot(v_F, calib["ez_F"])),
    ], dtype=float)


def figure8_offsets(phase, trajectory_plan):
    theta = 2.0 * math.pi * phase
    amplitudes = trajectory_plan["amplitudes_mm"]
    a = float(amplitudes[0]) * math.sin(theta)
    b = float(amplitudes[1]) * math.sin(2.0 * theta)
    c = float(amplitudes[2]) * math.sin(theta)
    return a, b, c


def em_trajectory_ref(t_s, total_time_s, calib, trajectory_plan=None):
    phase = (t_s / total_time_s * NUM_CYCLES) % 1.0
    shape = TRAJ_SHAPE.lower() if trajectory_plan is None else trajectory_plan["shape"].lower()
    if shape == "figure8_auto":
        if trajectory_plan is None:
            raise RuntimeError("figure8_auto trajectory requires a trajectory_plan")
        a, b, c = figure8_offsets(phase, trajectory_plan)
        p_B = np.asarray(trajectory_plan["center_B"], dtype=float) + np.array([a, b, c], dtype=float)
        p_ref_F = p_B_to_F(p_B, calib)
    elif shape == "circle":
        angle = 2.0 * math.pi * phase
        a = EM_TRAJ_RADIUS_MM * math.cos(angle)
        b = EM_TRAJ_RADIUS_MM * math.sin(angle)
        c = 0.0
        p_ref_F = make_p_F(calib["center_F"], calib["ex_F"], calib["ey_F"], calib["ez_F"], a, b, c)
    elif shape == "square":
        side = EM_TRAJ_SIDE_MM
        h = side / 2.0
        s = 4.0 * phase
        if s < 1.0:
            a, b = -h + side * s, -h
        elif s < 2.0:
            a, b = h, -h + side * (s - 1.0)
        elif s < 3.0:
            a, b = h - side * (s - 2.0), h
        else:
            a, b = -h, h - side * (s - 3.0)
        c = 0.0
        p_ref_F = make_p_F(calib["center_F"], calib["ex_F"], calib["ey_F"], calib["ez_F"], a, b, c)
    else:
        raise RuntimeError("TRAJ_SHAPE must be 'figure8_auto', 'circle', or 'square'")
    return p_ref_F, phase, a, b, c


def jacobian_correction_to_dq(J_raw, u_F, q_scale=IK_JAC_Q_SCALE, damping=IK_JAC_DAMPING):
    J_raw = np.asarray(J_raw, dtype=float)
    u_F = np.asarray(u_F, dtype=float)
    q_scale = np.asarray(q_scale, dtype=float)
    J_scaled = J_raw @ np.diag(q_scale)
    J_pinv_scaled = J_scaled.T @ np.linalg.inv(
        J_scaled @ J_scaled.T + (damping ** 2) * np.eye(3)
    )
    ds = J_pinv_scaled @ u_F
    return q_scale * ds


def jacobian_stats_scaled(J_raw, q_scale=IK_JAC_Q_SCALE):
    J_scaled = np.asarray(J_raw, dtype=float) @ np.diag(np.asarray(q_scale, dtype=float))
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
    u_B = delta_F_to_B(u_F, calib)
    phi = math.radians(wrap_to_signed_180(q_prev["rot_deg"]))
    radial_mm = u_B[0] * math.cos(phi) + u_B[1] * math.sin(phi)
    tangential_mm = -u_B[0] * math.sin(phi) + u_B[1] * math.cos(phi)
    return np.array([
        NOJAC_PID_Z_MM_PER_MM * u_B[2],
        NOJAC_PID_ROT_DEG_PER_MM * tangential_mm,
        NOJAC_PID_BEND_DEG_PER_MM * radial_mm,
    ], dtype=float)


class CartesianPID:
    def __init__(self):
        self.integral = np.zeros(3)
        self.prev_error = None

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error = None

    def update(self, p_ref_F, p_tip_F, dt_s):
        dt_s = max(float(dt_s), 1e-6)
        e = np.asarray(p_ref_F, dtype=float) - np.asarray(p_tip_F, dtype=float)
        self.integral += e * dt_s
        self.integral = np.clip(
            self.integral,
            -CART_PID_INTEGRAL_LIMIT_MM_S,
            CART_PID_INTEGRAL_LIMIT_MM_S,
        )
        de = np.zeros(3) if self.prev_error is None else (e - self.prev_error) / dt_s
        u = CART_PID_KP * e + CART_PID_KI * self.integral + CART_PID_KD * de
        u_clamped = np.clip(u, -CART_PID_OUTPUT_LIMIT_MM, CART_PID_OUTPUT_LIMIT_MM)
        saturated = bool(np.any(np.abs(u - u_clamped) > 1e-12))
        self.prev_error = e.copy()
        return u_clamped, e, saturated


class PseudoJointPID:
    def __init__(self):
        self.integral = np.zeros(3)
        self.prev_error = None

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error = None

    def update(self, q_ref, q_tip_est, dt_s):
        dt_s = max(float(dt_s), 1e-6)
        e = np.array([
            q_ref["z_mm"] - q_tip_est["z_mm"],
            wrap_to_signed_180(q_ref["rot_deg"] - q_tip_est["rot_deg"]),
            q_ref["bend_deg"] - q_tip_est["bend_deg"],
        ], dtype=float)

        self.integral += e * dt_s
        self.integral = np.clip(
            self.integral,
            -PSEUDO_JOINT_PID_INTEGRAL_LIMIT,
            PSEUDO_JOINT_PID_INTEGRAL_LIMIT,
        )
        de = np.zeros(3) if self.prev_error is None else (e - self.prev_error) / dt_s
        de[1] = 0.0 if self.prev_error is None else wrap_to_signed_180(e[1] - self.prev_error[1]) / dt_s
        u = (
            PSEUDO_JOINT_PID_KP * e
            + PSEUDO_JOINT_PID_KI * self.integral
            + PSEUDO_JOINT_PID_KD * de
        )
        u_clamped = np.clip(u, -PSEUDO_JOINT_PID_OUTPUT_LIMIT, PSEUDO_JOINT_PID_OUTPUT_LIMIT)
        saturated = bool(np.any(np.abs(u - u_clamped) > 1e-12))
        self.prev_error = e.copy()
        return u_clamped, e, saturated


class TransformerSweepCollectorAdaptive(Node):
    def __init__(self, active_roles=None, h5_filename="em_trajectory_tracking_ros_control.h5"):
        super().__init__("transformer_sweep_collector_adaptive")

        rt_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.config_data = load_rob_config()
        self.motor_config = {m["id"]: m for m in self.config_data["motors"]}
        self.role_to_id = {m["name"]: m["id"] for m in self.config_data["motors"]}
        self.active_roles = active_roles or list(INNER_HOME_ROLES)
        self.required_roles = list(FULL_HOME_ROLES)
        self._validate_motor_roles()

        self.h5_filename = h5_filename
        self.h5_file = h5py.File(self.h5_filename, "w")
        self._init_h5()

        self.base_matrix = None
        self.tip_matrix = None
        self.base_stamp = 0.0
        self.tip_stamp = 0.0

        self.calib = None
        self.trajectory_plan = None
        self.trajectory_duration_s = TRAJECTORY_DURATION_S
        self.trajectory_sample_count = int(math.ceil(TRAJECTORY_DURATION_S * SAMPLE_HZ))
        self.calib_points = {}
        self.sample_buffer = []
        self.current_q = self.home_q()
        self.current_goals = self.q_to_goals(self.current_q)
        self.cart_pid = CartesianPID()
        self.pseudo_joint_pid = PseudoJointPID()
        self.experiment_idx = 0
        self.metrics = None
        self.summary_rows = []
        self.sample_idx = 0
        self.msg_count = 0
        self.error_abort_count = max(1, int(SAMPLE_HZ * ERROR_ABORT_SECONDS))
        self.consecutive_large_error = 0
        self.consecutive_invalid_em = 0
        self.experiment_initial_ref_F = None
        self.experiment_start_good_count = 0
        self.em_xy_alignment_good_count = 0
        self.alignment_final_p_tip_base = np.array([np.nan, np.nan, np.nan], dtype=float)
        self.alignment_final_rho_mm = np.nan
        self.alignment_final_q = self.home_q()

        self.state = "startup"
        self.state_start = time.monotonic()
        self.pending_state_after_move = None
        self.move_label = None
        self.move_target_q = None
        self.calibration_steps = [
            ("center", np.array([CENTER_X_B_MM, CENTER_Y_B_MM, CENTER_TIP_Z_B_MM], dtype=float)),
            ("px", np.array([CENTER_X_B_MM + CALIB_X_STEP_MM, CENTER_Y_B_MM, CENTER_TIP_Z_B_MM], dtype=float)),
            ("py", np.array([CENTER_X_B_MM, CENTER_Y_B_MM + CALIB_Y_STEP_MM, CENTER_TIP_Z_B_MM], dtype=float)),
            ("pz", np.array([CENTER_X_B_MM, CENTER_Y_B_MM, CENTER_TIP_Z_B_MM + CALIB_Z_STEP_MM], dtype=float)),
        ]
        self.calibration_step_idx = 0

        self.sub_em_base = self.create_subscription(PoseStamped, BASE_TOPIC, self.on_base_feedback, rt_qos)
        self.sub_em_tip = self.create_subscription(PoseStamped, TIP_TOPIC, self.on_tip_feedback, rt_qos)
        self.cmd_pub = self.create_publisher(SyncSetPosition, COMMAND_TOPIC, rt_qos)

        self.start_full_home("startup full home", "waiting_em")

        self.timer_period = 1.0 / SAMPLE_HZ
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info(
            "ROS EM trajectory tracking initialized. "
            f"Base={BASE_TOPIC}, Tip={TIP_TOPIC}, Command={COMMAND_TOPIC}"
        )

    def _validate_motor_roles(self):
        missing = [role for role in self.required_roles if role not in self.role_to_id]
        if missing:
            raise RuntimeError(f"Missing motor role(s) in rob_config.yaml: {missing}")

    def _init_h5(self):
        self.samples_dset = self.h5_file.create_dataset(
            "samples",
            shape=(0, len(SAMPLE_COLUMNS)),
            maxshape=(None, len(SAMPLE_COLUMNS)),
            chunks=True,
            dtype="f8",
        )
        self.samples_dset.attrs["columns"] = np.array(SAMPLE_COLUMNS, dtype="S")
        self.summary_dset = self.h5_file.create_dataset(
            "summary",
            shape=(0, len(SUMMARY_COLUMNS)),
            maxshape=(None, len(SUMMARY_COLUMNS)),
            chunks=True,
            dtype="f8",
        )
        self.summary_dset.attrs["columns"] = np.array(SUMMARY_COLUMNS, dtype="S")
        self.h5_file.attrs["base_topic"] = BASE_TOPIC
        self.h5_file.attrs["tip_topic"] = TIP_TOPIC
        self.h5_file.attrs["command_topic"] = COMMAND_TOPIC
        self.h5_file.attrs["experiment_names"] = np.array(EXPERIMENTS, dtype="S")
        self.h5_file.attrs["trajectory_shape"] = TRAJ_SHAPE
        self.h5_file.attrs["trajectory_duration_s"] = np.nan
        self.h5_file.attrs["sample_hz"] = SAMPLE_HZ
        self.h5_file.attrs["fig8_min_duration_s"] = FIG8_MIN_DURATION_S
        self.h5_file.attrs["fig8_max_duration_s"] = FIG8_MAX_DURATION_S
        self.h5_file.attrs["fig8_rate_budget_fraction"] = FIG8_RATE_BUDGET_FRACTION
        self.h5_file.attrs["fig8_min_xy_bbox_mm"] = FIG8_MIN_XY_BBOX_MM
        self.h5_file.attrs["fig8_planner_status"] = "not_run"
        self.h5_file.attrs["experiment_start_settle_s"] = EXPERIMENT_START_SETTLE_S
        self.h5_file.attrs["experiment_start_error_mm"] = EXPERIMENT_START_ERROR_MM
        self.h5_file.attrs["experiment_start_stable_samples"] = EXPERIMENT_START_STABLE_SAMPLES
        self.h5_file.attrs["startup_sequence"] = "full_home_then_em_xy_alignment_then_calibration"
        self.h5_file.attrs["em_xy_alignment_tol_mm"] = EM_XY_ALIGNMENT_TOL_MM
        self.h5_file.attrs["em_xy_alignment_stable_samples"] = EM_XY_ALIGNMENT_STABLE_SAMPLES
        self.h5_file.attrs["em_xy_alignment_timeout_s"] = EM_XY_ALIGNMENT_TIMEOUT_S
        self.h5_file.attrs["em_xy_alignment_gain"] = EM_XY_ALIGNMENT_GAIN
        self.h5_file.attrs["em_xy_alignment_max_bend_deg"] = EM_XY_ALIGNMENT_MAX_BEND_DEG
        self.h5_file.attrs["motor_input_contract"] = "physical_q_to_config_input_then_map_val"
        self.h5_file.attrs["physical_q_order"] = np.array(["z_mm", "rot_deg", "bend_deg"], dtype="S")
        self.h5_file.attrs["physical_q_min"] = np.array(
            [Z_SAFE_MIN_MM, ROT_SAFE_MIN_DEG, BEND_SAFE_MIN_DEG],
            dtype=float,
        )
        self.h5_file.attrs["physical_q_max"] = np.array(
            [Z_SAFE_MAX_MM, ROT_SAFE_MAX_DEG, BEND_SAFE_MAX_DEG],
            dtype=float,
        )
        self.h5_file.attrs["home_definition"] = "full_home_config_input_zero_all_roles"
        self.h5_file.attrs["full_home_roles"] = np.array(FULL_HOME_ROLES, dtype="S")
        self.h5_file.attrs["full_home_duration_s"] = FULL_HOME_DURATION_S
        self.h5_file.attrs["experiment_0_controller"] = "polar_pseudo_joint_pid_no_jacobian"
        self.h5_file.attrs["experiment_1_controller"] = "cartesian_em_pid_analytical_jacobian"
        self.h5_file.attrs["abort_code_0"] = "none"
        self.h5_file.attrs["abort_code_1"] = "large_error"
        self.h5_file.attrs["abort_code_2"] = "invalid_em"
        self.h5_file.attrs["abort_code_3"] = "start_not_settled"
        self.h5_file.attrs["error_abort_seconds"] = ERROR_ABORT_SECONDS

    def map_val(self, val, m_id):
        cfg = self.motor_config[m_id]
        clamped = max(min(val, cfg["in_max"]), cfg["in_min"])
        ratio = (clamped - cfg["in_min"]) / (cfg["in_max"] - cfg["in_min"])
        return int(cfg["out_min"] + ratio * (cfg["out_max"] - cfg["out_min"]))

    def physical_to_config_input(self, role, physical_val, physical_min, physical_max):
        physical_val = clamp(float(physical_val), physical_min, physical_max)
        return self.physical_to_config_input_raw(role, physical_val, physical_min, physical_max)

    def physical_to_config_input_raw(self, role, physical_val, physical_min, physical_max):
        m_id = self.role_to_id[role]
        cfg = self.motor_config[m_id]
        ratio = (physical_val - physical_min) / (physical_max - physical_min)
        return cfg["in_min"] + ratio * (cfg["in_max"] - cfg["in_min"])

    def q_to_motor_inputs(self, q):
        q = safety_clamp_q(q)
        return {
            ROLE_TRANS: self.physical_to_config_input(
                ROLE_TRANS,
                q["z_mm"],
                Z_SAFE_MIN_MM,
                Z_SAFE_MAX_MM,
            ),
            ROLE_ROT: self.physical_to_config_input(
                ROLE_ROT,
                q["rot_deg"],
                ROT_SAFE_MIN_DEG,
                ROT_SAFE_MAX_DEG,
            ),
            ROLE_BEND: self.physical_to_config_input(
                ROLE_BEND,
                q["bend_deg"],
                BEND_SAFE_MIN_DEG,
                BEND_SAFE_MAX_DEG,
            ),
        }

    def msg_to_matrix(self, msg):
        q = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=float)
        p = np.array([
            msg.pose.position.x * 1000.0,
            msg.pose.position.y * 1000.0,
            msg.pose.position.z * 1000.0,
        ], dtype=float)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return None
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat(q).as_matrix()
        matrix[:3, 3] = p
        return matrix

    def on_base_feedback(self, msg):
        matrix = self.msg_to_matrix(msg)
        if matrix is not None:
            self.base_matrix = matrix
            self.base_stamp = time.monotonic()

    def on_tip_feedback(self, msg):
        matrix = self.msg_to_matrix(msg)
        if matrix is not None:
            self.tip_matrix = matrix
            self.tip_stamp = time.monotonic()

    def em_ready(self):
        now = time.monotonic()
        return (
            self.base_matrix is not None
            and self.tip_matrix is not None
            and now - self.base_stamp <= EM_TIMEOUT_S
            and now - self.tip_stamp <= EM_TIMEOUT_S
        )

    def relative_tip_position(self):
        if not self.em_ready():
            return None
        tip_h = np.ones(4)
        tip_h[:3] = self.tip_matrix[:3, 3]
        p_rel = np.linalg.inv(self.base_matrix) @ tip_h
        if not np.all(np.isfinite(p_rel[:3])):
            return None
        return p_rel[:3]

    def center_q(self):
        q, _ = simple_pcc_ik(CENTER_X_B_MM, CENTER_Y_B_MM, CENTER_TIP_Z_B_MM)
        return q

    def home_q(self):
        return safety_clamp_q({
            "z_mm": 0.5 * (Z_SAFE_MIN_MM + Z_SAFE_MAX_MM),
            "rot_deg": 0.0,
            "bend_deg": 0.0,
        })

    def q_to_goals(self, q):
        q = safety_clamp_q(q)
        motor_inputs = self.q_to_motor_inputs(q)
        return {
            self.role_to_id[ROLE_TRANS]: self.map_val(
                motor_inputs[ROLE_TRANS],
                self.role_to_id[ROLE_TRANS],
            ),
            self.role_to_id[ROLE_ROT]: self.map_val(
                motor_inputs[ROLE_ROT],
                self.role_to_id[ROLE_ROT],
            ),
            self.role_to_id[ROLE_BEND]: self.map_val(
                motor_inputs[ROLE_BEND],
                self.role_to_id[ROLE_BEND],
            ),
        }

    def q_to_motor_inputs_raw(self, q):
        return {
            ROLE_TRANS: self.physical_to_config_input_raw(
                ROLE_TRANS,
                q["z_mm"],
                Z_SAFE_MIN_MM,
                Z_SAFE_MAX_MM,
            ),
            ROLE_ROT: self.physical_to_config_input_raw(
                ROLE_ROT,
                q["rot_deg"],
                ROT_SAFE_MIN_DEG,
                ROT_SAFE_MAX_DEG,
            ),
            ROLE_BEND: self.physical_to_config_input_raw(
                ROLE_BEND,
                q["bend_deg"],
                BEND_SAFE_MIN_DEG,
                BEND_SAFE_MAX_DEG,
            ),
        }

    def q_has_config_input_margin(self, q):
        for role, val in self.q_to_motor_inputs_raw(q).items():
            cfg = self.motor_config[self.role_to_id[role]]
            if val < cfg["in_min"] - 1e-9 or val > cfg["in_max"] + 1e-9:
                return False
        return True

    def q_within_planning_limits(self, q):
        bend_hi = min(
            BEND_SAFE_MAX_DEG - FIG8_Q_MARGIN["bend_deg"],
            IK_BEND_LIMIT_DEG - FIG8_Q_MARGIN["bend_deg"],
        )
        return (
            Z_SAFE_MIN_MM + FIG8_Q_MARGIN["z_mm"] <= q["z_mm"] <= Z_SAFE_MAX_MM - FIG8_Q_MARGIN["z_mm"]
            and ROT_SAFE_MIN_DEG + FIG8_Q_MARGIN["rot_deg"] <= q["rot_deg"] <= ROT_SAFE_MAX_DEG - FIG8_Q_MARGIN["rot_deg"]
            and BEND_SAFE_MIN_DEG + FIG8_Q_MARGIN["bend_deg"] <= q["bend_deg"] <= bend_hi
            and self.q_has_config_input_margin(q)
        )

    def q_margin_vector(self, q_seq):
        z_vals = np.array([q["z_mm"] for q in q_seq], dtype=float)
        rot_vals = np.array([q["rot_deg"] for q in q_seq], dtype=float)
        bend_vals = np.array([q["bend_deg"] for q in q_seq], dtype=float)
        return np.array([
            min(np.min(z_vals - Z_SAFE_MIN_MM), np.min(Z_SAFE_MAX_MM - z_vals)),
            min(np.min(rot_vals - ROT_SAFE_MIN_DEG), np.min(ROT_SAFE_MAX_DEG - rot_vals)),
            min(np.min(bend_vals - BEND_SAFE_MIN_DEG), np.min(BEND_SAFE_MAX_DEG - bend_vals)),
        ], dtype=float)

    def figure8_q_sequence(self, center_B, amplitudes_mm, duration_s, sample_count=None):
        plan = {
            "shape": "figure8_auto",
            "center_B": np.asarray(center_B, dtype=float),
            "amplitudes_mm": np.asarray(amplitudes_mm, dtype=float),
        }
        if sample_count is None:
            sample_count = max(3, int(math.ceil(duration_s * SAMPLE_HZ)))
            full_cycle_sampling = False
        else:
            full_cycle_sampling = True
        q_seq = []
        p_seq = []
        for idx in range(sample_count):
            if full_cycle_sampling:
                phase = (idx / sample_count * NUM_CYCLES) % 1.0
            else:
                phase = (idx / (duration_s * SAMPLE_HZ) * NUM_CYCLES) % 1.0
            a, b, c = figure8_offsets(phase, plan)
            p_B = plan["center_B"] + np.array([a, b, c], dtype=float)
            q, reachable = simple_pcc_ik_raw(p_B[0], p_B[1], p_B[2])
            if not reachable or not self.q_within_planning_limits(q):
                return None, None
            q_seq.append(q)
            p_seq.append(p_B)
        return q_seq, np.vstack(p_seq)

    def figure8_rate_usage(self, center_B, amplitudes_mm, duration_s):
        q_seq, _ = self.figure8_q_sequence(center_B, amplitudes_mm, duration_s)
        if q_seq is None or len(q_seq) < 2:
            return None
        max_step = np.zeros(3, dtype=float)
        for idx in range(1, len(q_seq)):
            max_step = np.maximum(max_step, q_step_abs(q_seq[idx], q_seq[idx - 1]))
        hard_limit = np.array([
            MAX_DQ_PER_SAMPLE["z_mm"],
            MAX_DQ_PER_SAMPLE["rot_deg"],
            MAX_DQ_PER_SAMPLE["bend_deg"],
        ], dtype=float)
        usage = max_step / hard_limit
        return usage, max_step, q_seq

    def choose_figure8_duration(self, center_B, amplitudes_mm, dense_q_seq):
        dense_max_step = np.zeros(3, dtype=float)
        for idx in range(len(dense_q_seq)):
            q_next = dense_q_seq[(idx + 1) % len(dense_q_seq)]
            dense_max_step = np.maximum(dense_max_step, q_step_abs(q_next, dense_q_seq[idx]))
        hard_limit = np.array([
            MAX_DQ_PER_SAMPLE["z_mm"],
            MAX_DQ_PER_SAMPLE["rot_deg"],
            MAX_DQ_PER_SAMPLE["bend_deg"],
        ], dtype=float)
        max_dq_per_phase = dense_max_step * len(dense_q_seq)
        required_duration = float(np.max(
            max_dq_per_phase * max(NUM_CYCLES, 1) / (hard_limit * FIG8_RATE_BUDGET_FRACTION * SAMPLE_HZ)
        ))
        duration_s = max(FIG8_MIN_DURATION_S, required_duration)
        duration_s = math.ceil(duration_s / FIG8_DURATION_STEP_S) * FIG8_DURATION_STEP_S
        if duration_s > FIG8_MAX_DURATION_S:
            return None

        while duration_s <= FIG8_MAX_DURATION_S + 1e-9:
            result = self.figure8_rate_usage(center_B, amplitudes_mm, duration_s)
            if result is None:
                return None
            usage, max_step, q_seq = result
            if np.max(usage) <= FIG8_RATE_BUDGET_FRACTION + 1e-9:
                return {
                    "duration_s": float(duration_s),
                    "rate_usage": usage,
                    "max_step": max_step,
                    "q_seq": q_seq,
                }
            duration_s += FIG8_DURATION_STEP_S
        return None

    def evaluate_figure8_candidate(self, x0_mm, ax_mm):
        ay_mm = FIG8_AY_RATIO * ax_mm
        az_mm = min(FIG8_AZ_RATIO * ax_mm, FIG8_AZ_MAX_MM)
        phases = np.linspace(0.0, 1.0, FIG8_PLAN_SAMPLES, endpoint=False)
        z_base = []
        for phase in phases:
            theta = 2.0 * math.pi * phase
            x_mm = x0_mm + ax_mm * math.sin(theta)
            y_mm = ay_mm * math.sin(2.0 * theta)
            c_mm = az_mm * math.sin(theta)
            q, reachable = simple_pcc_ik_raw(x_mm, y_mm, c_mm)
            if not reachable:
                return None
            z_base.append(q["z_mm"])

        z_base = np.asarray(z_base, dtype=float)
        z_lo = Z_SAFE_MIN_MM + FIG8_Q_MARGIN["z_mm"]
        z_hi = Z_SAFE_MAX_MM - FIG8_Q_MARGIN["z_mm"]
        z0_min = z_lo - float(np.min(z_base))
        z0_max = z_hi - float(np.max(z_base))
        if z0_min > z0_max:
            return None
        z0_mm = clamp(CENTER_TIP_Z_B_MM, z0_min, z0_max)
        center_B = np.array([x0_mm, 0.0, z0_mm], dtype=float)
        amplitudes_mm = np.array([ax_mm, ay_mm, az_mm], dtype=float)

        dense_q_seq, dense_p_seq = self.figure8_q_sequence(
            center_B,
            amplitudes_mm,
            duration_s=FIG8_MAX_DURATION_S,
            sample_count=FIG8_PLAN_SAMPLES,
        )
        if dense_q_seq is None:
            return None

        duration_result = self.choose_figure8_duration(center_B, amplitudes_mm, dense_q_seq)
        if duration_result is None:
            return None

        q_margins = self.q_margin_vector(dense_q_seq)
        xy_bbox = np.ptp(dense_p_seq[:, :2], axis=0)
        xy_area = float(xy_bbox[0] * xy_bbox[1])
        start_B = center_B.copy()
        start_q, start_reachable = simple_pcc_ik_raw(start_B[0], start_B[1], start_B[2])
        if not start_reachable or not self.q_within_planning_limits(start_q):
            return None
        return {
            "shape": "figure8_auto",
            "center_B": center_B,
            "amplitudes_mm": amplitudes_mm,
            "duration_s": duration_result["duration_s"],
            "rate_usage": duration_result["rate_usage"],
            "max_step": duration_result["max_step"],
            "q_margins": q_margins,
            "xy_bbox_mm": xy_bbox,
            "xy_area_mm2": xy_area,
            "start_B": start_B,
            "start_q": start_q,
            "sampled_q": dense_q_seq,
            "sampled_p_B": dense_p_seq,
        }

    def build_auto_figure8_plan(self):
        bend_hi = min(
            BEND_SAFE_MAX_DEG - FIG8_Q_MARGIN["bend_deg"],
            IK_BEND_LIMIT_DEG - FIG8_Q_MARGIN["bend_deg"],
        )
        rho_limit = pcc_lateral_mm(bend_hi)
        ax_upper = max(FIG8_MIN_AX_MM, 0.5 * rho_limit)
        ax_values = np.linspace(ax_upper, FIG8_MIN_AX_MM, FIG8_SEARCH_AX_STEPS)
        best = None
        for ax_mm in ax_values:
            x0_low = ax_mm
            x0_high = rho_limit - ax_mm
            if x0_low > x0_high:
                continue
            for x0_mm in np.linspace(x0_low, x0_high, FIG8_SEARCH_X0_STEPS):
                candidate = self.evaluate_figure8_candidate(float(x0_mm), float(ax_mm))
                if candidate is None:
                    continue
                if (
                    best is None
                    or candidate["xy_area_mm2"] > best["xy_area_mm2"] + 1e-12
                    or (
                        abs(candidate["xy_area_mm2"] - best["xy_area_mm2"]) <= 1e-12
                        and np.min(candidate["xy_bbox_mm"]) > np.min(best["xy_bbox_mm"]) + 1e-12
                    )
                    or (
                        abs(candidate["xy_area_mm2"] - best["xy_area_mm2"]) <= 1e-12
                        and abs(np.min(candidate["xy_bbox_mm"]) - np.min(best["xy_bbox_mm"])) <= 1e-12
                        and candidate["duration_s"] < best["duration_s"]
                    )
                ):
                    best = candidate
        if best is not None:
            if np.min(best["xy_bbox_mm"]) < FIG8_MIN_XY_BBOX_MM:
                raise RuntimeError(
                    "No figure8_auto candidate reached minimum visible xy bbox "
                    f"{FIG8_MIN_XY_BBOX_MM:.3f} mm within current safe limits; "
                    f"best={best['xy_bbox_mm']}"
                )
            best["planner_status"] = "ok"
            best["rho_limit_mm"] = float(rho_limit)
            best["objective"] = "maximize_xy_bbox_within_current_safe_limits"
            return best
        raise RuntimeError("No safe figure8_auto trajectory candidate found")

    def write_trajectory_plan_h5(self):
        plan = self.trajectory_plan
        self.trajectory_duration_s = float(plan["duration_s"])
        self.trajectory_sample_count = int(math.ceil(self.trajectory_duration_s * SAMPLE_HZ))
        self.h5_file.attrs["trajectory_shape"] = plan["shape"]
        self.h5_file.attrs["trajectory_duration_s"] = self.trajectory_duration_s
        self.h5_file.attrs["trajectory_sample_count"] = self.trajectory_sample_count
        self.h5_file.attrs["fig8_center_B"] = np.asarray(plan["center_B"], dtype=float)
        self.h5_file.attrs["fig8_amplitudes_mm"] = np.asarray(plan["amplitudes_mm"], dtype=float)
        self.h5_file.attrs["fig8_duration_s"] = self.trajectory_duration_s
        self.h5_file.attrs["fig8_rate_usage"] = np.asarray(plan["rate_usage"], dtype=float)
        self.h5_file.attrs["fig8_q_margins"] = np.asarray(plan["q_margins"], dtype=float)
        self.h5_file.attrs["fig8_planner_status"] = plan["planner_status"]
        self.h5_file.attrs["fig8_rho_limit_mm"] = float(plan["rho_limit_mm"])
        self.h5_file.attrs["fig8_xy_bbox_mm"] = np.asarray(plan["xy_bbox_mm"], dtype=float)
        self.h5_file.attrs["fig8_start_B"] = np.asarray(plan["start_B"], dtype=float)
        self.h5_file.attrs["fig8_start_q"] = q_dict_to_vec(plan["start_q"])
        self.h5_file.attrs["fig8_objective"] = plan["objective"]
        group = self.h5_file.require_group("trajectory_plan")
        for key in (
            "center_B", "amplitudes_mm", "rate_usage", "max_step", "q_margins",
            "xy_bbox_mm", "start_B", "sampled_p_B",
        ):
            if key in group:
                del group[key]
            group.create_dataset(key, data=np.asarray(plan[key], dtype=float))
        if "start_q" in group:
            del group["start_q"]
        group.create_dataset("start_q", data=q_dict_to_vec(plan["start_q"]))
        if "sampled_q" in group:
            del group["sampled_q"]
        group.create_dataset(
            "sampled_q",
            data=np.vstack([q_dict_to_vec(q) for q in plan["sampled_q"]]),
        )
        group.attrs["q_order"] = np.array(["z_mm", "rot_deg", "bend_deg"], dtype="S")
        self.h5_file.flush()

    def publish_q(self, q):
        q = safety_clamp_q(q)
        goals = self.q_to_goals(q)
        msg = SyncSetPosition()
        for role in self.active_roles:
            m_id = self.role_to_id.get(role)
            if m_id is not None and m_id in goals:
                msg.id.append(m_id)
                msg.position.append(int(goals[m_id]))
        if msg.id:
            self.cmd_pub.publish(msg)
        self.current_q = q
        self.current_goals = goals
        return goals

    def role_input_to_tick(self, role, input_value):
        m_id = self.role_to_id[role]
        return self.map_val(float(input_value), m_id)

    def publish_role_inputs(self, role_inputs):
        msg = SyncSetPosition()
        goals = {}
        for role, input_value in role_inputs.items():
            m_id = self.role_to_id[role]
            tick = self.role_input_to_tick(role, input_value)
            msg.id.append(m_id)
            msg.position.append(int(tick))
            goals[m_id] = tick
        self.cmd_pub.publish(msg)
        return goals

    def full_home_inputs(self):
        return {role: 0.0 for role in FULL_HOME_ROLES}

    def publish_full_home(self):
        goals = self.publish_role_inputs(self.full_home_inputs())
        self.current_q = self.home_q()
        self.current_goals = self.q_to_goals(self.current_q)
        return goals

    def start_full_home(self, label, next_state):
        self.move_label = label
        self.pending_state_after_move = next_state
        self.state = "full_home"
        self.state_start = time.monotonic()
        goals = self.publish_full_home()
        self.get_logger().info(f"Moving to {label}: config input 0 goals={goals}")

    def start_move(self, q, label, next_state):
        self.move_target_q = safety_clamp_q(q)
        self.move_label = label
        self.pending_state_after_move = next_state
        self.state = "move"
        self.state_start = time.monotonic()
        self.publish_q(self.move_target_q)
        self.get_logger().info(f"Moving to {label}: {self.move_target_q}")

    def em_xy_alignment_target_q(self, p_tip_base):
        x_mm = float(p_tip_base[0])
        y_mm = float(p_tip_base[1])
        rho_mm = math.hypot(x_mm, y_mm)
        if rho_mm < 1e-9:
            rot_deg = self.current_q["rot_deg"]
        else:
            rot_deg = wrap_to_signed_180(math.degrees(math.atan2(-y_mm, -x_mm)))
            if rot_deg == -180.0:
                rot_deg = 180.0
        lateral_cmd_mm = min(
            EM_XY_ALIGNMENT_GAIN * rho_mm,
            pcc_lateral_mm(EM_XY_ALIGNMENT_MAX_BEND_DEG),
        )
        bend_deg, _ = theta_deg_from_lateral(lateral_cmd_mm, EM_XY_ALIGNMENT_MAX_BEND_DEG)
        return {
            "z_mm": self.home_q()["z_mm"],
            "rot_deg": rot_deg,
            "bend_deg": bend_deg,
        }

    def begin_em_xy_alignment(self):
        self.state = "em_xy_alignment"
        self.state_start = time.monotonic()
        self.em_xy_alignment_good_count = 0
        self.current_q = self.home_q()
        self.publish_q(self.current_q)
        self.get_logger().info(
            "Starting EM xy alignment: target tip/base xy rho <= "
            f"{EM_XY_ALIGNMENT_TOL_MM:.2f} mm for "
            f"{EM_XY_ALIGNMENT_STABLE_SAMPLES} samples."
        )

    def em_xy_alignment_step(self):
        p_tip_base = self.relative_tip_position()
        elapsed = time.monotonic() - self.state_start
        if p_tip_base is None:
            self.em_xy_alignment_good_count = 0
            self.publish_q(self.current_q)
        else:
            rho_mm = math.hypot(float(p_tip_base[0]), float(p_tip_base[1]))
            q_target = self.em_xy_alignment_target_q(p_tip_base)
            q_cmd, _ = rate_limit_q(safety_clamp_q(q_target), self.current_q)
            self.publish_q(q_cmd)

            if rho_mm <= EM_XY_ALIGNMENT_TOL_MM:
                self.em_xy_alignment_good_count += 1
            else:
                self.em_xy_alignment_good_count = 0

            if self.em_xy_alignment_good_count >= EM_XY_ALIGNMENT_STABLE_SAMPLES:
                self.alignment_final_p_tip_base = np.asarray(p_tip_base, dtype=float)
                self.alignment_final_rho_mm = float(rho_mm)
                self.alignment_final_q = dict(self.current_q)
                self.get_logger().info(
                    "EM xy alignment complete: "
                    f"p_tip_base={self.alignment_final_p_tip_base}, "
                    f"rho={self.alignment_final_rho_mm:.3f} mm"
                )
                self.start_move(self.center_q(), "calibration center", "calibration_sample")
                return

        if elapsed >= EM_XY_ALIGNMENT_TIMEOUT_S:
            raise RuntimeError(
                "EM xy alignment failed: tip/base xy did not settle within "
                f"{EM_XY_ALIGNMENT_TOL_MM:.2f} mm after "
                f"{EM_XY_ALIGNMENT_TIMEOUT_S:.1f}s"
            )

    def start_calibration_sample(self):
        self.sample_buffer = []
        self.state = "calibration_sample"
        self.state_start = time.monotonic()
        label, _ = self.calibration_steps[self.calibration_step_idx]
        self.get_logger().info(f"Sampling EM calibration point: {label}")

    def finish_calibration_sample(self):
        label, _ = self.calibration_steps[self.calibration_step_idx]
        if len(self.sample_buffer) < 5:
            raise RuntimeError(f"Not enough valid EM samples for calibration point {label}")
        self.calib_points[label] = np.mean(np.vstack(self.sample_buffer), axis=0)
        self.calibration_step_idx += 1

        if self.calibration_step_idx < len(self.calibration_steps):
            next_label, target_B = self.calibration_steps[self.calibration_step_idx]
            q, reachable = simple_pcc_ik(target_B[0], target_B[1], target_B[2])
            if not reachable:
                self.get_logger().warn(f"IK target clamped for calibration {next_label}")
            self.start_move(q, f"calibration {next_label}", "calibration_sample")
            return

        self.finalize_calibration()
        self.start_full_home("full home after calibration", "start_experiment")

    def finalize_calibration(self):
        center_F = self.calib_points["center"]
        px_F = self.calib_points["px"]
        py_F = self.calib_points["py"]
        pz_F = self.calib_points["pz"]

        ex_F = normalize(px_F - center_F, "ex_F")
        ez_raw = normalize(pz_F - center_F, "ez_raw")
        ey_F = normalize(np.cross(ez_raw, ex_F), "ey_F")
        if np.dot(ey_F, py_F - center_F) < 0:
            ey_F = -ey_F
        ez_F = normalize(np.cross(ex_F, ey_F), "ez_F")
        if np.dot(ez_F, pz_F - center_F) < 0:
            ez_F = -ez_F
            ey_F = -ey_F

        self.calib = {
            "center_B": np.array([CENTER_X_B_MM, CENTER_Y_B_MM, CENTER_TIP_Z_B_MM], dtype=float),
            "center_F": center_F,
            "ex_F": ex_F,
            "ey_F": ey_F,
            "ez_F": ez_F,
            "q_center": self.center_q(),
            "q_home": self.home_q(),
        }
        group = self.h5_file.require_group("calibration")
        for key, val in self.calib.items():
            if key in group:
                del group[key]
            if isinstance(val, dict):
                group.create_dataset(key, data=q_dict_to_vec(val))
            else:
                group.create_dataset(key, data=np.asarray(val, dtype=float))
        group.attrs["base_topic"] = BASE_TOPIC
        group.attrs["tip_topic"] = TIP_TOPIC
        group.attrs["q_order"] = np.array(["z_mm", "rot_deg", "bend_deg"], dtype="S")
        for key, val in (
            ("alignment_final_p_tip_base", self.alignment_final_p_tip_base),
            ("alignment_final_rho_mm", np.array([self.alignment_final_rho_mm], dtype=float)),
            ("alignment_final_q", q_dict_to_vec(self.alignment_final_q)),
        ):
            if key in group:
                del group[key]
            group.create_dataset(key, data=np.asarray(val, dtype=float))
        try:
            self.trajectory_plan = self.build_auto_figure8_plan()
            self.write_trajectory_plan_h5()
        except Exception as exc:
            self.h5_file.attrs["fig8_planner_status"] = f"failed: {exc}"
            self.h5_file.flush()
            raise
        self.get_logger().info(
            "Calibration complete. "
            f"center={center_F}, ex={ex_F}, ey={ey_F}, ez={ez_F}"
        )
        self.get_logger().info(
            "Figure-eight trajectory planned. "
            f"center_B={self.trajectory_plan['center_B']}, "
            f"amp={self.trajectory_plan['amplitudes_mm']}, "
            f"xy_bbox={self.trajectory_plan['xy_bbox_mm']}, "
            f"duration={self.trajectory_duration_s:.2f}s, "
            f"rate_usage={self.trajectory_plan['rate_usage']}"
        )

    def start_experiment(self, idx):
        if idx >= len(EXPERIMENTS):
            self.start_full_home("final full home", "done")
            return
        if self.calib is None:
            raise RuntimeError("Cannot start experiment before calibration")
        if self.trajectory_plan is None:
            raise RuntimeError("Cannot start experiment before trajectory planning")

        self.experiment_idx = idx
        self.experiment_name = EXPERIMENTS[idx]
        if self.experiment_name == "em_pid_no_jacobian_baseline":
            self.pseudo_joint_pid.reset()
        else:
            self.cart_pid.reset()
        self.consecutive_large_error = 0
        self.consecutive_invalid_em = 0
        self.metrics = {
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
            "aborted": False,
            "abort_code": ABORT_NONE,
            "abort_t_s": np.nan,
        }

        p_ref0_F, _, _, _, _ = em_trajectory_ref(
            0.0,
            self.trajectory_duration_s,
            self.calib,
            self.trajectory_plan,
        )
        self.experiment_initial_ref_F = p_ref0_F
        if self.experiment_name == "em_pid_no_jacobian_baseline":
            q0, reachable = point_to_polar_q(
                p_ref0_F,
                self.calib,
                self.current_q["rot_deg"],
            )
            if not reachable:
                self.get_logger().warn("Initial polar target was clamped")
        else:
            q0, reachable = self.ik_initial_q(p_ref0_F)
            if not reachable:
                self.get_logger().warn("Initial IK target was clamped")

        self.start_move(q0, f"{self.experiment_name} initial pose", "settle_experiment_start")

    def begin_experiment_start_settle(self):
        if self.experiment_initial_ref_F is None:
            raise RuntimeError("Cannot settle experiment start without initial reference")
        self.state = "settle_experiment_start"
        self.state_start = time.monotonic()
        self.experiment_start_good_count = 0
        self.get_logger().info(
            f"Settling at {self.experiment_name} start until EM error <= "
            f"{EXPERIMENT_START_ERROR_MM:.2f} mm or {EXPERIMENT_START_SETTLE_S:.1f}s timeout."
        )

    def settle_experiment_start_step(self):
        self.publish_q(self.move_target_q)
        p_tip_F = self.relative_tip_position()
        elapsed = time.monotonic() - self.state_start
        if self.experiment_initial_ref_F is None:
            raise RuntimeError("Missing experiment initial reference during start settle")
        if p_tip_F is None:
            self.experiment_start_good_count = 0
        else:
            e_norm = vec_norm(np.asarray(self.experiment_initial_ref_F, dtype=float) - p_tip_F)
            if e_norm <= EXPERIMENT_START_ERROR_MM:
                self.experiment_start_good_count += 1
            else:
                self.experiment_start_good_count = 0

            if self.experiment_start_good_count >= EXPERIMENT_START_STABLE_SAMPLES:
                self.get_logger().info(
                    f"Experiment start settled: e={e_norm:.3f} mm, "
                    f"stable_samples={self.experiment_start_good_count}"
                )
                self.begin_run_experiment()
                return

        if elapsed >= EXPERIMENT_START_SETTLE_S:
            abort_t_s = float(elapsed)
            if p_tip_F is None:
                self.get_logger().warn("Experiment start settle timed out without valid EM; aborting experiment.")
            else:
                self.get_logger().warn(
                    f"Experiment start settle timed out: e={e_norm:.3f} mm; aborting experiment."
                )
            self.finish_experiment(ABORT_START_NOT_SETTLED, abort_t_s)

    def begin_run_experiment(self):
        self.run_start = time.monotonic()
        self.prev_sample_time = self.run_start
        self.local_sample_idx = 0
        self.state = "run_experiment"
        self.state_start = self.run_start
        self.get_logger().info(f"Running experiment: {self.experiment_name}")

    def ik_initial_q(self, p_ref_F):
        p_B = p_F_to_B(p_ref_F, self.calib)
        return simple_pcc_ik(p_B["x_mm"], p_B["y_mm"], p_B["tip_z_mm"])

    def baseline_update(self, p_ref_F, p_tip_F, dt_s):
        q_ref, ref_reachable = point_to_polar_q(
            p_ref_F,
            self.calib,
            self.current_q["rot_deg"],
        )
        q_tip_est, _ = point_to_polar_q(
            p_tip_F,
            self.calib,
            self.current_q["rot_deg"],
        )
        dq_pid, q_err, sat = self.pseudo_joint_pid.update(q_ref, q_tip_est, dt_s)
        q_des = dq_vec_to_dict_add(q_ref, dq_pid)
        q_cmd, rate_limited = rate_limit_q(safety_clamp_q(q_des), self.current_q)
        e_F = np.asarray(p_ref_F, dtype=float) - np.asarray(p_tip_F, dtype=float)
        return q_cmd, {
            "u_F": np.array([np.nan, np.nan, np.nan], dtype=float),
            "e_F": e_F,
            "pid_saturated": sat,
            "rate_limited": rate_limited,
            "ik_reachable_cmd": bool(ref_reachable),
            "dq": dq_pid,
            "q_ref": q_ref,
            "q_tip_est": q_tip_est,
            "q_err": q_err,
            "q_ff": {"z_mm": np.nan, "rot_deg": np.nan, "bend_deg": np.nan},
            "jacobian_used": False,
            "jacobian_cond_scaled": np.nan,
            "jacobian_svals_scaled": np.array([np.nan, np.nan, np.nan], dtype=float),
            "J_raw": np.full((3, 3), np.nan, dtype=float),
        }

    def ik_jacobian_update(self, p_ref_F, p_tip_F, dt_s):
        p_ref_B = p_F_to_B(p_ref_F, self.calib)
        q_ff, reachable_ff = simple_pcc_ik(
            p_ref_B["x_mm"],
            p_ref_B["y_mm"],
            p_ref_B["tip_z_mm"],
        )
        u_F, e_F, sat = self.cart_pid.update(p_ref_F, p_tip_F, dt_s)
        J_raw = simple_pcc_jacobian_F(q_ff, self.calib)
        dq_fb = jacobian_correction_to_dq(J_raw, u_F)
        jacobian_cond, jacobian_svals = jacobian_stats_scaled(J_raw)
        q_des = dq_vec_to_dict_add(q_ff, dq_fb)
        q_cmd, rate_limited = rate_limit_q(safety_clamp_q(q_des), self.current_q)
        return q_cmd, {
            "u_F": u_F,
            "e_F": e_F,
            "pid_saturated": sat,
            "rate_limited": rate_limited,
            "ik_reachable_cmd": bool(reachable_ff),
            "dq": dq_fb,
            "q_ref": {"z_mm": np.nan, "rot_deg": np.nan, "bend_deg": np.nan},
            "q_tip_est": {"z_mm": np.nan, "rot_deg": np.nan, "bend_deg": np.nan},
            "q_err": np.array([np.nan, np.nan, np.nan], dtype=float),
            "q_ff": q_ff,
            "jacobian_used": True,
            "jacobian_cond_scaled": jacobian_cond,
            "jacobian_svals_scaled": jacobian_svals,
            "J_raw": J_raw,
        }

    def append_sample(self, row):
        self.samples_dset.resize((self.sample_idx + 1, len(SAMPLE_COLUMNS)))
        self.samples_dset[self.sample_idx] = row
        self.sample_idx += 1

    def append_summary(self, row):
        idx = len(self.summary_rows)
        self.summary_rows.append(row)
        self.summary_dset.resize((idx + 1, len(SUMMARY_COLUMNS)))
        self.summary_dset[idx] = row
        self.h5_file.flush()

    def run_experiment_step(self):
        now = time.monotonic()
        if self.trajectory_plan is None:
            raise RuntimeError("Cannot run experiment without trajectory_plan")
        if self.local_sample_idx >= self.trajectory_sample_count:
            self.finish_experiment()
            return
        t_s = self.local_sample_idx / SAMPLE_HZ

        dt_s = now - self.prev_sample_time
        self.prev_sample_time = now
        if dt_s <= 1e-6 or dt_s > 1.0:
            dt_s = 1.0 / SAMPLE_HZ

        p_tip_F = self.relative_tip_position()
        if p_tip_F is None:
            self.metrics["num_invalid_em"] += 1
            self.consecutive_invalid_em += 1
            self.consecutive_large_error = 0
            if self.consecutive_invalid_em >= self.error_abort_count:
                self.finish_experiment(ABORT_INVALID_EM, t_s)
                return
            self.publish_q(self.current_q)
            return

        p_ref_F, phase, local_a_mm, local_b_mm, local_c_mm = em_trajectory_ref(
            t_s,
            self.trajectory_duration_s,
            self.calib,
            self.trajectory_plan,
        )
        raw_error = p_ref_F - p_tip_F
        raw_error_norm = vec_norm(raw_error)
        if raw_error_norm > MAX_EM_ERROR_MM:
            self.metrics["num_large_error"] += 1
            self.consecutive_large_error += 1
            self.consecutive_invalid_em = 0
            self.get_logger().warn(
                f"Large EM error {raw_error_norm:.3f} mm; holding command",
                throttle_duration_sec=1.0,
            )
            if self.consecutive_large_error >= self.error_abort_count:
                self.finish_experiment(ABORT_LARGE_ERROR, t_s)
                return
            self.publish_q(self.current_q)
            return

        self.consecutive_invalid_em = 0
        self.consecutive_large_error = 0

        if self.experiment_name == "em_pid_no_jacobian_baseline":
            q_cmd, dbg = self.baseline_update(p_ref_F, p_tip_F, dt_s)
        else:
            q_cmd, dbg = self.ik_jacobian_update(p_ref_F, p_tip_F, dt_s)

        goals = self.publish_q(q_cmd)
        e_F = dbg["e_F"]
        e_norm = vec_norm(e_F)

        self.metrics["errors"].append(e_norm)
        self.metrics["ex"].append(float(e_F[0]))
        self.metrics["ey"].append(float(e_F[1]))
        self.metrics["ez"].append(float(e_F[2]))
        self.metrics["num_samples"] += 1
        self.metrics["num_rate_limited"] += int(dbg["rate_limited"])
        self.metrics["num_pid_saturated"] += int(dbg["pid_saturated"])
        self.metrics["num_ik_unreachable_cmd"] += int(not dbg["ik_reachable_cmd"])
        self.metrics["num_jacobian_used_samples"] += int(dbg["jacobian_used"])

        q_ff = dbg["q_ff"]
        q_ref = dbg["q_ref"]
        q_tip_est = dbg["q_tip_est"]
        q_err = dbg["q_err"]
        dq = dbg["dq"]
        J_raw = dbg["J_raw"]
        svals = dbg["jacobian_svals_scaled"]
        row = np.array([
            float(self.experiment_idx),
            time.time(),
            t_s,
            dt_s,
            phase,
            local_a_mm,
            local_b_mm,
            local_c_mm,
            p_ref_F[0], p_ref_F[1], p_ref_F[2],
            p_tip_F[0], p_tip_F[1], p_tip_F[2],
            e_F[0], e_F[1], e_F[2], e_norm,
            dbg["u_F"][0], dbg["u_F"][1], dbg["u_F"][2],
            float(dbg["pid_saturated"]),
            float(dbg["rate_limited"]),
            float(dbg["ik_reachable_cmd"]),
            q_ref["z_mm"], q_ref["rot_deg"], q_ref["bend_deg"],
            q_tip_est["z_mm"], q_tip_est["rot_deg"], q_tip_est["bend_deg"],
            q_err[0], q_err[1], q_err[2],
            q_ff["z_mm"], q_ff["rot_deg"], q_ff["bend_deg"],
            dq[0], dq[1], dq[2],
            q_cmd["z_mm"], q_cmd["rot_deg"], q_cmd["bend_deg"],
            float(goals[self.role_to_id[ROLE_TRANS]]),
            float(goals[self.role_to_id[ROLE_ROT]]),
            float(goals[self.role_to_id[ROLE_BEND]]),
            float(dbg["jacobian_used"]),
            dbg["jacobian_cond_scaled"],
            svals[0], svals[1], svals[2],
            J_raw[0, 0], J_raw[0, 1], J_raw[0, 2],
            J_raw[1, 0], J_raw[1, 1], J_raw[1, 2],
            J_raw[2, 0], J_raw[2, 1], J_raw[2, 2],
        ], dtype=float)
        self.append_sample(row)
        self.local_sample_idx += 1

        self.msg_count += 1
        if self.msg_count % max(1, int(SAMPLE_HZ / 2.0)) == 0:
            self.refresh_dashboard(t_s, e_norm, q_cmd)

    def finish_experiment(self, abort_code=ABORT_NONE, abort_t_s=np.nan):
        if abort_code != ABORT_NONE:
            self.metrics["aborted"] = True
            self.metrics["abort_code"] = abort_code
            self.metrics["abort_t_s"] = float(abort_t_s)
        summary = self.summarize_metrics()
        self.append_summary(summary)
        name = self.experiment_name
        if abort_code == ABORT_NONE:
            self.get_logger().info(
                f"Finished {name}: rms={summary[2]:.3f} mm, "
                f"samples={int(summary[1])}"
            )
        else:
            self.get_logger().warn(
                f"Aborted {name}: code={abort_code}, t={abort_t_s:.2f}s, "
                f"samples={int(summary[1])}"
            )
        self.start_full_home(f"return full home after {name}", "next_experiment")

    def summarize_metrics(self):
        m = self.metrics
        err = np.asarray(m["errors"], dtype=float)
        ex = np.asarray(m["ex"], dtype=float)
        ey = np.asarray(m["ey"], dtype=float)
        ez = np.asarray(m["ez"], dtype=float)
        if len(err) == 0:
            vals = [np.nan] * 6
            num_samples = 0
        else:
            vals = [
                float(np.sqrt(np.mean(err ** 2))),
                float(np.mean(err)),
                float(np.max(err)),
                float(np.sqrt(np.mean(ex ** 2))),
                float(np.sqrt(np.mean(ey ** 2))),
                float(np.sqrt(np.mean(ez ** 2))),
            ]
            num_samples = len(err)
        return np.array([
            float(self.experiment_idx),
            float(num_samples),
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
            float(m["num_invalid_em"]),
            float(m["num_large_error"]),
            float(m["num_rate_limited"]),
            float(m["num_pid_saturated"]),
            float(m["num_ik_unreachable_cmd"]),
            float(m["num_jacobian_used_samples"]),
            float(m["aborted"]),
            float(m["abort_code"]),
            float(m["abort_t_s"]),
        ], dtype=float)

    def refresh_dashboard(self, t_s, e_norm, q_cmd):
        clear_line = "\033[K"
        status = (
            f"\r[{self.experiment_idx + 1}/{len(EXPERIMENTS)} {self.experiment_name}] "
            f"t={t_s:5.1f}s e={e_norm:5.3f}mm "
            f"q=({q_cmd['z_mm']:.2f}, {q_cmd['rot_deg']:.2f}, {q_cmd['bend_deg']:.2f}) "
            f"rows={self.sample_idx}{clear_line}"
        )
        sys.stdout.write(status)
        sys.stdout.flush()

    def control_loop(self):
        try:
            if self.state == "waiting_em":
                if self.em_ready():
                    self.get_logger().info("Both base and tip EM channels are live.")
                    self.begin_em_xy_alignment()
                else:
                    self.msg_count += 1
                    if self.msg_count % int(SAMPLE_HZ) == 0:
                        self.get_logger().info(
                            f"Waiting for EM: base={self.base_matrix is not None}, "
                            f"tip={self.tip_matrix is not None}",
                            throttle_duration_sec=1.0,
                        )
                return

            if self.state == "em_xy_alignment":
                self.em_xy_alignment_step()
                return

            if self.state == "full_home":
                self.publish_full_home()
                if time.monotonic() - self.state_start >= FULL_HOME_DURATION_S:
                    next_state = self.pending_state_after_move
                    if next_state == "waiting_em":
                        self.state = "waiting_em"
                        self.state_start = time.monotonic()
                        self.get_logger().info("Startup full home reached. Waiting for EM channels.")
                    elif next_state == "start_experiment":
                        self.start_experiment(0)
                    elif next_state == "next_experiment":
                        self.start_experiment(self.experiment_idx + 1)
                    elif next_state == "done":
                        self.stop_and_clean_exit()
                    else:
                        raise RuntimeError(f"Unknown post-full-home state: {next_state}")
                return

            if self.state == "move":
                self.publish_q(self.move_target_q)
                if time.monotonic() - self.state_start >= MOVE_SETTLE_S:
                    next_state = self.pending_state_after_move
                    if next_state == "waiting_em":
                        self.state = "waiting_em"
                        self.state_start = time.monotonic()
                        self.get_logger().info("Startup home reached. Waiting for EM channels.")
                    elif next_state == "calibration_sample":
                        self.start_calibration_sample()
                    elif next_state == "start_experiment":
                        self.start_experiment(0)
                    elif next_state == "settle_experiment_start":
                        self.begin_experiment_start_settle()
                    elif next_state == "next_experiment":
                        self.start_experiment(self.experiment_idx + 1)
                    elif next_state == "done":
                        self.stop_and_clean_exit()
                    else:
                        raise RuntimeError(f"Unknown post-move state: {next_state}")
                return

            if self.state == "calibration_sample":
                self.publish_q(self.move_target_q)
                p_tip = self.relative_tip_position()
                if p_tip is not None:
                    self.sample_buffer.append(p_tip)
                if time.monotonic() - self.state_start >= CALIB_SAMPLE_DURATION_S:
                    self.finish_calibration_sample()
                return

            if self.state == "settle_experiment_start":
                self.settle_experiment_start_step()
                return

            if self.state == "run_experiment":
                self.run_experiment_step()
                return

            if self.state == "done":
                return

            raise RuntimeError(f"Unknown state: {self.state}")
        except Exception as exc:
            self.get_logger().error(f"Control loop failed: {exc}")
            self.stop_and_clean_exit()

    def stop_and_clean_exit(self):
        if self.state != "done":
            self.state = "done"
            try:
                self.publish_full_home()
            except Exception:
                pass
        try:
            self.timer.cancel()
        except Exception:
            pass
        if self.h5_file:
            try:
                self.h5_file.flush()
                self.h5_file.close()
            except Exception:
                pass
        self.get_logger().info("Trajectory tracking HDF5 finalized.")
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        if self.h5_file:
            try:
                self.h5_file.flush()
                self.h5_file.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TransformerSweepCollectorAdaptive(
        active_roles=list(INNER_HOME_ROLES),
        h5_filename="em_trajectory_tracking_ros_control.h5",
    )

    try:
        if rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        sys.stdout.write("\nTrajectory tracking interrupted.\n")
        node.stop_and_clean_exit()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
