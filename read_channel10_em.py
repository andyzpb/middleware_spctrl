#!/usr/bin/env python3
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


DEFAULT_TOPIC = "/ndi/channel_10/pose"


class Channel10EMReader(Node):
    def __init__(self, topic, print_hz):
        super().__init__("channel10_em_reader")
        self.topic = topic
        self.print_period_s = 1.0 / print_hz if print_hz > 0.0 else 0.0
        self.last_print_s = 0.0
        self.count = 0

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(PoseStamped, self.topic, self.on_pose, qos)
        self.get_logger().info(f"Reading EM PoseStamped from {self.topic}")

    def on_pose(self, msg):
        now = time.monotonic()
        if self.print_period_s > 0.0 and now - self.last_print_s < self.print_period_s:
            return
        self.last_print_s = now
        self.count += 1

        p = msg.pose.position
        q = msg.pose.orientation
        x_mm = p.x * 1000.0
        y_mm = p.y * 1000.0
        z_mm = p.z * 1000.0
        q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        print(
            f"[{self.count:06d}] stamp={stamp:.6f} "
            f"pos_mm=({x_mm:+.3f}, {y_mm:+.3f}, {z_mm:+.3f}) "
            f"quat=({q.x:+.6f}, {q.y:+.6f}, {q.z:+.6f}, {q.w:+.6f}) "
            f"|q|={q_norm:.6f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Read one Aurora EM PoseStamped channel.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"Pose topic, default: {DEFAULT_TOPIC}")
    parser.add_argument("--print-hz", type=float, default=10.0, help="Console print frequency; <=0 prints every message.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_hz < 0.0:
        raise ValueError("--print-hz must be >= 0")

    rclpy.init()
    node = Channel10EMReader(args.topic, args.print_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nChannel 10 EM reader interrupted.", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
