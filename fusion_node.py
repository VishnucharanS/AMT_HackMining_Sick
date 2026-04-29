import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import time


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        self.declare_parameter("mode", "fusion")
        self.mode = self.get_parameter("mode").value

        # Inputs
        self.ml = None
        self.rule = None

        # (Optional) ML confidence if you add later
        self.ml_conf = 0.5  

        # Subscribers
        self.create_subscription(Int32, '/multimodal/prediction', self.ml_cb, 10)
        self.create_subscription(Int32, '/trafic_light_color_raw', self.rule_cb, 10)

        # Publisher
        self.pub = self.create_publisher(Int32, '/trafic_light_color', 10)

        # Severity mapping
        # 1 = CRITICAL, 3 = REDUCED, 2 = NORMAL
        self.severity = {1: 3, 3: 2, 2: 1}
        self.label_map = {
            1: "CRITICAL",
            3: "REDUCED",
            2: "CLEAN"
        }
        self.ml_time = None
        self.rule_time = None
        self.TIMEOUT = 1.0  # seconds

    

    def ml_cb(self, msg):
        self.ml = msg.data
        self.ml_time = time.time()
        self.compute()

    def rule_cb(self, msg):
        self.rule = msg.data
        self.rule_time = time.time()
        self.compute()

    def compute(self):
        if self.ml is None or self.rule is None:
            return
        now = time.time()

        # ❗ Drop stale data
        if (now - self.ml_time > self.TIMEOUT):
            self.get_logger().warn("ML data stale")
            return

        if (now - self.rule_time > self.TIMEOUT):
            self.get_logger().warn("Rule data stale")
            return

        # =========================
        # MODE SWITCH
        # =========================
        if self.mode == "ml_only":
            final = self.ml

        elif self.mode == "rule_only":
            final = self.rule

        else:
            # =========================
            # 🔴 STAGE 1: HARD SAFETY OVERRIDE
            # =========================
            if self.rule == 1:  # CRITICAL
                final = 1
                self.publish(final)
                return

            if self.ml == 1 and self.rule != 2:
                final = 1
                self.publish(final)
                return

            # =========================
            # 🟡 STAGE 2: WEIGHTED FUSION
            # =========================
            ml_score = self.severity.get(self.ml, 1)
            rule_score = self.severity.get(self.rule, 1)

            # Fixed weights (you chose this 👍)
            w_rule = 0.6
            w_ml = 0.4

            # OPTIONAL: dynamic weighting (uncomment if needed)
            # w_ml = 0.4 * self.ml_conf
            # w_rule = 1 - w_ml

            fused_score = w_rule * rule_score + w_ml * ml_score

            # =========================
            # 🔵 STAGE 3: BIAS CORRECTION
            # =========================
            if self.ml != self.rule:
                fused_score += 0.2 * (rule_score - ml_score)

            # =========================
            # 🔁 FINAL DECISION
            # =========================
            if fused_score >= 2.5:
                final = 1  # CRITICAL
            elif fused_score >= 1.5:
                final = 3  # REDUCED
            else:
                final = 2  # NORMAL

        self.publish(final)

    def publish(self, final):
        msg = Int32()
        msg.data = final
        self.pub.publish(msg)

        self.get_logger().info(
            f"ML:{self.label_map.get(self.ml)} "
            f"RULE:{self.label_map.get(self.rule)} "
            f"FINAL:{self.label_map.get(final)} "
            f"(mode={self.mode})"
        )

def main():
    rclpy.init()
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


#self.traffic_pub = self.create_publisher(Int32, '/trafic_light_color_raw', 10) change in utkarsh's code
#ros2 param set /fusion_node mode ml_only
#ros2 param set /fusion_node mode rule_only
#ros2 param set /fusion_node mode fusion