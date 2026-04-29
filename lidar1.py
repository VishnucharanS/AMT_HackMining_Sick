import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from collections import deque

class LidarNode(Node):
    def __init__(self):
        super().__init__("lidar_node")
        self.subscription = self.create_subscription(
            PointCloud2,
            "/lidar/cloud/device_id47",
            self.callback,
            10
        )
        self.history = deque(maxlen=50)
        self.smoothed_health = 100
        
        

    def callback(self, msg):
        points = []

        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "intensity", "range", "reflectivity"),
            skip_nans=True
        ):
            points.append([
                p[0],  # x
                p[1],  # y
                p[2],  # z
                p[3],  # intensity
                p[4],  # range
                p[5]   # reflectivity
            ])

        points = np.array(points)

        #print("Frame:", points.shape)  # should be (N, 6)
        features = self.extract_features(points)
        #print(features)

        if features is not None:
            self.history.append([
                features["num_points"],
                features["mean_range"],
                features["std_range"],
                features["mean_intensity"],
                features["mean_reflectivity"]
    ])
            if len(self.history) < 20:
                print("Collecting baseline...")
                return
            
            hist_array = np.array(self.history)

            mean = np.mean(hist_array, axis=0)
            std = np.std(hist_array, axis=0) + 1e-6

            current = np.array(self.history[-1])

            health = self.compute_health(current, mean, std)

            self.smoothed_health = 0.8 * self.smoothed_health + 0.2 * health

            state = self.classify(self.smoothed_health)
            
            print(f"Health: {health:.2f} | State: {state}")
            
            

    def compute_health(self, current, mean, std):
        z = np.abs((current - mean) / std)
        anomaly = np.mean(z)

        health = max(0, 100 - anomaly * 20)
        return health
    
        
    def classify(self, score):
        if score > 75:
            return "NORMAL"
        elif score > 40:
            return "DEGRADED"
        else:
            return "CRITICAL"    
        
    def extract_features(self,points):
        if len(points) == 0:
            return None

        num_points = len(points)

        mean_range = np.mean(points[:, 4])
        std_range = np.std(points[:, 4])

        mean_intensity = np.mean(points[:, 3])
        mean_reflectivity = np.mean(points[:, 5])

        return {
            "num_points": num_points,
            "mean_range": mean_range,
            "std_range": std_range,
            "mean_intensity": mean_intensity,
            "mean_reflectivity": mean_reflectivity
        }

    
def main():
    rclpy.init()
    node = LidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()

