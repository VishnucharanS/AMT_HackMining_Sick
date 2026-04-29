import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

#from ros2_node_hack import lidar
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge

import numpy as np
import cv2
import struct
import random

import torch
import torch.nn as nn
import torchvision.models as models
from std_msgs import msg
from std_msgs.msg import Int32
from collections import Counter, deque
from collections import Counter
import matplotlib.pyplot as plt

qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.BEST_EFFORT

def decode_pc2(msg):
    fields = {f.name: f for f in msg.fields}
    if 'x' not in fields:
        return np.zeros((0,3)), np.array([])

    ps = msg.point_step
    data = bytes(msg.data)
    n = msg.width * msg.height

    ox = fields['x'].offset
    oy = fields['y'].offset
    oz = fields['z'].offset
    oi = fields['intensity'].offset if 'intensity' in fields else None

    xyz_list = []
    int_list = []

    for i in range(n):
        base = i * ps
        x = struct.unpack_from('<f', data, base + ox)[0]
        y = struct.unpack_from('<f', data, base + oy)[0]
        z = struct.unpack_from('<f', data, base + oz)[0]

        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            xyz_list.append((x, y, z))
            if oi is not None:
                int_list.append(struct.unpack_from('<f', data, base + oi)[0])

    xyz = np.array(xyz_list, dtype=np.float32)
    ints = np.array(int_list, dtype=np.float32) if int_list else np.ones(len(xyz))*128

    return xyz, ints

def decode_image(msg, bridge):
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    img = cv2.resize(img, (224,224))
    return img



def augment_image(img):
    # brightness
    if random.random() < 0.5:
        alpha = 0.8 + 0.4 * random.random()
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    # blur (simulate dirt)
    if random.random() < 0.3:
        k = random.choice([3,5])
        img = cv2.GaussianBlur(img, (k,k), 0)

    return img

def augment_lidar(xyz, intensity):
    # simulate dust → reduce intensity
    if np.random.rand() < 0.3:
        intensity = intensity * np.random.uniform(0.5, 0.9)

    # random dropout (occlusion)
    if np.random.rand() < 0.3:
        mask = np.random.rand(len(xyz)) > 0.1
        xyz = xyz[mask]
        intensity = intensity[mask]

    return xyz, intensity

def pointcloud_to_range_image(xyz, intensities, H=64, W=512):
    if len(xyz) == 0:
        return np.zeros((H, W, 3), dtype=np.float32)

    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    r = np.linalg.norm(xyz, axis=1)

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / (r + 1e-6))   

    u = ((azimuth + np.pi) / (2*np.pi) * W).astype(int)
    v = ((elevation + np.pi/4) / (np.pi/2) * H).astype(int)

    u = np.clip(u, 0, W-1)
    v = np.clip(v, 0, H-1)

    img_range = np.zeros((H, W), dtype=np.float32)
    img_int = np.zeros((H, W), dtype=np.float32)

    img_range[v, u] = r
    img_int[v, u] = intensities if len(intensities) == len(r) else r

    final = np.stack([img_range, img_int, img_range], axis=-1)

    return final

class MultiModalResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_net = models.resnet18(weights="DEFAULT")
        self.cam_net = models.resnet18(weights="DEFAULT")




        for p in self.lidar_net.parameters():
            p.requires_grad = False
        for p in self.cam_net.parameters():
            p.requires_grad = False

        self.lidar_net.fc = nn.Identity()
        self.cam_net.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, lidar, cam):
        f1 = self.lidar_net(lidar)
        f2 = self.cam_net(cam)
        return self.fc(torch.cat([f1, f2], dim=1))

class MultiModalNode(Node):

    def __init__(self):
        super().__init__('multimodal_detector')

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = MultiModalResNet().to(self.device)
        self.model.load_state_dict(torch.load("model2.pth", map_location=self.device))
        self.model.eval()
        self.gradcam = GradCAM(self.model, self.model.cam_net.layer4[-1])
        self.gradlidar = GradCAM(self.model, self.model.lidar_net.layer4[-1])
        self.history = deque(maxlen=35)   # try 50 first
        self.conf_history = deque(maxlen=35)
        self.lidar_msg = None
        self.camera_msg = None
        self.create_subscription(
            PointCloud2,
            '/lidar/cloud/device_id47',
            self.lidar_callback,
            qos
        )

        self.create_subscription(
            Image,
            '/visionary2/bgr/device_id4',
            self.camera_callback,
            qos
        )
        self.pred_pub = self.create_publisher(Int32, '/multimodal/prediction', 10)
        self.timer = self.create_timer(0.2, self.process)  

    def lidar_callback(self, msg):
        self.lidar_msg = msg

    def camera_callback(self, msg):
        self.camera_msg = msg

    
    def process(self):

        if self.lidar_msg is None:
            return

        if self.camera_msg is None:
            return

        # Decode
        xyz, ints = decode_pc2(self.lidar_msg)
        img = decode_image(self.camera_msg, self.bridge)

        # Convert
        range_img = pointcloud_to_range_image(xyz, ints)

        # Resize
        range_img = cv2.resize(range_img, (224,224))
        img = cv2.resize(img, (224,224))

        # Normalize input
        
        range_img = (range_img / 100.0 * 255).astype(np.uint8)
        range_img = cv2.flip(range_img, 0)
        range_img = range_img.astype(np.uint8)

        # To tensor
        lidar_tensor = torch.from_numpy(range_img).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
        cam_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)

            lidar_tensor = (lidar_tensor - mean) / std
            cam_tensor = (cam_tensor - mean) / std

            output = self.model(lidar_tensor, cam_tensor)

            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            
            confidence = probs[0][pred_class].item()
        lidar_tensor.requires_grad_()
        cam_tensor.requires_grad_()
        cam_map = self.gradcam.generate(lidar_tensor, cam_tensor, pred_class)
        lidar_map = self.gradlidar.generate(lidar_tensor, cam_tensor, pred_class)
        # Store history
        self.history.append(pred_class)
        self.conf_history.append(confidence)

        # Wait for buffer
        if len(self.history) < self.history.maxlen:
            return

        # Weighted voting
        scores = {0: 0.0, 1: 0.0, 2: 0.0}
        for cls, conf in zip(self.history, self.conf_history):
            scores[cls] += conf


        stable_class = Counter(self.history).most_common(1)[0][0]
        avg_conf = sum(self.conf_history) / len(self.conf_history)
        classes = ["clean", "caution", "dirty"]

        msg = Int32()

        if avg_conf > 0.4:
            if stable_class == 0:
                msg.data = 2
            elif stable_class == 1:
                msg.data = 3
            else:
                msg.data = 1
        else:
            msg.data = 0

        self.pred_pub.publish(msg)
        img_np = cam_tensor[0].permute(1,2,0).detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        heatmap_cam = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap_lidar = cv2.applyColorMap(np.uint8(255 * lidar_map), cv2.COLORMAP_JET)
        overlay_cam = cv2.addWeighted(img_np, 0.6, heatmap_cam, 0.4, 0)
        overlay_lidar = cv2.addWeighted(img_np, 0.6, heatmap_lidar, 0.4, 0)
        print(f"Stable: {msg.data} | Class: {classes[stable_class]} | AvgConf: {avg_conf:.3f}")
    
        combined = np.hstack([
            cv2.resize(img, (224,224)),
            cv2.resize(range_img, (224,224)),
            overlay_cam,
            overlay_lidar
        ])

        cv2.imshow("Camera | LiDAR | GradCAM", combined)
        cv2.waitKey(1)

        

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, lidar_tensor, cam_tensor, class_idx):
        self.gradients = None
        self.activations = None
        self.model.zero_grad()
        
        output = self.model(lidar_tensor, cam_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        if self.gradients is None:
            print("Gradients not captured!")
            return np.zeros((224,224))

        return cam
    
def main():
    rclpy.init()
    node = MultiModalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()