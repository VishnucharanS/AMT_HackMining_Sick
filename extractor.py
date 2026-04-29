import os
import cv2
import numpy as np
import struct
from cv_bridge import CvBridge

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# -------- CONFIG --------
root_bag_dir = "bags"
output_dir = "dataset"

lidar_topic = "/lidar/cloud/device_id47"
cam_topic = "/visionary2/bgr/device_id4"

label_map = {
    "clean": "clean",
    "caution": "caution",
    "dirty": "dirty"
}

bridge = CvBridge()
os.makedirs(output_dir, exist_ok=True)

# -------- FUNCTIONS --------
def decode_pc2(msg):
    fields = {f.name: f for f in msg.fields}

    ps = msg.point_step
    data = bytes(msg.data)
    n = msg.width * msg.height

    ox = fields['x'].offset
    oy = fields['y'].offset
    oz = fields['z'].offset
    oi = fields['intensity'].offset if 'intensity' in fields else None

    xyz = []
    intensity = []

    for i in range(n):
        base = i * ps
        x = struct.unpack_from('<f', data, base + ox)[0]
        y = struct.unpack_from('<f', data, base + oy)[0]
        z = struct.unpack_from('<f', data, base + oz)[0]

        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            xyz.append((x, y, z))

            if oi is not None:
                intensity.append(struct.unpack_from('<f', data, base + oi)[0])

    xyz = np.array(xyz, dtype=np.float32)
    intensity = np.array(intensity, dtype=np.float32) if intensity else None

    return xyz, intensity


def pointcloud_to_range_image(xyz, intensity=None, H=64, W=512):
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
    img_int   = np.zeros((H, W), dtype=np.float32)

    img_range[v, u] = r

    if intensity is not None and len(intensity) == len(r):
        img_int[v, u] = intensity
    else:
        img_int[v, u] = r

    final = np.stack([img_range, img_int, img_range], axis=-1)

    return final


# -------- MAIN --------
def process_bag(bag_path, label, idx_start):

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    lidar_msg = None
    idx = idx_start

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == lidar_topic:
            msg_type = get_message(type_map[topic])
            lidar_msg = deserialize_message(data, msg_type)

        elif topic == cam_topic and lidar_msg is not None:
            msg_type = get_message(type_map[topic])
            cam_msg = deserialize_message(data, msg_type)

            # Decode
            xyz, intensity = decode_pc2(lidar_msg)
            cam_img = bridge.imgmsg_to_cv2(cam_msg, desired_encoding='bgr8')

            # Convert
            range_img = pointcloud_to_range_image(xyz, intensity)
            # Resize
            range_img = cv2.resize(range_img, (224,224))
            cam_img = cv2.resize(cam_img, (224,224))

            # Normalize
            range_img = cv2.normalize(range_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # FIX ORIENTATION (keep consistent always)
            range_img = cv2.flip(range_img, 0)

            # Save
            save_path = os.path.join(output_dir, label)
            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite(f"{save_path}/{idx}_range.png", range_img)
            cv2.imwrite(f"{save_path}/{idx}_cam.png", cam_img)

            idx += 1

    return idx


def main():
    idx = 0

    for folder in ["clean", "caution", "dirty"]:
        mapped_label = label_map[folder]
        folder_path = os.path.join(root_bag_dir, folder)

        if not os.path.exists(folder_path):
            print("Skipping missing:", folder_path)
            continue

        for bag_name in os.listdir(folder_path):
            bag_path = os.path.join(folder_path, bag_name)
            print("Processing:", bag_path)

            idx = process_bag(bag_path, mapped_label, idx)

    print("DONE. Total samples:", idx)


if __name__ == "__main__":
    main()