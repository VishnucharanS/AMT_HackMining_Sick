from mcap.reader import make_reader
import numpy as np

def read_mcap(file_path):
    frames = []

    with open(file_path, "rb") as f:
        reader = make_reader(f)

        for schema, channel, message in reader.iter_messages():
            
            if "/lidar/cloud/device_id47" in channel.topic.lower():
                
                data = message.data  # raw bytes
                
                # ⚠️ You must decode based on format
                # For now, we store raw — we’ll parse next
                frames.append(data)

    print(f"Total frames: {len(frames)}")
    return frames

import numpy as np

def extract_features(points):
    # points: Nx3 or Nx4 array

    if len(points) == 0:
        return None

    num_points = len(points)

    distances = np.linalg.norm(points[:, :3], axis=1)

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # if intensity exists
    if points.shape[1] > 3:
        intensity = points[:, 3]
        mean_intensity = np.mean(intensity)
    else:
        mean_intensity = 0

    return {
        "num_points": num_points,
        "mean_dist": mean_dist,
        "std_dist": std_dist,
        "mean_intensity": mean_intensity
    }

def compute_baseline(feature_list):
    arr = np.array(feature_list)

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    return mean, std

def compute_score(feature, mean, std):
    feature_arr = np.array(feature)

    z = np.abs((feature_arr - mean) / (std + 1e-6))

    anomaly_score = np.mean(z)

    # convert to health score
    health = max(0, 100 - anomaly_score * 20)

    return health

import numpy as np

all_features = []

# 👉 Replace this with actual parsed frames
frames = [...]  

for frame in frames:
    # frame must be Nx3 or Nx4 numpy array
    features = extract_features(frame)
    
    if features is not None:
        all_features.append([
            features["num_points"],
            features["mean_dist"],
            features["std_dist"],
            features["mean_intensity"]
        ])

# --- baseline ---
mean, std = compute_baseline(all_features)

print("Baseline mean:", mean)
print("Baseline std:", std)

# --- scoring ---
for i, f in enumerate(all_features):
    score = compute_score(f, mean, std)
    print(f"Frame {i}: Health Score = {score:.2f}")