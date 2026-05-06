# AMT_HackMining_Sick
ROS2 Multimodal Contamination Detection System

--------------------------------------------------

Overview

This project implements a real-time multimodal contamination detection pipeline using LiDAR, Camera, and optional ML-based inputs in ROS2. It combines sensor data and rule-based logic to estimate environmental contamination levels and outputs a unified system state.

--------------------------------------------------

System Description

The system integrates:

- LiDAR data → intensity-based contamination detection
- Camera data → blur and brightness-based contamination cues
- ML predictions → optional external model inference
- Fusion logic → combines all inputs into a final state

Final output is a traffic-light style status:
- NORMAL
- REDUCED
- CRITICAL

--------------------------------------------------

Architecture

LiDAR Node ─┐
            ├──► Contamination Monitor ───► Fusion Node ───► Final Output
Camera Node ─┘

(Optional)
ML Node ────────────────────────────────┘

--------------------------------------------------

Project Structure

.
├── contamination_demo/          (ROS2 package)
├── fusion_node.py              (Fusion logic node)
├── ros_multi_modal_detector.py (ML-based detector)
├── Dockerfile                  (if used)
├── README.txt

--------------------------------------------------

Features

- Real-time ROS2 node-based architecture
- Sensor fusion with weighted scoring
- Temporal smoothing and consensus logic
- State machine with cooldown handling
- Robust fallback when one sensor is unavailable
- Traffic light output for downstream systems

--------------------------------------------------

How It Works

1. LiDAR Processing
   - Extracts intensity values from PointCloud2
   - Computes contamination score using thresholds

2. Camera Processing
   - Converts image to grayscale
   - Uses Laplacian variance (blur detection)
   - Combines with brightness

3. Fusion Logic
   - Weighted combination:
     fused = 0.65 * lidar + 0.35 * camera
   - Applies rules such as:
     - Sensor dominance
     - Dust escalation
     - Confidence smoothing

4. State Machine
   - Handles transitions between NORMAL, REDUCED, and CRITICAL
   - Uses time-window consensus and cooldown to prevent flickering

--------------------------------------------------

ROS Topics

Subscribed:
- /lidar/cloud/device_id47
- /visionary2/bgr/device_id4

Published:
- /contamination_status (String)
  Values: NORMAL / REDUCED / CRITICAL

- /trafic_light_color_raw (Int32)
  2 = NORMAL
  3 = REDUCED
  1 = CRITICAL

--------------------------------------------------

How to Run

1. Build workspace
   colcon build
   source install/setup.bash

2. Run contamination node
   ros2 run contamination_demo contamination_monitor_node

3. (Optional) Run ML node
   python ros_multi_modal_detector.py

4. Play data (if using rosbag)
   ros2 bag play <your_bag_file>

--------------------------------------------------

Requirements

- ROS2 (Humble or newer recommended)
- Python 3
- NumPy
- OpenCV
- PyTorch (for ML node)

Install dependencies:
pip install numpy opencv-python torch torchvision

--------------------------------------------------

Notes

- Do not upload the following folders:
  build/
  install/
  log/
  venv/

- Ensure ROS topics are actively publishing (e.g., using rosbag)
- ROS logger may suppress repeated messages; use print() for debugging if needed

--------------------------------------------------

Future Improvements

- Dynamic threshold tuning
- Improved ML integration
- Visualization dashboard (RViz or web UI)
- ROS2 launch file for full pipeline

--------------------------------------------------

Author

Vishnucharan
Utkarsh Anand
Saimothish
Kishore 

--------------------------------------------------