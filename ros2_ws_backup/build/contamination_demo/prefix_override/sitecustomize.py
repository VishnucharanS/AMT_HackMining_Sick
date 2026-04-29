import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/vishnucharan/ROS/amthack/ros2_node_hack/ros2_ws_backup/install/contamination_demo'
