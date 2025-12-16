---
sidebar_label: 'Chapter 9: NVIDIA Isaac Sim Advanced Features'
sidebar_position: 10
---

# Chapter 9: NVIDIA Isaac Sim Advanced Features

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement complex perception systems in Isaac Sim
- Create realistic sensor models and calibration procedures
- Develop navigation and path planning systems
- Integrate AI models for robotic tasks
- Optimize simulation performance for large-scale environments

## Table of Contents
1. [Advanced Perception Systems](#advanced-perception-systems)
2. [Sensor Modeling and Calibration](#sensor-modeling-and-calibration)
3. [Navigation and Path Planning](#navigation-and-path-planning)
4. [AI Integration in Robotics](#ai-integration-in-robotics)
5. [Performance Optimization](#performance-optimization)
6. [Lab Exercise](#lab-exercise)
7. [Summary](#summary)
8. [Quiz](#quiz)

## Advanced Perception Systems

### Synthetic Data Generation

Isaac Sim excels at generating synthetic training data for AI models:

```python
import omni
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import cv2

class SyntheticDataGenerator:
    def __init__(self):
        self.sd_helper = SyntheticDataHelper()
        self.frame_count = 0

    def capture_training_data(self, robot_path, environment_path):
        # Set up different lighting conditions
        self.setup_lighting_variations()

        # Capture RGB, depth, and semantic segmentation
        rgb_data = self.sd_helper.get_rgb_data(robot_path)
        depth_data = self.sd_helper.get_depth_data(robot_path)
        seg_data = self.sd_helper.get_segmentation_data(robot_path)

        # Save data with annotations
        self.save_training_sample(rgb_data, depth_data, seg_data)

        self.frame_count += 1

    def setup_lighting_variations(self):
        # Cycle through different lighting conditions
        lights = self.get_all_lights()
        for i, light in enumerate(lights):
            # Vary intensity, color temperature, and position
            intensity = np.random.uniform(0.5, 2.0)
            color_temp = np.random.uniform(3000, 8000)  # Kelvin
            position = [
                np.random.uniform(-10, 10),
                np.random.uniform(5, 15),
                np.random.uniform(-10, 10)
            ]

            self.set_light_properties(light, intensity, color_temp, position)

    def save_training_sample(self, rgb, depth, segmentation):
        # Save RGB image
        cv2.imwrite(f"training_data/rgb/frame_{self.frame_count:06d}.png", rgb)

        # Save depth map
        np.save(f"training_data/depth/frame_{self.frame_count:06d}.npy", depth)

        # Save segmentation mask
        cv2.imwrite(f"training_data/seg/frame_{self.frame_count:06d}.png", segmentation)

        # Save annotations
        annotations = {
            "frame_id": self.frame_count,
            "timestamp": omni.usd.get_context().get_current_time()
        }
        np.save(f"training_data/annotations/frame_{self.frame_count:06d}_ann.npy", annotations)

    def add_random_objects(self):
        # Add random objects to the environment for data diversity
        objects = ["cube", "sphere", "cylinder", "cone"]
        for _ in range(np.random.randint(5, 15)):
            obj_type = np.random.choice(objects)
            position = [
                np.random.uniform(-5, 5),
                0.5,
                np.random.uniform(-5, 5)
            ]
            scale = np.random.uniform(0.1, 0.5)

            self.create_random_object(obj_type, position, scale)
```

### Multi-Sensor Fusion

Implementing sensor fusion for robust perception:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.range_sensor import _range_sensor

class MultiSensorFusion:
    def __init__(self):
        self.lidar_data = None
        self.camera_data = None
        self.imu_data = None
        self.odom_data = None

        # Calibration matrices
        self.lidar_to_camera = np.eye(4)  # Transformation matrix
        self.camera_to_imu = np.eye(4)

        # Data buffers
        self.data_buffer = {
            'lidar': [],
            'camera': [],
            'imu': [],
            'odom': []
        }

        # Fusion weights
        self.weights = {
            'lidar': 0.4,
            'camera': 0.4,
            'imu': 0.2
        }

    def process_lidar_data(self, lidar_msg):
        # Process LiDAR point cloud
        points = np.array(lidar_msg.ranges)

        # Filter and clean data
        valid_points = points[np.isfinite(points)]

        # Transform to global frame
        global_points = self.transform_points(valid_points, self.lidar_to_camera)

        self.data_buffer['lidar'].append(global_points)

        # Keep only recent data (sliding window)
        if len(self.data_buffer['lidar']) > 10:
            self.data_buffer['lidar'].pop(0)

    def process_camera_data(self, image_msg):
        # Process camera image for feature extraction
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # Extract features (example: ORB features)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(img, None)

        # Convert to world coordinates using depth
        world_features = self.project_features_to_world(keypoints, img)

        self.data_buffer['camera'].append(world_features)

        if len(self.data_buffer['camera']) > 10:
            self.data_buffer['camera'].pop(0)

    def process_imu_data(self, imu_msg):
        # Process IMU data for orientation and acceleration
        orientation = np.array([
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ])

        angular_velocity = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])

        linear_acceleration = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        self.data_buffer['imu'].append({
            'orientation': orientation,
            'angular_velocity': angular_velocity,
            'linear_acceleration': linear_acceleration
        })

        if len(self.data_buffer['imu']) > 50:  # Higher frequency for IMU
            self.data_buffer['imu'].pop(0)

    def fuse_sensor_data(self):
        # Weighted fusion of sensor data
        fused_position = np.zeros(3)

        # LiDAR-based position (for obstacle detection)
        if self.data_buffer['lidar']:
            lidar_pos = np.mean(self.data_buffer['lidar'][-1], axis=0)[:3]
            fused_position += self.weights['lidar'] * lidar_pos

        # Camera-based position (for landmark detection)
        if self.data_buffer['camera']:
            camera_pos = np.mean(self.data_buffer['camera'][-1], axis=0)
            fused_position += self.weights['camera'] * camera_pos

        # IMU-based position (for motion tracking)
        if self.data_buffer['imu']:
            imu_data = self.data_buffer['imu'][-1]
            # Integrate acceleration to get position (simplified)
            imu_pos = self.integrate_imu(imu_data)
            fused_position += self.weights['imu'] * imu_pos

        return fused_position

    def integrate_imu(self, imu_data):
        # Simplified integration of IMU data
        # In practice, use more sophisticated filtering (Kalman, particle filter)
        return np.zeros(3)  # Placeholder
```

### Semantic Segmentation and Object Detection

Advanced perception using Isaac Sim's rendering capabilities:

```python
from omni.isaac.synthetic_utils import plot
from pxr import UsdGeom
import torch
import torchvision.transforms as transforms

class PerceptionSystem:
    def __init__(self, robot_path):
        self.robot_path = robot_path
        self.semantic_labels = {}
        self.setup_semantic_segmentation()

        # Load pre-trained model for object detection
        self.detection_model = self.load_detection_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def setup_semantic_segmentation(self):
        # Assign semantic labels to objects in the scene
        stage = omni.usd.get_context().get_stage()

        # Find all prims in the scene
        for prim in stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                # Assign semantic labels based on object type
                object_type = self.get_object_type(prim)
                self.assign_semantic_label(prim, object_type)

    def get_object_type(self, prim):
        # Determine object type based on prim name or properties
        prim_name = prim.GetName().lower()

        if "wall" in prim_name or "floor" in prim_name:
            return "structure"
        elif "robot" in prim_name:
            return "robot"
        elif "person" in prim_name or "character" in prim_name:
            return "person"
        elif "furniture" in prim_name or "table" in prim_name or "chair" in prim_name:
            return "furniture"
        else:
            return "object"

    def assign_semantic_label(self, prim, label):
        # Assign semantic label to the prim
        prim_type = prim.GetPrimTypeInfo().GetTypeName()

        # Set the semantic label (this is conceptual - actual implementation varies)
        semantic_api = self.get_semantic_api(prim)
        semantic_api.set_semantic_label(label)

        # Store in our mapping
        self.semantic_labels[prim.GetPath()] = label

    def capture_semantic_data(self):
        # Capture semantic segmentation data from Isaac Sim
        viewport = omni.kit.viewport.get_viewport_interface()
        stage = omni.usd.get_context().get_stage()

        # Get semantic segmentation texture
        semantic_texture = self.get_semantic_texture()

        # Process and return segmentation mask
        segmentation_mask = self.process_semantic_texture(semantic_texture)

        return segmentation_mask

    def run_object_detection(self, rgb_image):
        # Run object detection on RGB image
        input_tensor = self.transform(rgb_image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.detection_model(input_tensor)

        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter predictions by confidence
        valid_detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                valid_detections.append({
                    'bbox': boxes[i],
                    'label': self.get_label_name(labels[i]),
                    'confidence': scores[i]
                })

        return valid_detections

    def get_label_name(self, label_id):
        # Map label ID to human-readable name
        label_map = {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            5: "airplane",
            6: "bus",
            7: "train",
            8: "truck",
            9: "boat",
            10: "traffic light"
            # Add more as needed
        }
        return label_map.get(label_id, f"object_{label_id}")

    def load_detection_model(self):
        # Load a pre-trained detection model
        # In practice, this would load from a checkpoint
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model
```

## Sensor Modeling and Calibration

### Camera Calibration in Isaac Sim

```python
import numpy as np
import cv2
from omni.isaac.sensor import Camera

class CameraCalibrator:
    def __init__(self, camera_path):
        self.camera = Camera(prim_path=camera_path)
        self.calibration_images = []
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image space

    def capture_calibration_data(self, num_images=20):
        """Capture calibration images with checkerboard pattern"""
        for i in range(num_images):
            # Move checkerboard to different positions/orientations
            self.move_checkerboard(i)

            # Capture image
            image = self.camera.get_current_frame()
            self.calibration_images.append(image)

            # Find checkerboard corners
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, (9, 6), None
            )

            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )

                # Store points
                objp = self.generate_object_points()  # 3D points
                self.object_points.append(objp)
                self.image_points.append(corners_refined)

    def generate_object_points(self, squares_x=9, squares_y=6, square_size=1.0):
        """Generate 3D points for checkerboard"""
        objp = np.zeros((squares_x * squares_y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2) * square_size
        return objp

    def move_checkerboard(self, step):
        """Move checkerboard to different position for calibration"""
        # In Isaac Sim, move the checkerboard object
        # This is conceptual - actual implementation would use USD transforms
        pass

    def calibrate_camera(self):
        """Perform camera calibration"""
        if len(self.object_points) < 10:
            raise ValueError("Need at least 10 calibration images")

        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.calibration_images[0].shape[:2][::-1],  # image size
            None, None
        )

        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(
                self.image_points[i],
                imgpoints2,
                cv2.NORM_L2
            ) / len(imgpoints2)
            total_error += error

        avg_error = total_error / len(self.object_points)

        print(f"Calibration completed with average error: {avg_error:.4f}")

        # Store calibration parameters
        self.camera_matrix = mtx
        self.dist_coeffs = dist

        return ret, mtx, dist, rvecs, tvecs

    def apply_calibration(self):
        """Apply calibration to Isaac Sim camera"""
        # Update Isaac Sim camera properties with calibration data
        # This would involve setting camera intrinsics in USD
        pass
```

### LiDAR Simulation and Calibration

```python
from omni.isaac.range_sensor import _range_sensor
import numpy as np

class LidarSimulator:
    def __init__(self, prim_path):
        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        self.prim_path = prim_path

        # LiDAR parameters
        self.params = {
            'rotation_count': 1,
            'rows': 16,
            'horizontal_pixels': 1024,
            'horizontal_fov': 360,
            'range': 100.0,
            'min_range': 0.1
        }

        # Add LiDAR to stage
        self.lidar_interface.add_lidar_to_stage(
            prim_path=prim_path,
            sensor_config=self.params
        )

    def get_point_cloud(self):
        """Get point cloud data from LiDAR"""
        return self.lidar_interface.get_linear_depth_data(self.prim_path)

    def calibrate_lidar(self, calibration_target):
        """Calibrate LiDAR using known target"""
        # Move calibration target to known positions
        positions = self.generate_calibration_positions()

        for pos in positions:
            self.move_calibration_target(pos)

            # Capture point cloud
            point_cloud = self.get_point_cloud()

            # Process and compare with expected values
            self.analyze_calibration_data(point_cloud, pos)

    def generate_calibration_positions(self):
        """Generate positions for calibration target"""
        positions = []
        for x in np.linspace(-2, 2, 5):
            for y in np.linspace(-2, 2, 5):
                positions.append([x, 0, y])  # [x, y, z]
        return positions

    def analyze_calibration_data(self, point_cloud, expected_pos):
        """Analyze calibration data and adjust parameters"""
        # Calculate actual position from point cloud
        actual_pos = self.estimate_position_from_pointcloud(point_cloud)

        # Calculate error
        error = np.linalg.norm(np.array(actual_pos) - np.array(expected_pos))

        # Adjust calibration parameters if error is too large
        if error > 0.05:  # 5cm threshold
            self.adjust_calibration(expected_pos, actual_pos)

    def estimate_position_from_pointcloud(self, point_cloud):
        """Estimate target position from point cloud"""
        # Simple approach: use centroid of points
        if len(point_cloud) > 0:
            return np.mean(point_cloud, axis=0)
        else:
            return [0, 0, 0]
```

## Navigation and Path Planning

### Path Planning in Isaac Sim

```python
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import cv2

class NavigationSystem:
    def __init__(self, map_resolution=0.1, map_size=(20, 20)):
        self.map_resolution = map_resolution
        self.map_size = map_size
        self.occupancy_grid = np.zeros(
            (int(map_size[0]/map_resolution), int(map_size[1]/map_resolution))
        )

        # Robot parameters
        self.robot_radius = 0.3  # meters

        # Path planning
        self.path = []
        self.current_goal = None

    def update_occupancy_grid(self, lidar_data, robot_pose):
        """Update occupancy grid from LiDAR data"""
        # Convert LiDAR readings to grid coordinates
        angles = np.linspace(0, 2*np.pi, len(lidar_data))

        for i, distance in enumerate(lidar_data):
            if np.isfinite(distance):
                # Calculate position in world coordinates
                angle = angles[i]
                world_x = robot_pose[0] + distance * np.cos(angle)
                world_y = robot_pose[1] + distance * np.sin(angle)

                # Convert to grid coordinates
                grid_x = int((world_x + self.map_size[0]/2) / self.map_resolution)
                grid_y = int((world_y + self.map_size[1]/2) / self.map_resolution)

                # Check bounds
                if 0 <= grid_x < self.occupancy_grid.shape[1] and \
                   0 <= grid_y < self.occupancy_grid.shape[0]:
                    # Mark as occupied
                    self.occupancy_grid[grid_y, grid_x] = 1

    def plan_path(self, start, goal):
        """Plan path using A* algorithm"""
        self.current_goal = goal

        # Create graph from occupancy grid
        graph = self.create_graph_from_grid()

        # Convert start and goal to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Find path using A*
        try:
            path = nx.astar_path(graph, start_grid, goal_grid,
                                heuristic=self.manhattan_distance)

            # Convert back to world coordinates
            self.path = [self.grid_to_world(p) for p in path]
            return self.path
        except nx.NetworkXNoPath:
            print("No path found to goal")
            return None

    def create_graph_from_grid(self):
        """Create navigation graph from occupancy grid"""
        graph = nx.Graph()

        height, width = self.occupancy_grid.shape

        # Add nodes for free space
        for y in range(height):
            for x in range(width):
                if self.occupancy_grid[y, x] == 0:  # Free space
                    # Check if robot can fit here
                    if self.is_free_space(x, y):
                        graph.add_node((x, y))

        # Add edges between adjacent free nodes
        for y in range(height):
            for x in range(width):
                if graph.has_node((x, y)):
                    # Check 8-connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue

                            nx_neighbor = x + dx
                            ny_neighbor = y + dy

                            if (0 <= nx_neighbor < width and
                                0 <= ny_neighbor < height and
                                graph.has_node((nx_neighbor, ny_neighbor))):

                                # Calculate edge weight (distance)
                                weight = np.sqrt(dx*dx + dy*dy)
                                graph.add_edge(
                                    (x, y),
                                    (nx_neighbor, ny_neighbor),
                                    weight=weight
                                )

        return graph

    def is_free_space(self, x, y):
        """Check if space is free considering robot radius"""
        # Check a small area around the point
        radius_in_grid = int(self.robot_radius / self.map_resolution)

        for dy in range(-radius_in_grid, radius_in_grid + 1):
            for dx in range(-radius_in_grid, radius_in_grid + 1):
                check_x = x + dx
                check_y = y + dy

                if (0 <= check_x < self.occupancy_grid.shape[1] and
                    0 <= check_y < self.occupancy_grid.shape[0]):

                    if self.occupancy_grid[check_y, check_x] == 1:
                        return False  # Obstacle found

        return True

    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_pos[0] + self.map_size[0]/2) / self.map_resolution)
        grid_y = int((world_pos[1] + self.map_size[1]/2) / self.map_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_pos[0] * self.map_resolution - self.map_size[0]/2
        world_y = grid_pos[1] * self.map_resolution - self.map_size[1]/2
        return (world_x, world_y)

    def manhattan_distance(self, node1, node2):
        """Manhattan distance heuristic for A*"""
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    def follow_path(self, robot_pose, look_ahead_distance=1.0):
        """Follow planned path with local obstacle avoidance"""
        if not self.path:
            return None

        # Find closest point on path
        closest_idx = self.find_closest_point(robot_pose)

        # Look ahead on path
        target_idx = self.find_look_ahead_point(closest_idx, look_ahead_distance)

        if target_idx is not None:
            target = self.path[target_idx]

            # Calculate direction to target
            direction = np.array(target) - np.array(robot_pose[:2])
            distance = np.linalg.norm(direction)

            # Normalize direction
            if distance > 0:
                direction = direction / distance

            return {
                'target': target,
                'direction': direction,
                'distance': distance
            }

        return None

    def find_closest_point(self, robot_pose):
        """Find closest point on path to robot"""
        min_dist = float('inf')
        closest_idx = 0

        for i, point in enumerate(self.path):
            dist = np.linalg.norm(np.array(point) - np.array(robot_pose[:2]))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def find_look_ahead_point(self, start_idx, look_ahead_distance):
        """Find point on path at look-ahead distance"""
        current_dist = 0

        for i in range(start_idx, len(self.path)):
            if i == len(self.path) - 1:
                return i

            # Calculate distance to next point
            dist_to_next = np.linalg.norm(
                np.array(self.path[i+1]) - np.array(self.path[i])
            )

            if current_dist + dist_to_next >= look_ahead_distance:
                # Interpolate to exact look-ahead distance
                remaining_dist = look_ahead_distance - current_dist
                direction = np.array(self.path[i+1]) - np.array(self.path[i])
                direction = direction / np.linalg.norm(direction)
                interpolated_point = np.array(self.path[i]) + direction * remaining_dist
                return i + 1  # Return next index as approximation

            current_dist += dist_to_next

        return len(self.path) - 1
```

## AI Integration in Robotics

### Reinforcement Learning in Isaac Sim

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RobotDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.gamma = 0.95  # discount rate

        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

        # Keep memory size manageable
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = torch.LongTensor([self.memory[i][1] for i in batch])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.BoolTensor([self.memory[i][4] for i in batch])

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class IsaacSimRLAgent:
    def __init__(self, world):
        self.world = world
        self.agent = None
        self.setup_rl_agent()

    def setup_rl_agent(self):
        """Setup RL agent with appropriate state and action space"""
        # Define state space (e.g., LiDAR readings, robot pose, goal direction)
        state_size = 360 + 6  # 360 LiDAR readings + 6 pose values
        action_size = 5  # 5 discrete actions: forward, left, right, slight left, slight right

        self.agent = RobotDQNAgent(state_size, action_size)

    def get_state(self):
        """Get current state from Isaac Sim sensors"""
        # Get LiDAR data
        lidar_data = self.get_lidar_readings()

        # Get robot pose
        robot_pose = self.get_robot_pose()

        # Get goal direction (relative to robot)
        goal_direction = self.get_goal_direction()

        # Combine into state vector
        state = np.concatenate([lidar_data, robot_pose, goal_direction])
        return state

    def get_lidar_readings(self):
        """Get LiDAR readings from Isaac Sim"""
        # This would interface with Isaac Sim's LiDAR sensor
        # Return 360 distance readings
        pass

    def get_robot_pose(self):
        """Get robot pose (position and orientation)"""
        # Return [x, y, z, roll, pitch, yaw]
        pass

    def get_goal_direction(self):
        """Get direction to goal relative to robot"""
        # Return [dx, dy] normalized
        pass

    def execute_action(self, action):
        """Execute action in Isaac Sim environment"""
        # Convert discrete action to continuous velocity commands
        vel_cmd = self.discrete_to_continuous_action(action)

        # Send command to robot in Isaac Sim
        self.send_velocity_command(vel_cmd)

    def discrete_to_continuous_action(self, action):
        """Convert discrete action to continuous velocity"""
        action_map = {
            0: [0.5, 0.0],    # Forward
            1: [0.3, 0.3],    # Left
            2: [0.3, -0.3],   # Right
            3: [0.4, 0.1],    # Slight left
            4: [0.4, -0.1]    # Slight right
        }
        return action_map.get(action, [0.0, 0.0])

    def calculate_reward(self, state, action, next_state, done):
        """Calculate reward based on state transition"""
        # Positive reward for moving toward goal
        goal_reward = self.distance_to_goal_reward(next_state)

        # Negative reward for collisions
        collision_penalty = self.collision_penalty(state)

        # Small time penalty to encourage efficiency
        time_penalty = -0.01

        # Bonus for reaching goal
        goal_bonus = 100 if done else 0

        total_reward = goal_reward + collision_penalty + time_penalty + goal_bonus
        return total_reward

    def train_episode(self, max_steps=1000):
        """Train agent for one episode"""
        state = self.get_state()
        total_reward = 0

        for step in range(max_steps):
            # Choose action
            action = self.agent.act(state)

            # Execute action in environment
            self.execute_action(action)

            # Step the simulation
            self.world.step(render=True)

            # Get next state
            next_state = self.get_state()

            # Calculate reward
            done = self.check_episode_done(next_state)
            reward = self.calculate_reward(state, action, next_state, done)

            # Store experience
            self.agent.remember(state, action, reward, next_state, done)

            # Train on batch
            self.agent.replay()

            # Update state
            state = next_state
            total_reward += reward

            if done:
                break

        return total_reward
```

## Performance Optimization

### Simulation Optimization Techniques

```python
import omni
from omni.isaac.core import World
import carb

class PerformanceOptimizer:
    def __init__(self, world):
        self.world = world
        self.settings = carb.settings.get_settings()

    def optimize_physics(self):
        """Optimize physics parameters for performance"""
        # Set physics substeps
        self.settings.set("/physics/solverPositionIterations", 4)
        self.settings.set("/physics/solverVelocityIterations", 1)

        # Adjust solver parameters
        self.settings.set("/physics/solverType", 0)  # 0=PBD, 1=TGS
        self.settings.set("/physics/frictionModel", 1)  # 1=patch, 2=cone

        # Set fixed time step
        self.settings.set("/physics/timeStepsPerSecond", 60)  # Lower for performance

    def optimize_rendering(self):
        """Optimize rendering settings"""
        # Reduce rendering quality in simulation mode
        self.settings.set("/app/renderer/resolution/width", 640)
        self.settings.set("/app/renderer/resolution/height", 480)

        # Disable expensive rendering features
        self.settings.set("/rtx/ambientOcclusion/enabled", False)
        self.settings.set("/rtx/directLighting/enable", False)
        self.settings.set("/rtx/pathTracing/enable", False)

    def optimize_collisions(self):
        """Optimize collision detection"""
        # Use simpler collision shapes where possible
        # Reduce collision margin
        self.settings.set("/physics/collisionMargin", 0.001)

        # Enable broadphase culling
        self.settings.set("/physics/broadphaseType", 1)  # 1=SAP, 2=MBP

    def optimize_memory_usage(self):
        """Optimize memory usage"""
        # Reduce texture streaming budget
        self.settings.set("/renderer/texturePoolSize", 512)  # MB

        # Enable texture streaming
        self.settings.set("/renderer/enableTextureStreaming", True)

    def set_simulation_mode(self, mode="balanced"):
        """Set overall simulation optimization mode"""
        if mode == "performance":
            self.optimize_physics()
            self.optimize_rendering()
            self.optimize_collisions()
        elif mode == "quality":
            # Higher quality settings
            self.settings.set("/physics/timeStepsPerSecond", 240)
            self.settings.set("/app/renderer/resolution/width", 1920)
            self.settings.set("/app/renderer/resolution/height", 1080)
        elif mode == "balanced":
            # Default balanced settings
            self.settings.set("/physics/timeStepsPerSecond", 120)
            self.settings.set("/app/renderer/resolution/width", 1280)
            self.settings.set("/app/renderer/resolution/height", 720)

    def get_performance_metrics(self):
        """Get current performance metrics"""
        metrics = {}

        # Get physics stats
        physics_stats = self.world.get_physics_stats()
        metrics['physics_time'] = physics_stats.get("solverTime", 0)

        # Get rendering stats
        render_stats = carb.app.get_render_stats()
        metrics['render_time'] = render_stats.get("renderTime", 0)
        metrics['frame_time'] = render_stats.get("frameTime", 0)

        # Get memory usage
        metrics['memory_usage'] = carb.app.get_memory_usage()

        return metrics

    def adaptive_optimization(self):
        """Adaptively optimize based on performance metrics"""
        metrics = self.get_performance_metrics()

        # If physics is taking too long, reduce complexity
        if metrics.get('physics_time', 0) > 0.016:  # More than 16ms
            self.settings.set("/physics/timeStepsPerSecond", 60)

        # If rendering is slow, reduce quality
        if metrics.get('frame_time', 0) > 0.033:  # More than 33ms (30 FPS)
            self.settings.set("/app/renderer/resolution/width", 640)
            self.settings.set("/app/renderer/resolution/height", 480)
```

## Lab Exercise

### Objective
Implement an advanced perception and navigation system in Isaac Sim with AI integration.

### Instructions
1. Set up a complex environment with multiple rooms and obstacles
2. Implement a multi-sensor fusion system (LiDAR + camera)
3. Create a navigation system with path planning and obstacle avoidance
4. Integrate a reinforcement learning agent for navigation
5. Optimize the simulation for real-time performance

### Expected Outcome
You should have a complete perception and navigation system running in Isaac Sim with AI capabilities and optimized performance.

## Summary

In this chapter, we explored advanced features of NVIDIA Isaac Sim including synthetic data generation, multi-sensor fusion, perception systems, navigation algorithms, and AI integration. We also covered performance optimization techniques to ensure efficient simulation. These advanced capabilities make Isaac Sim a powerful platform for developing sophisticated robotic systems with realistic sensor simulation and AI training capabilities.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.