---
sidebar_label: 'Chapter 13: NVIDIA Isaac Sim Integration'
sidebar_position: 14
---

# Chapter 13: NVIDIA Isaac Sim Integration

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up and configure NVIDIA Isaac Sim for robotics simulation
- Create and import robot models into Isaac Sim
- Implement sensor systems and perception pipelines in Isaac Sim
- Integrate Isaac Sim with ROS2 using Isaac ROS
- Develop realistic physics and material properties for simulation
- Optimize simulation performance for complex humanoid robotics scenarios

## Table of Contents
1. [Introduction to Isaac Sim](#introduction-to-isaac-sim)
2. [Installation and Setup](#installation-and-setup)
3. [Isaac Sim Interface and Navigation](#isaac-sim-interface-and-navigation)
4. [Robot Modeling in Isaac Sim](#robot-modeling-in-isaac-sim)
5. [Sensor Integration](#sensor-integration)
6. [Physics and Materials](#physics-and-materials)
7. [ROS2 Integration](#ros2-integration)
8. [Simulation Optimization](#simulation-optimization)
9. [Lab Exercise](#lab-exercise)
10. [Summary](#summary)
11. [Quiz](#quiz)

## Introduction to Isaac Sim

NVIDIA Isaac Sim is a comprehensive robotics simulation application built on NVIDIA Omniverse. It provides a photorealistic, physically-accurate simulation environment for developing, testing, and validating robotics applications. Isaac Sim is particularly well-suited for complex humanoid robotics and Physical AI applications due to its high-fidelity physics simulation, realistic rendering, and seamless integration with the NVIDIA ecosystem.

### Key Features of Isaac Sim

1. **Photorealistic Rendering**: RTX-accelerated rendering for realistic sensor simulation
2. **Accurate Physics**: PhysX 4.0 integration for precise physics simulation
3. **Large-Scale Environments**: Support for complex, large-scale simulation environments
4. **Multi-Robot Simulation**: Efficient simulation of multiple robots simultaneously
5. **ROS/ROS2 Integration**: Native support for ROS and ROS2 communication
6. **AI Training Environment**: Built-in tools for training AI models with synthetic data
7. **Cloud + Local Deployment**: Runs on both local RTX workstations and cloud instances

### Why Isaac Sim for Physical AI?

Isaac Sim is particularly valuable for Physical AI applications because:

- **High-Fidelity Physics**: Accurate simulation of contact forces, friction, and dynamics essential for humanoid locomotion
- **Realistic Sensor Simulation**: High-quality camera, LiDAR, and IMU simulation with realistic noise models
- **NVIDIA Hardware Integration**: Optimized for Jetson and RTX hardware used in real Physical AI systems
- **AI-Native**: Built for training and testing AI models with synthetic data
- **Embodied AI Focus**: Designed specifically for embodied AI applications

### Isaac Sim vs Other Simulation Platforms

| Feature | Isaac Sim | Gazebo | Unity | Custom Engines |
|---------|-----------|--------|-------|----------------|
| Physics Accuracy | Excellent | Excellent | Good | Variable |
| Rendering Quality | Excellent | Good | Excellent | Variable |
| NVIDIA Hardware Optimization | Excellent | Good | Good | Poor |
| ROS Integration | Good (Isaac ROS) | Excellent | Good (Unity Robotics) | Variable |
| AI Training Support | Excellent | Good | Good | Variable |
| Large-Scale Environments | Excellent | Good | Excellent | Variable |

## Installation and Setup

### System Requirements

Isaac Sim has demanding system requirements due to its high-fidelity rendering and physics simulation:

**Minimum Requirements:**
- GPU: NVIDIA RTX 3070 or equivalent with 8GB+ VRAM
- CPU: Multi-core processor (Intel i7 or AMD Ryzen 7)
- RAM: 32GB or more
- OS: Ubuntu 20.04 LTS or Windows 10/11
- CUDA: CUDA 11.8 or later
- Storage: 50GB+ free space

**Recommended Requirements:**
- GPU: NVIDIA RTX 4080/4090 or RTX A5000/A6000 with 16GB+ VRAM
- CPU: Intel i9 or AMD Threadripper
- RAM: 64GB or more
- Network: Gigabit Ethernet for multi-machine setups

### Installation Process

#### 1. Install NVIDIA Omniverse Launcher

First, download and install the NVIDIA Omniverse Launcher:

```bash
# Visit https://www.nvidia.com/en-us/omniverse/download/
# Download the Omniverse Launcher for your platform
# Run the installer and create an NVIDIA developer account if needed
```

#### 2. Install Isaac Sim

Through the Omniverse Launcher:
1. Open Omniverse Launcher
2. Go to "Apps" tab
3. Find "Isaac Sim" in the applications list
4. Click "Install" and select the latest stable version
5. Wait for the installation to complete

#### 3. Install Isaac ROS Bridge

For ROS2 integration:

```bash
# Add NVIDIA package repository
wget https://repo.download.nvidia.com/nvidia-isaac-ros.gpg
sudo apt-key add nvidia-isaac-ros.gpg
echo "deb https://repo.download.nvidia.com/isaac_ros/main $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/nvidia-isaac-ros.list

sudo apt update
sudo apt install nvidia-isaac-ros-foxy-packages
```

#### 4. Verify Installation

Launch Isaac Sim from the Omniverse Launcher and verify that:

1. The application starts without errors
2. GPU acceleration is working (check the console for CUDA initialization messages)
3. Basic physics simulation runs smoothly
4. ROS2 bridge components are accessible

### Initial Configuration

After installation, configure Isaac Sim for your development environment:

1. **Set up workspace directory**:
```bash
mkdir -p ~/isaac_sim_workspace/{scenes,robots,sensors,materials}
```

2. **Configure environment variables**:
```bash
# Add to ~/.bashrc
export ISAAC_SIM_PATH=$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1  # Adjust version as needed
export NVIDIA_OMNIVERSE_ADDONS_PATHS=$ISAAC_SIM_PATH/exts
export PATH=$PATH:$ISAAC_SIM_PATH/python.sh
```

3. **Install Python dependencies**:
```bash
pip3 install --upgrade pip
pip3 install nvidia-isaac
pip3 install open3d  # For 3D point cloud processing
pip3 install opencv-contrib-python  # For computer vision
```

## Isaac Sim Interface and Navigation

### Main Interface Components

When you first launch Isaac Sim, you'll encounter several key interface components:

1. **Viewport**: The main 3D scene view where you see your simulation
2. **Stage Panel**: Hierarchical view of all objects in the scene
3. **Property Panel**: Properties and settings for selected objects
4. **Timeline**: Animation and simulation controls
5. **Content Browser**: Asset library and file management
6. **Console**: Output and scripting console

### Navigation Controls

Mastering Isaac Sim navigation is crucial for efficient workflow:

**Viewport Navigation:**
- **Orbit**: Alt + Left Mouse Button (Alt + LMB)
- **Pan**: Alt + Middle Mouse Button (Alt + MMB) or Shift + LMB
- **Zoom**: Alt + Right Mouse Button (Alt + RMB) or Mouse Wheel
- **Focus**: F key on selected object
- **Frame All**: A key to frame all objects in view

**Scene Navigation:**
- **Walk Mode**: Toggle with Ctrl + F for first-person navigation
- **Fly Mode**: Toggle with Ctrl + G for free movement
- **Grid Snap**: Hold X while moving to snap to grid
- **Rotation Snap**: Hold C while rotating to snap to angles

### Basic Operations

#### Creating Objects

1. **Through Menu**: Create → Primitive → [Shape] (Cube, Sphere, Capsule, etc.)
2. **Through Stage Panel**: Right-click → Create → [Primitive Type]
3. **Through Content Browser**: Drag assets from the library to the stage

#### Transform Operations

Each object in Isaac Sim has three transform handles:
- **Move Tool**: Arrow handles for translation (W key)
- **Rotate Tool**: Circular handles for rotation (E key)
- **Scale Tool**: Box handles for scaling (R key)

### Scene Management

#### Scene Structure

Isaac Sim uses the Universal Scene Description (USD) format, which organizes scenes hierarchically:

```
World
├── Environment
│   ├── GroundPlane
│   ├── Walls
│   └── Furniture
├── Robots
│   ├── Robot1
│   └── Robot2
├── Sensors
├── Lights
└── Cameras
```

#### Managing Complexity

For complex humanoid robotics scenarios:
- Use groups to organize related objects
- Implement instancing for repeated elements
- Use variants for different configurations of the same base object
- Apply level-of-detail (LOD) systems for performance

## Robot Modeling in Isaac Sim

### Supported Robot Formats

Isaac Sim supports multiple robot description formats:

1. **URDF (Unified Robot Description Format)**: Most common in ROS ecosystem
2. **MJCF (MuJoCo XML)**: Used in DeepMind and other research environments
3. **USD (Universal Scene Description)**: Native Omniverse format
4. **FBX/OBJ**: Standard 3D formats with custom configurations

### Importing URDF Robots

The most common approach is to import URDF robots:

```python
# Python script to import URDF in Isaac Sim
import omni
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from pxr import UsdGeom

# Get the current stage
stage = omni.usd.get_context().get_stage()

# Import URDF robot
# This would typically be done through the Isaac Sim UI or Python API
# Example: Import a simple wheeled robot
robot_path = "/Isaac/Robots/Franka/franka_alt_fingers.usd"  # Example path
add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot1")

# Set initial pose
robot_prim = stage.GetPrimAtPath("/World/Robot1")
if robot_prim.IsValid():
    UsdGeom.XformCommonAPI(robot_prim).SetTranslate((0, 0, 0.5))  # Position robot
    UsdGeom.XformCommonAPI(robot_prim).SetRotate((0, 0, 0, 1))   # Set orientation
```

### Creating Custom Robot Models

For humanoid robots, you'll often need to create custom models:

```python
# Creating a simple humanoid robot in USD
import omni
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.stage import get_current_stage

def create_simple_humanoid_robot(robot_name="/World/HumanoidRobot"):
    """Create a simple humanoid robot model"""
    stage = get_current_stage()

    # Create root prim for the robot
    robot_prim = UsdGeom.Xform.Define(stage, robot_name)

    # Create body (torso)
    body_path = f"{robot_name}/Body"
    body_geom = UsdGeom.Capsule.Define(stage, body_path)
    body_geom.CreateRadiusAttr(0.15)
    body_geom.CreateHeightAttr(0.8)

    # Set body properties
    body_geom.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0, 0, 0.9))

    # Create head
    head_path = f"{robot_name}/Head"
    head_geom = UsdGeom.Sphere.Define(stage, head_path)
    head_geom.CreateRadiusAttr(0.12)
    head_geom.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0, 0, 1.4))

    # Create limbs
    # Left arm
    left_upper_arm_path = f"{robot_name}/LeftUpperArm"
    left_upper_arm = UsdGeom.Capsule.Define(stage, left_upper_arm_path)
    left_upper_arm.CreateRadiusAttr(0.05)
    left_upper_arm.CreateHeightAttr(0.4)
    left_upper_arm.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(-0.25, 0, 1.2))

    left_lower_arm_path = f"{robot_name}/LeftLowerArm"
    left_lower_arm = UsdGeom.Capsule.Define(stage, left_lower_arm_path)
    left_lower_arm.CreateRadiusAttr(0.04)
    left_lower_arm.CreateHeightAttr(0.35)
    left_lower_arm.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(-0.45, 0, 1.2))

    # Right arm
    right_upper_arm_path = f"{robot_name}/RightUpperArm"
    right_upper_arm = UsdGeom.Capsule.Define(stage, right_upper_arm_path)
    right_upper_arm.CreateRadiusAttr(0.05)
    right_upper_arm.CreateHeightAttr(0.4)
    right_upper_arm.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.25, 0, 1.2))

    right_lower_arm_path = f"{robot_name}/RightLowerArm"
    right_lower_arm = UsdGeom.Capsule.Define(stage, right_lower_arm_path)
    right_lower_arm.CreateRadiusAttr(0.04)
    right_lower_arm.CreateHeightAttr(0.35)
    right_lower_arm.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.45, 0, 1.2))

    # Legs
    left_thigh_path = f"{robot_name}/LeftThigh"
    left_thigh = UsdGeom.Capsule.Define(stage, left_thigh_path)
    left_thigh.CreateRadiusAttr(0.06)
    left_thigh.CreateHeightAttr(0.5)
    left_thigh.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(-0.1, 0, 0.5))

    left_calf_path = f"{robot_name}/LeftCalf"
    left_calf = UsdGeom.Capsule.Define(stage, left_calf_path)
    left_calf.CreateRadiusAttr(0.05)
    left_calf.CreateHeightAttr(0.45)
    left_calf.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(-0.1, 0, 0.1))

    right_thigh_path = f"{robot_name}/RightThigh"
    right_thigh = UsdGeom.Capsule.Define(stage, right_thigh_path)
    right_thigh.CreateRadiusAttr(0.06)
    right_thigh.CreateHeightAttr(0.5)
    right_thigh.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.1, 0, 0.5))

    right_calf_path = f"{robot_name}/RightCalf"
    right_calf = UsdGeom.Capsule.Define(stage, right_calf_path)
    right_calf.CreateRadiusAttr(0.05)
    right_calf.CreateHeightAttr(0.45)
    right_calf.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.1, 0, 0.1))

    # Add physics properties to each part
    add_physics_properties_to_robot(stage, robot_name)

def add_physics_properties_to_robot(stage, robot_path):
    """Add physics properties to robot parts"""
    # Add rigid body properties to each part
    parts = [
        f"{robot_path}/Body",
        f"{robot_path}/Head",
        f"{robot_path}/LeftUpperArm", f"{robot_path}/LeftLowerArm",
        f"{robot_path}/RightUpperArm", f"{robot_path}/RightLowerArm",
        f"{robot_path}/LeftThigh", f"{robot_path}/LeftCalf",
        f"{robot_path}/RightThigh", f"{robot_path}/RightCalf"
    ]

    for part_path in parts:
        part_prim = stage.GetPrimAtPath(part_path)

        # Add rigid body API
        UsdPhysics.RigidBodyAPI.Apply(part_prim)

        # Add collision API
        UsdPhysics.CollisionAPI.Apply(part_prim)

        # Add mass properties
        mass_api = UsdPhysics.MassAPI.Apply(part_prim)
        mass_api.CreateMassAttr().Set(1.0)  # 1 kg for each part as example

        # Add joint constraints between parts (simplified)
        # In a real implementation, you'd add proper joints with limits and drives
```

### Joint Configuration

For articulated robots, proper joint configuration is crucial:

```python
def add_joints_to_humanoid(stage, robot_path):
    """Add joints to connect robot parts"""

    # Neck joint (head to body)
    neck_joint_path = f"{robot_path}/NeckJoint"
    neck_joint = PhysxSchema.PhysxJoint.Create(stage, neck_joint_path)
    neck_joint.GetActor0Rel().SetTargets([f"{robot_path}/Body"])
    neck_joint.GetActor1Rel().SetTargets([f"{robot_path}/Head"])

    # Configure joint properties
    neck_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0.8))  # Connection point on body
    neck_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -0.2))  # Connection point on head
    neck_joint.CreateBreakForceAttr().Set(1000000.0)  # Very strong joint

    # Shoulder joints
    left_shoulder_path = f"{robot_path}/LeftShoulderJoint"
    left_shoulder = PhysxSchema.PhysxJoint.Create(stage, left_shoulder_path)
    left_shoulder.GetActor0Rel().SetTargets([f"{robot_path}/Body"])
    left_shoulder.GetActor1Rel().SetTargets([f"{robot_path}/LeftUpperArm"])
    left_shoulder.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.15, 0, 0.7))
    left_shoulder.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -0.2))

    # Elbow joints
    left_elbow_path = f"{robot_path}/LeftElbowJoint"
    left_elbow = PhysxSchema.PhysxJoint.Create(stage, left_elbow_path)
    left_elbow.GetActor0Rel().SetTargets([f"{robot_path}/LeftUpperArm"])
    left_elbow.GetActor1Rel().SetTargets([f"{robot_path}/LeftLowerArm"])
    left_elbow.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, -0.2))
    left_elbow.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -0.175))

    # Hip joints
    left_hip_path = f"{robot_path}/LeftHipJoint"
    left_hip = PhysxSchema.PhysxJoint.Create(stage, left_hip_path)
    left_hip.GetActor0Rel().SetTargets([f"{robot_path}/Body"])
    left_hip.GetActor1Rel().SetTargets([f"{robot_path}/LeftThigh"])
    left_hip.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.05, 0, 0.2))
    left_hip.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0.25))

    # Knee joints
    left_knee_path = f"{robot_path}/LeftKneeJoint"
    left_knee = PhysxSchema.PhysxJoint.Create(stage, left_knee_path)
    left_knee.GetActor0Rel().SetTargets([f"{robot_path}/LeftThigh"])
    left_knee.GetActor1Rel().SetTargets([f"{robot_path}/LeftCalf"])
    left_knee.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, -0.25))
    left_knee.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0.225))
```

## Sensor Integration

### Camera Sensors

Isaac Sim provides high-quality camera simulation with realistic optical properties:

```python
from omni.isaac.sensor import Camera
import carb

def add_camera_to_robot(robot_path, camera_name, position, orientation):
    """Add a camera sensor to the robot"""

    # Create camera prim
    camera_path = f"{robot_path}/{camera_name}"
    camera = Camera(
        prim_path=camera_path,
        frequency=30,  # Hz
        resolution=(640, 480),
        position=position,
        orientation=orientation
    )

    # Configure camera properties
    camera_config = {
        'focal_length': 24.0,  # mm
        'horizontal_aperture': 36.0,  # mm
        'f_stop': 1.4,  # Aperture
        'focus_distance': 10.0,  # meters
        'iso': 100,  # ISO setting
        'shutter_speed': 1/60,  # seconds
    }

    # Apply configuration
    for prop, value in camera_config.items():
        try:
            camera._sensor_prim.GetAttribute(f"primvars:{prop}").Set(value)
        except:
            carb.log_warn(f"Could not set camera property {prop}")

    return camera

def setup_rgbd_camera(robot_path, name="rgbd_camera"):
    """Setup RGB-D camera with depth sensing capabilities"""

    # Position camera at head level
    camera_position = [0.05, 0.0, 1.35]  # Slightly forward and at head height
    camera_orientation = [0.707, 0.0, 0.707, 0.0]  # Looking forward (quaternion)

    camera = add_camera_to_robot(robot_path, name, camera_position, camera_orientation)

    # Configure depth sensing
    # Isaac Sim has built-in depth rendering capabilities
    # Depth is automatically generated alongside RGB

    return camera

def setup_stereo_camera(robot_path, baseline=0.1):
    """Setup stereo camera system for depth perception"""

    # Left camera
    left_cam = add_camera_to_robot(
        robot_path,
        "left_camera",
        [0.05, baseline/2, 1.35],  # Offset horizontally by half baseline
        [0.707, 0.0, 0.707, 0.0]
    )

    # Right camera
    right_cam = add_camera_to_robot(
        robot_path,
        "right_camera",
        [0.05, -baseline/2, 1.35],  # Offset horizontally by negative half baseline
        [0.707, 0.0, 0.707, 0.0]
    )

    return left_cam, right_cam
```

### LiDAR Sensors

LiDAR simulation is crucial for robotics applications:

```python
from omni.isaac.range_sensor import _range_sensor
import numpy as np

class LiDARSimulator:
    def __init__(self, robot_path, sensor_name="lidar", parent_path=None):
        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        self.robot_path = robot_path
        self.sensor_name = sensor_name
        self.parent_path = parent_path or robot_path

        # Default LiDAR parameters
        self.params = {
            'rotation_count': 1,  # Number of full rotations to accumulate
            'rows': 16,  # Vertical channels (for multi-line LiDAR)
            'horizontal_pixels': 1024,  # Horizontal resolution
            'horizontal_fov': 360,  # Horizontal field of view in degrees
            'vertical_fov': 30,  # Vertical field of view in degrees
            'range': 25.0,  # Maximum range in meters
            'min_range': 0.1,  # Minimum range in meters
            'upper_fov': 15,  # Upper vertical FOV limit
            'lower_fov': -15,  # Lower vertical FOV limit
            'rotation_frequency': 10,  # Hz
            'measurements_per_rotation': 1024,  # Samples per rotation
        }

        # Create LiDAR sensor in the scene
        self.create_lidar_sensor()

    def create_lidar_sensor(self):
        """Create LiDAR sensor in Isaac Sim"""
        sensor_path = f"{self.parent_path}/{self.sensor_name}"

        # Add LiDAR to stage
        self.lidar_interface.add_lidar_to_stage(
            prim_path=sensor_path,
            sensor_config=self.params
        )

        # Position the LiDAR on the robot (typically on top or front)
        self.position_lidar(sensor_path)

    def position_lidar(self, sensor_path):
        """Position the LiDAR sensor on the robot"""
        import omni
        from pxr import UsdGeom, Gf

        stage = omni.usd.get_context().get_stage()
        lidar_prim = stage.GetPrimAtPath(sensor_path)

        # Position LiDAR at robot's head level, centered
        if lidar_prim.IsValid():
            # Set position
            UsdGeom.XformCommonAPI(lidar_prim).SetTranslate(Gf.Vec3f(0.1, 0.0, 1.3))  # Slightly forward and at head height

            # Set orientation (typically looking forward)
            UsdGeom.XformCommonAPI(lidar_prim).SetRotate(Gf.Vec3f(0, 0, 0))  # No rotation (looking forward)

    def get_point_cloud(self):
        """Get point cloud data from the LiDAR"""
        sensor_path = f"{self.parent_path}/{self.sensor_name}"
        return self.lidar_interface.get_point_cloud_data(sensor_path)

    def get_laser_scan(self):
        """Get 2D laser scan data from the LiDAR"""
        sensor_path = f"{self.parent_path}/{self.sensor_name}"
        return self.lidar_interface.get_laser_scan_data(sensor_path)

    def configure_sensor(self, **kwargs):
        """Configure LiDAR parameters"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # Reconfigure the sensor with new parameters
        sensor_path = f"{self.parent_path}/{self.sensor_name}"
        self.lidar_interface.remove_sensor(sensor_path)
        self.lidar_interface.add_lidar_to_stage(
            prim_path=sensor_path,
            sensor_config=self.params
        )

# Example of creating different LiDAR configurations
def create_robot_with_sensors(robot_path):
    """Create a robot with multiple sensor types"""

    # Add RGB-D camera
    rgbd_camera = setup_rgbd_camera(robot_path, "head_camera")

    # Add front-facing LiDAR
    front_lidar = LiDARSimulator(robot_path, "front_lidar")
    front_lidar.configure_sensor(
        rows=32,
        horizontal_pixels=2048,
        range=30.0
    )

    # Add 360-degree LiDAR on top
    top_lidar = LiDARSimulator(robot_path, "top_lidar")
    top_lidar.configure_sensor(
        rows=64,
        horizontal_pixels=2048,
        range=25.0,
        vertical_fov=45
    )

    # Position top LiDAR on the head/upper body
    import omni
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    top_lidar_prim = stage.GetPrimAtPath(f"{robot_path}/top_lidar")
    if top_lidar_prim.IsValid():
        UsdGeom.XformCommonAPI(top_lidar_prim).SetTranslate(Gf.Vec3f(0.0, 0.0, 1.45))  # On top of head

    return {
        'rgbd_camera': rgbd_camera,
        'front_lidar': front_lidar,
        'top_lidar': top_lidar
    }
```

### IMU and Force/Torque Sensors

For humanoid robotics, inertial and force sensors are essential:

```python
def add_imu_to_robot(robot_path, link_name="Body", sensor_name="imu"):
    """Add IMU sensor to a robot link"""

    import omni
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
    from omni.isaac.core.utils.prims import get_prim_at_path

    stage = omni.usd.get_context().get_stage()

    # IMU is typically attached to the main body of the robot
    link_path = f"{robot_path}/{link_name}"
    link_prim = stage.GetPrimAtPath(link_path)

    if not link_prim.IsValid():
        carb.log_error(f"Link {link_path} does not exist")
        return None

    # Create IMU sensor prim as a child of the link
    imu_path = f"{link_path}/{sensor_name}"
    imu_prim = UsdGeom.Xform.Define(stage, imu_path)

    # Position IMU at the center of the link
    UsdGeom.XformCommonAPI(imu_prim).SetTranslate(Gf.Vec3f(0, 0, 0))

    # In Isaac Sim, IMU data is typically obtained through the physics engine
    # This is a conceptual representation - actual implementation uses physics callbacks

    return imu_path

def add_force_torque_sensors(robot_path):
    """Add force/torque sensors to robot joints"""

    import omni
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()

    # Add force/torque sensors to critical joints
    critical_joints = [
        f"{robot_path}/LeftHipJoint",
        f"{robot_path}/RightHipJoint",
        f"{robot_path}/LeftKneeJoint",
        f"{robot_path}/RightKneeJoint",
        f"{robot_path}/LeftShoulderJoint",
        f"{robot_path}/RightShoulderJoint"
    ]

    force_torque_sensors = []

    for joint_path in critical_joints:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if joint_prim.IsValid():
            # Enable force sensing on the joint
            physx_joint = PhysxSchema.PhysxJoint(joint_prim)
            # Note: Actual force/torque sensing in Isaac Sim requires custom plugins
            # or accessing the physics engine directly
            force_torque_sensors.append(joint_path)

    return force_torque_sensors
```

## Physics and Materials

### Physics Configuration

Proper physics configuration is essential for realistic humanoid simulation:

```python
def configure_robot_physics(robot_path):
    """Configure physics properties for humanoid robot"""

    import omni
    from pxr import UsdPhysics, UsdGeom, PhysxSchema, Gf
    from omni.isaac.core.utils.stage import get_current_stage

    stage = get_current_stage()

    # Configure global physics settings
    scene_path = "/World/PhysicsScene"
    scene = UsdPhysics.Scene.Define(stage, scene_path)

    # Set gravity (Earth normal)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Physics solver settings
    scene.CreateEnableCCDAttr().Set(True)  # Enable continuous collision detection
    scene.CreateEnableStabilizationAttr().Set(True)  # Enable stabilization
    scene.CreateEnableAdaptiveForceAttr().Set(False)  # Disable adaptive force for consistency

    # Configure individual robot parts
    configure_robot_part_physics(stage, robot_path)

def configure_robot_part_physics(stage, robot_path):
    """Configure physics properties for individual robot parts"""

    # Define material properties for different parts
    part_properties = {
        'Body': {
            'mass': 15.0,  # kg
            'linear_damping': 0.05,
            'angular_damping': 0.1,
            'restitution': 0.2,  # Bounciness
            'friction': 0.8  # Friction coefficient
        },
        'Head': {
            'mass': 2.0,
            'linear_damping': 0.1,
            'angular_damping': 0.15,
            'restitution': 0.1,
            'friction': 0.6
        },
        'UpperArm': {
            'mass': 1.5,
            'linear_damping': 0.1,
            'angular_damping': 0.2,
            'restitution': 0.1,
            'friction': 0.7
        },
        'LowerArm': {
            'mass': 1.0,
            'linear_damping': 0.15,
            'angular_damping': 0.25,
            'restitution': 0.1,
            'friction': 0.7
        },
        'Thigh': {
            'mass': 3.0,
            'linear_damping': 0.05,
            'angular_damping': 0.1,
            'restitution': 0.1,
            'friction': 0.9  # Higher friction for feet/grippers
        },
        'Calf': {
            'mass': 2.5,
            'linear_damping': 0.05,
            'angular_damping': 0.1,
            'restitution': 0.1,
            'friction': 0.9
        }
    }

    # Apply properties to each part
    parts = ['Body', 'Head', 'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm',
             'LeftThigh', 'RightThigh', 'LeftCalf', 'RightCalf']

    for part in parts:
        part_path = f"{robot_path}/{part}"
        part_prim = stage.GetPrimAtPath(part_path)

        if part_prim.IsValid():
            # Determine which property template to use
            prop_key = part
            if 'UpperArm' in part or 'LowerArm' in part:
                prop_key = 'UpperArm' if 'Upper' in part else 'LowerArm'
            elif 'Thigh' in part or 'Calf' in part:
                prop_key = 'Thigh' if 'Thigh' in part else 'Calf'

            if prop_key in part_properties:
                props = part_properties[prop_key]

                # Apply mass
                mass_api = UsdPhysics.MassAPI.Apply(part_prim)
                mass_api.CreateMassAttr().Set(props['mass'])

                # Apply damping
                body_api = UsdPhysics.RigidBodyAPI.Apply(part_prim)
                body_api.CreateLinearDampingAttr().Set(props['linear_damping'])
                body_api.CreateAngularDampingAttr().Set(props['angular_damping'])

                # Apply material properties
                collision_api = UsdPhysics.CollisionAPI.Apply(part_prim)
                collision_api.CreateRestitutionAttr().Set(props['restitution'])

                # Apply PhysX-specific properties
                physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(part_prim)
                physx_body.CreateSleepThresholdAttr().Set(0.01)
                physx_body.CreateStabilizationThresholdAttr().Set(0.01)

def create_realistic_materials():
    """Create realistic materials for robot and environment"""

    import omni
    from pxr import UsdShade, Sdf

    stage = omni.usd.get_context().get_stage()

    # Robot body material (metallic)
    robot_material_path = "/World/Looks/RobotMaterial"
    robot_material = UsdShade.Material.Define(stage, robot_material_path)

    # Create shader
    shader = UsdShade.Shader.Define(stage, f"{robot_material_path}/Shader")
    shader.SetShaderId("OmniPBR")

    # Set material properties
    shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set((0.7, 0.7, 0.8))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.5)

    # Connect shader to material
    robot_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")

    # Floor material (for traction)
    floor_material_path = "/World/Looks/FloorMaterial"
    floor_material = UsdShade.Material.Define(stage, floor_material_path)

    floor_shader = UsdShade.Shader.Define(stage, f"{floor_material_path}/Shader")
    floor_shader.SetShaderId("OmniPBR")

    floor_shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set((0.3, 0.3, 0.3))
    floor_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)  # High friction
    floor_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    floor_material.CreateSurfaceOutput().ConnectToSource(floor_shader.ConnectableAPI(), "out")

    return {
        'robot_material': robot_material_path,
        'floor_material': floor_material_path
    }

def apply_materials_to_robot(robot_path, materials):
    """Apply materials to robot parts"""

    import omni
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()

    # Apply robot material to all body parts
    parts = ['Body', 'Head', 'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm',
             'LeftThigh', 'RightThigh', 'LeftCalf', 'RightCalf']

    for part in parts:
        part_path = f"{robot_path}/{part}"
        part_prim = stage.GetPrimAtPath(part_path)

        if part_prim.IsValid():
            # Apply material binding
            UsdShade.MaterialBindingAPI(part_prim).Bind(
                stage.GetPrimAtPath(materials['robot_material'])
            )
```

## ROS2 Integration

### Isaac ROS Bridge Setup

The Isaac ROS bridge provides seamless integration between Isaac Sim and ROS2:

```python
# ros_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
from cv_bridge import CvBridge
import tf2_ros

class IsaacSimROSBridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_bridge')

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers for simulated sensors
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/pointcloud', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Subscribers for robot commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Timer for publishing sensor data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10Hz

        # Robot state
        self.robot_position = [0.0, 0.0, 0.0]
        self.robot_orientation = [0.0, 0.0, 0.0, 1.0]  # quaternion
        self.robot_twist = [0.0, 0.0, 0.0]  # linear velocities
        self.last_update_time = self.get_clock().now()

        self.get_logger().info('Isaac Sim ROS Bridge initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS2"""
        # In a real implementation, this would interface with Isaac Sim's physics engine
        # to control the simulated robot
        linear_vel = [msg.linear.x, msg.linear.y, msg.linear.z]
        angular_vel = [msg.angular.x, msg.angular.y, msg.angular.z]

        # Update robot state based on commanded velocities
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time

        # Simple kinematic update (in practice, this would use physics simulation)
        self.robot_position[0] += linear_vel[0] * dt
        self.robot_position[1] += linear_vel[1] * dt
        # Update orientation based on angular velocity
        # This is a simplified approach - in practice, use quaternion integration

        self.get_logger().debug(f'Received cmd_vel: linear={linear_vel}, angular={angular_vel}')

    def publish_sensor_data(self):
        """Publish simulated sensor data to ROS2 topics"""
        current_time = self.get_clock().now()

        # Publish camera image (simulated)
        self.publish_camera_image(current_time)

        # Publish LiDAR scan (simulated)
        self.publish_laser_scan(current_time)

        # Publish IMU data (simulated)
        self.publish_imu_data(current_time)

        # Publish odometry (simulated)
        self.publish_odometry(current_time)

        # Broadcast TF transforms
        self.broadcast_transforms(current_time)

    def publish_camera_image(self, stamp):
        """Publish simulated camera image"""
        # In a real implementation, this would get image data from Isaac Sim
        # For now, create a dummy image
        import cv2
        import numpy as np

        # Create a dummy image (in practice, get from Isaac Sim camera)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[:] = [100, 100, 100]  # Gray background

        # Add some simulated features
        cv2.circle(dummy_image, (320, 240), 50, (255, 0, 0), -1)  # Blue circle

        ros_image = self.bridge.cv2_to_imgmsg(dummy_image, encoding="bgr8")
        ros_image.header.stamp = stamp.to_msg()
        ros_image.header.frame_id = "camera_link"

        self.image_pub.publish(ros_image)

    def publish_laser_scan(self, stamp):
        """Publish simulated laser scan data"""
        scan_msg = LaserScan()
        scan_msg.header.stamp = stamp.to_msg()
        scan_msg.header.frame_id = "laser_link"

        # Configure scan parameters
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = 2 * np.pi / 1024  # 1024 points for 360 degrees
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1  # 10Hz
        scan_msg.range_min = 0.1
        scan_msg.range_max = 25.0

        # Generate simulated ranges (in practice, get from Isaac Sim LiDAR)
        num_ranges = 1024
        ranges = []

        for i in range(num_ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            # Simulate environment with some obstacles
            distance = 10.0  # Default range

            # Add some simulated obstacles
            if 0.5 < abs(angle) < 0.6:  # Simulated wall at specific angles
                distance = 2.0
            elif abs(angle) < 0.1:  # Simulated object straight ahead
                distance = 1.5
            elif 1.5 < abs(angle) < 1.6:  # Another obstacle
                distance = 3.0

            ranges.append(distance)

        scan_msg.ranges = ranges
        scan_msg.intensities = [100.0] * num_ranges  # Dummy intensities

        self.scan_pub.publish(scan_msg)

    def publish_imu_data(self, stamp):
        """Publish simulated IMU data"""
        imu_msg = Imu()
        imu_msg.header.stamp = stamp.to_msg()
        imu_msg.header.frame_id = "imu_link"

        # In a real simulation, get actual IMU readings from Isaac Sim
        # For now, simulate realistic values

        # Orientation (in practice, integrate from gyro)
        imu_msg.orientation.x = self.robot_orientation[0]
        imu_msg.orientation.y = self.robot_orientation[1]
        imu_msg.orientation.z = self.robot_orientation[2]
        imu_msg.orientation.w = self.robot_orientation[3]

        # Angular velocity (gyroscope simulation)
        imu_msg.angular_velocity.x = 0.0  # Assuming no rotation for simplicity
        imu_msg.angular_velocity.y = 0.0
        imu_msg.angular_velocity.z = 0.0

        # Linear acceleration (accelerometer simulation)
        # Include gravity in z-direction
        imu_msg.linear_acceleration.x = 0.0
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = 9.81  # Gravity

        self.imu_pub.publish(imu_msg)

    def publish_odometry(self, stamp):
        """Publish simulated odometry data"""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Position
        odom_msg.pose.pose.position.x = self.robot_position[0]
        odom_msg.pose.pose.position.y = self.robot_position[1]
        odom_msg.pose.pose.position.z = self.robot_position[2]

        # Orientation
        odom_msg.pose.pose.orientation.x = self.robot_orientation[0]
        odom_msg.pose.pose.orientation.y = self.robot_orientation[1]
        odom_msg.pose.pose.orientation.z = self.robot_orientation[2]
        odom_msg.pose.pose.orientation.w = self.robot_orientation[3]

        # Velocity (in practice, derive from physics simulation)
        odom_msg.twist.twist.linear.x = self.robot_twist[0]
        odom_msg.twist.twist.linear.y = self.robot_twist[1]
        odom_msg.twist.twist.linear.z = self.robot_twist[2]

        # For now, set dummy covariance values
        odom_msg.pose.covariance = [0.0] * 36
        odom_msg.twist.covariance = [0.0] * 36

        self.odom_pub.publish(odom_msg)

    def broadcast_transforms(self, stamp):
        """Broadcast TF transforms"""
        # Robot base link to camera
        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "camera_link"
        t.transform.translation.x = 0.1  # Camera offset forward
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.3  # Camera height
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

        # Robot base link to LiDAR
        t2 = TransformStamped()
        t2.header.stamp = stamp.to_msg()
        t2.header.frame_id = "base_link"
        t2.child_frame_id = "laser_link"
        t2.transform.translation.x = 0.0  # LiDAR at center
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 1.4  # LiDAR height
        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = 0.0
        t2.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t2)

        # Robot base link to IMU
        t3 = TransformStamped()
        t3.header.stamp = stamp.to_msg()
        t3.header.frame_id = "base_link"
        t3.child_frame_id = "imu_link"
        t3.transform.translation.x = 0.0  # IMU at body center
        t3.transform.translation.y = 0.0
        t3.transform.translation.z = 1.0  # IMU height
        t3.transform.rotation.x = 0.0
        t3.transform.rotation.y = 0.0
        t3.transform.rotation.z = 0.0
        t3.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t3)

def main(args=None):
    rclpy.init(args=args)

    bridge = IsaacSimROSBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Extensions

Isaac ROS provides specialized packages for various robotic functions:

```python
# perception_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

class IsaacSimPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_sim_perception_pipeline')

        self.bridge = CvBridge()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.object_pub = self.create_publisher(MarkerArray, '/detected_objects', 10)
        self.free_space_pub = self.create_publisher(MarkerArray, '/free_space', 10)

        # Internal state
        self.latest_depth = None
        self.depth_timestamp = None

        self.get_logger().info('Isaac Sim Perception Pipeline initialized')

    def image_callback(self, msg):
        """Process camera image for object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection (using OpenCV for this example)
            # In practice, use a trained neural network (YOLO, SSD, etc.)
            detections = self.simple_color_detection(cv_image)

            # Publish detected objects as markers
            self.publish_object_markers(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # Convert ROS image to OpenCV (depth format)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Store for use with other sensors
            self.latest_depth = depth_image
            self.depth_timestamp = msg.header.stamp

        except Exception as e:
            self.get_logger().error(f'Error processing depth: {str(e)}')

    def scan_callback(self, msg):
        """Process LiDAR scan for obstacle detection and free space mapping"""
        try:
            # Convert scan to point cloud in 2D
            points_2d = self.scan_to_2d_points(msg)

            # Cluster points to identify obstacles
            obstacle_clusters = self.cluster_scan_points(points_2d)

            # Identify free space (areas without obstacles)
            free_space_regions = self.identify_free_space(points_2d, msg)

            # Publish results
            self.publish_obstacle_markers(obstacle_clusters, msg.header)
            self.publish_free_space_markers(free_space_regions, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing scan: {str(e)}')

    def simple_color_detection(self, image):
        """Simple color-based object detection for demonstration"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for blue objects (example)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small detections
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                detection = {
                    'bbox': [x, y, x+w, y+h],
                    'center': [x + w//2, y + h//2],
                    'area': area,
                    'contour': contour
                }
                detections.append(detection)

        return detections

    def scan_to_2d_points(self, scan_msg):
        """Convert laser scan to 2D points in robot frame"""
        points = []

        angle = scan_msg.angle_min
        for i, range_val in enumerate(scan_msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)) and scan_msg.range_min <= range_val <= scan_msg.range_max:
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                points.append([x, y])

            angle += scan_msg.angle_increment

        return np.array(points)

    def cluster_scan_points(self, points):
        """Cluster LiDAR points to identify individual obstacles"""
        if len(points) < 2:
            return []

        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise points
                continue

            if label not in clusters:
                clusters[label] = {'points': [], 'centroid': None, 'bbox': None}

            clusters[label]['points'].append(points[i])

        # Calculate properties for each cluster
        for label, cluster in clusters.items():
            cluster_points = np.array(cluster['points'])
            cluster['centroid'] = np.mean(cluster_points, axis=0)

            # Calculate bounding box
            min_pt = np.min(cluster_points, axis=0)
            max_pt = np.max(cluster_points, axis=0)
            cluster['bbox'] = [min_pt[0], min_pt[1], max_pt[0], max_pt[1]]

        return clusters

    def identify_free_space(self, obstacle_points, scan_msg):
        """Identify free space in the environment"""
        # Create a polar grid of potential free space points
        free_space_points = []

        # Sample points along each scan ray up to the detected obstacle
        angle = scan_msg.angle_min
        for i, range_val in enumerate(scan_msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)) and scan_msg.range_min <= range_val <= scan_msg.range_max:
                # Points up to the obstacle are potentially free
                for dist in np.linspace(0.1, range_val - 0.1, num=max(1, int(range_val * 10))):
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)
                    free_space_points.append([x, y])
            elif np.isnan(range_val) or range_val > scan_msg.range_max:
                # If no obstacle detected, sample up to max range
                for dist in np.linspace(0.1, scan_msg.range_max - 0.1, num=10):
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)
                    free_space_points.append([x, y])

            angle += scan_msg.angle_increment

        return np.array(free_space_points)

    def publish_object_markers(self, detections, header):
        """Publish detected objects as visualization markers"""
        marker_array = MarkerArray()

        for i, detection in enumerate(detections):
            # Bounding box marker
            bbox_marker = Marker()
            bbox_marker.header = header
            bbox_marker.ns = "detected_objects_bbox"
            bbox_marker.id = i * 2
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD

            # Define rectangle points
            x1, y1, x2, y2 = detection['bbox']
            points = [
                PointStamped(point=[x1, y1, 0.1]).point,
                PointStamped(point=[x2, y1, 0.1]).point,
                PointStamped(point=[x2, y2, 0.1]).point,
                PointStamped(point=[x1, y2, 0.1]).point,
                PointStamped(point=[x1, y1, 0.1]).point  # Close the loop
            ]
            bbox_marker.points = points

            bbox_marker.scale.x = 0.05  # Line width
            bbox_marker.color.r = 1.0
            bbox_marker.color.g = 0.0
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = 0.8

            # Center point marker
            center_marker = Marker()
            center_marker.header = header
            center_marker.ns = "detected_objects_center"
            center_marker.id = i * 2 + 1
            center_marker.type = Marker.SPHERE
            center_marker.action = Marker.ADD

            center_x = detection['center'][0]
            center_y = detection['center'][1]
            center_marker.pose.position.x = center_x
            center_marker.pose.position.y = center_y
            center_marker.pose.position.z = 0.1
            center_marker.pose.orientation.w = 1.0

            center_marker.scale.x = 0.2
            center_marker.scale.y = 0.2
            center_marker.scale.z = 0.2

            center_marker.color.r = 1.0
            center_marker.color.g = 0.0
            center_marker.color.b = 0.0
            center_marker.color.a = 0.8

            marker_array.markers.extend([bbox_marker, center_marker])

        self.object_pub.publish(marker_array)

    def publish_obstacle_markers(self, obstacle_clusters, header):
        """Publish obstacle clusters as visualization markers"""
        marker_array = MarkerArray()

        for i, (label, cluster) in enumerate(obstacle_clusters.items()):
            # Centroid marker
            centroid_marker = Marker()
            centroid_marker.header = header
            centroid_marker.ns = "obstacle_centroids"
            centroid_marker.id = i * 2
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD

            centroid_marker.pose.position.x = cluster['centroid'][0]
            centroid_marker.pose.position.y = cluster['centroid'][1]
            centroid_marker.pose.position.z = 0.5  # Height above ground
            centroid_marker.pose.orientation.w = 1.0

            centroid_marker.scale.x = 0.3
            centroid_marker.scale.y = 0.3
            centroid_marker.scale.z = 0.3

            centroid_marker.color.r = 1.0
            centroid_marker.color.g = 0.0
            centroid_marker.color.b = 0.0
            centroid_marker.color.a = 0.8

            # Bounding box marker
            bbox_marker = Marker()
            bbox_marker.header = header
            bbox_marker.ns = "obstacle_bboxes"
            bbox_marker.id = i * 2 + 1
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD

            x1, y1, x2, y2 = cluster['bbox']
            points = [
                PointStamped(point=[x1, y1, 0.1]).point,
                PointStamped(point=[x2, y1, 0.1]).point,
                PointStamped(point=[x2, y2, 0.1]).point,
                PointStamped(point=[x1, y2, 0.1]).point,
                PointStamped(point=[x1, y1, 0.1]).point
            ]
            bbox_marker.points = points

            bbox_marker.scale.x = 0.05
            bbox_marker.color.r = 1.0
            bbox_marker.color.g = 0.5
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = 0.8

            marker_array.markers.extend([centroid_marker, bbox_marker])

        self.object_pub.publish(marker_array)

    def publish_free_space_markers(self, free_space_points, header):
        """Publish free space regions as visualization markers"""
        # Due to the large number of free space points, we'll create a grid-based representation
        # For visualization purposes, we'll sample points on a grid

        marker_array = MarkerArray()

        # Create a sparse sampling of free space points
        sampled_points = free_space_points[::50]  # Take every 50th point to reduce count

        for i, point in enumerate(sampled_points):
            marker = Marker()
            marker.header = header
            marker.ns = "free_space"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.05  # Slightly above ground
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1

            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.3  # Semi-transparent

            marker_array.markers.append(marker)

        self.free_space_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)

    perception_pipeline = IsaacSimPerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()