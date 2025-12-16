---
sidebar_label: 'Chapter 5: Gazebo Simulation Basics'
sidebar_position: 6
---

# Chapter 5: Gazebo Simulation Basics

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up and configure Gazebo simulation environment
- Create and customize robot models for simulation
- Implement physics properties and sensors in Gazebo
- Control robots using ROS2 in simulated environments
- Debug simulation issues and optimize performance

## Table of Contents
1. [Introduction to Gazebo](#introduction-to-gazebo)
2. [Gazebo Installation and Setup](#gazebo-installation-and-setup)
3. [Robot Modeling for Simulation](#robot-modeling-for-simulation)
4. [Physics and Sensors in Gazebo](#physics-and-sensors-in-gazebo)
5. [ROS2 Integration](#ros2-integration)
6. [Lab Exercise](#lab-exercise)
7. [Summary](#summary)
8. [Quiz](#quiz)

## Introduction to Gazebo

Gazebo is a 3D simulation environment for robotics that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It is widely used in robotics research and development for testing algorithms before deployment on real robots.

### Key Features of Gazebo

- **Realistic Physics**: Accurate simulation of rigid body dynamics, contact, and friction
- **High-Quality Graphics**: Advanced rendering capabilities for visual simulation
- **Sensor Simulation**: Support for cameras, lidars, IMUs, GPS, and other sensors
- **Plugin Architecture**: Extensible through plugins for custom functionality
- **ROS Integration**: Native support for ROS and ROS2 communication

### Why Use Simulation?

Simulation offers several advantages in robotics development:
- **Safety**: Test dangerous scenarios without risk to hardware or humans
- **Cost-Effective**: Reduce costs of physical prototypes and experiments
- **Repeatability**: Create controlled, repeatable experiments
- **Speed**: Run simulations faster than real-time to accelerate development
- **Debugging**: Visualize internal states and debug complex behaviors

## Gazebo Installation and Setup

### Prerequisites

- Ubuntu 22.04 (Jammy Jellyfish) or equivalent
- ROS2 Humble Hawksbill installed
- Minimum 8GB RAM and dedicated GPU recommended

### Installation Steps

1. **Install Gazebo Garden** (recommended version for robotics):
```bash
# Add Gazebo repository
sudo apt update && sudo apt install wget lsb-release gnupg
sudo sh -c 'echo "deb [arch=amd64] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

sudo apt update
sudo apt install gz-garden
```

2. **Install Gazebo ROS2 packages**:
```bash
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-gazebo-dev
```

3. **Verify Installation**:
```bash
gz sim --version
```

### Basic Gazebo Commands

```bash
# Launch Gazebo GUI
gz sim

# Launch Gazebo without GUI (headless)
gz sim -s

# Launch with a specific world file
gz sim -r empty.sdf
```

## Robot Modeling for Simulation

### URDF vs SDF

Gazebo can use two main formats for robot models:
- **URDF (Unified Robot Description Format)**: ROS standard for robot description
- **SDF (Simulation Description Format)**: Gazebo native format with more features

### Creating a Simple Robot Model

Here's a basic URDF robot model:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheel links -->
  <link name="wheel_front">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>

  <!-- Joint connecting wheel to base -->
  <joint name="wheel_front_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_front"/>
    <origin xyz="0.2 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

### Adding Gazebo-Specific Elements

To make the robot work in Gazebo, add Gazebo-specific tags:

```xml
<!-- Add to the end of the URDF file -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
</gazebo>

<gazebo reference="wheel_front">
  <material>Gazebo/Black</material>
  <mu1>0.8</mu1>
  <mu2>0.8</mu2>
</gazebo>

<!-- Add differential drive plugin -->
<gazebo>
  <plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
    <left_joint>wheel_left_joint</left_joint>
    <right_joint>wheel_right_joint</right_joint>
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>
```

## Physics and Sensors in Gazebo

### Physics Configuration

Gazebo's physics engine can be configured in the world file:

```xml
<sdf version="1.7">
  <world name="default">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include a robot -->
    <include>
      <uri>model://simple_robot</uri>
      <pose>0 0 0.1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Adding Sensors

Common sensors in Gazebo include:

**Camera Sensor**:
```xml
<gazebo reference="camera_link">
  <sensor name="camera1" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

**Lidar Sensor**:
```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

## ROS2 Integration

### Launching Robot in Gazebo

Create a launch file to spawn your robot in Gazebo:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot_description = get_package_share_directory('my_robot_description')

    # Launch Gazebo environment
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py'),
        )
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
```

### Controlling the Robot

Once your robot is in Gazebo, you can control it using ROS2 topics:

```bash
# Send velocity commands
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'

# Check robot's odometry
ros2 topic echo /odom

# Check laser scan data
ros2 topic echo /scan
```

## Lab Exercise

### Objective
Create a differential drive robot model and control it in Gazebo simulation.

### Instructions
1. Create a URDF model of a differential drive robot with wheels
2. Add Gazebo plugins for differential drive control
3. Create a launch file to spawn the robot in Gazebo
4. Control the robot using ROS2 topics
5. Add a camera sensor and visualize the camera feed

### Expected Outcome
You should have a robot model that can be controlled in Gazebo simulation with proper ROS2 integration.

## Summary

In this chapter, we covered the basics of Gazebo simulation for robotics. We learned how to install and set up Gazebo, create robot models, configure physics and sensors, and integrate with ROS2. Simulation is a crucial tool for robotics development, allowing for safe and cost-effective testing of algorithms before deployment on real hardware.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.