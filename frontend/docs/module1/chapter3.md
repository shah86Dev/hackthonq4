---
sidebar_label: 'Chapter 3: ROS2 Parameters and Launch Files'
sidebar_position: 3
---

# Chapter 3: ROS2 Parameters and Launch Files

## Learning Objectives

By the end of this chapter, you will be able to:
- Configure ROS2 nodes using parameters
- Create and use launch files to manage complex systems
- Implement parameter validation and callbacks
- Use YAML files for parameter configuration
- Debug parameter-related issues

## Table of Contents
1. [ROS2 Parameters](#ros2-parameters)
2. [Parameter Declaration and Usage](#parameter-declaration-and-usage)
3. [Launch Files](#launch-files)
4. [Parameter YAML Files](#parameter-yaml-files)
5. [Lab Exercise](#lab-exercise)
6. [Summary](#summary)
7. [Quiz](#quiz)

## ROS2 Parameters

Parameters in ROS2 allow you to configure node behavior without recompiling code. They provide a flexible way to adjust node functionality for different environments or use cases.

### Parameter Types

ROS2 supports several parameter types:
- Integer (`int`)
- Double (`double`)
- Boolean (`bool`)
- String (`string`)
- Array of integers/floats/strings

### Benefits of Parameters

- **Configuration Flexibility**: Change behavior without recompiling
- **Environment Adaptability**: Different configurations for simulation vs. real hardware
- **Runtime Adjustability**: Some parameters can be changed while the node is running

## Parameter Declaration and Usage

### Declaring Parameters

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('use_camera', True)
        self.declare_parameter('sensor_ids', [1, 2, 3])

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.use_camera = self.get_parameter('use_camera').value
        self.sensor_ids = self.get_parameter('sensor_ids').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
```

### Parameter Callbacks

You can set up callbacks to respond to parameter changes:

```python
from rclpy.parameter import Parameter
from rclpy.node import Node

class ParameterCallbackNode(Node):
    def __init__(self):
        super().__init__('parameter_callback_node')

        self.declare_parameter('target_velocity', 0.5)
        self.target_velocity = 0.5

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'target_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.target_velocity = param.value
                self.get_logger().info(f'New target velocity: {self.target_velocity}')

        return SetParametersResult(successful=True)
```

## Launch Files

Launch files allow you to start multiple nodes with specific configurations in a single command. They are essential for managing complex robotic systems.

### Basic Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='publisher_node',
            name='publisher',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42}
            ],
            remappings=[
                ('original_topic', 'new_topic')
            ]
        ),
        Node(
            package='my_package',
            executable='subscriber_node',
            name='subscriber',
            parameters=[
                os.path.join(get_package_share_directory('my_package'), 'config', 'config.yaml')
            ]
        )
    ])
```

### Advanced Launch File Features

Launch files support many advanced features:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    # Get launch arguments
    robot_namespace = LaunchConfiguration('robot_namespace').perform(context)

    return [
        Node(
            package='my_package',
            executable='robot_controller',
            name='controller',
            namespace=robot_namespace,
            parameters=[
                {'robot_namespace': robot_namespace}
            ]
        )
    ]

def generate_launch_description():
    # Declare launch arguments
    robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='robot1',
        description='Namespace for the robot'
    )

    return LaunchDescription([
        robot_namespace_arg,
        OpaqueFunction(function=launch_setup)
    ])
```

## Parameter YAML Files

YAML files provide a convenient way to store parameter configurations:

### Example YAML Configuration

```yaml
# config/robot_params.yaml
/**:
  ros__parameters:
    robot_name: "turtlebot4"
    max_linear_velocity: 0.5
    max_angular_velocity: 1.0
    use_sim_time: false

navigation:
  ros__parameters:
    planner_frequency: 1.0
    controller_frequency: 10.0
    recovery_enabled: true

sensors:
  ros__parameters:
    lidar_enabled: true
    camera_enabled: true
    imu_enabled: true
```

### Loading Parameters from YAML

```python
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import os

class YAMLParameterNode(Node):
    def __init__(self):
        super().__init__('yaml_parameter_node')

        # Get path to config file
        config_path = os.path.join(
            get_package_share_directory('my_package'),
            'config',
            'robot_params.yaml'
        )

        # Load parameters from YAML file
        self.declare_parameter('config_file', config_path)

        self.get_logger().info('Parameters loaded from YAML file')
```

## Lab Exercise

### Objective
Create a ROS2 system that uses parameters and launch files to configure a robot's behavior in different environments.

### Instructions
1. Create a robot controller node that uses parameters for configuration
2. Implement parameter validation to ensure values are within safe ranges
3. Create multiple YAML configuration files for different environments (indoor, outdoor, simulation)
4. Create a launch file that loads the appropriate configuration based on a launch argument
5. Implement parameter callbacks to respond to runtime changes

### Expected Outcome
You should have a flexible robot controller that can be easily reconfigured for different environments using launch files and YAML parameters.

## Summary

In this chapter, we explored ROS2 parameters and launch files, which are essential tools for configuring and managing robotic systems. Parameters provide a way to configure node behavior without recompiling, while launch files allow you to start complex systems with a single command. Together, these tools make ROS2 systems more flexible and maintainable.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.