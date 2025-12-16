---
sidebar_label: 'Chapter 6: Advanced Gazebo Simulation'
sidebar_position: 7
---

# Chapter 6: Advanced Gazebo Simulation

## Learning Objectives

By the end of this chapter, you will be able to:
- Create complex multi-robot simulation environments
- Implement custom Gazebo plugins for specialized functionality
- Design realistic world models with physics properties
- Integrate external controllers and perception systems
- Optimize simulation performance for large-scale scenarios

## Table of Contents
1. [Multi-Robot Simulation](#multi-robot-simulation)
2. [Custom Gazebo Plugins](#custom-gazebo-plugins)
3. [World Design and Modeling](#world-design-and-modeling)
4. [Advanced Sensor Integration](#advanced-sensor-integration)
5. [Performance Optimization](#performance-optimization)
6. [Lab Exercise](#lab-exercise)
7. [Summary](#summary)
8. [Quiz](#quiz)

## Multi-Robot Simulation

Multi-robot simulation allows for testing coordination, communication, and collaborative behaviors in a controlled environment.

### Robot Namespacing

When running multiple robots, it's crucial to namespace topics and transforms to avoid conflicts:

```xml
<!-- In your robot's URDF/Xacro -->
<group ns="$(arg robot_name)">
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher">
    <param name="robot_description" command="$(find xacro)/xacro.py $(arg model)"/>
  </node>
</group>
```

### Launch File for Multiple Robots

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch configuration variables
    world = LaunchConfiguration('world')

    # Paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot_sim = get_package_share_directory('multi_robot_sim')

    # Launch Gazebo environment
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py'),
        ),
        launch_arguments={
            'world': world
        }.items()
    )

    # Spawn Robot 1
    spawn_robot1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot1',
            '-file', os.path.join(pkg_robot_sim, 'models', 'turtlebot3_waffle.sdf'),
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # Spawn Robot 2
    spawn_robot2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot2',
            '-file', os.path.join(pkg_robot_sim, 'models', 'turtlebot3_waffle.sdf'),
            '-x', '1.0',
            '-y', '1.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(pkg_robot_sim, 'worlds', 'multi_room.world'),
            description='SDF world file'
        ),
        gazebo,
        spawn_robot1,
        spawn_robot2
    ])
```

### Communication Between Robots

Robots can communicate through ROS2 topics, services, or actions. For multi-robot coordination:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from multi_robot_msgs.msg import RobotStatus  # Custom message

class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Subscribe to status from all robots
        self.robot_status_subscribers = []
        for i in range(5):  # Assuming up to 5 robots
            sub = self.create_subscription(
                RobotStatus,
                f'/robot{i}/status',
                lambda msg, robot_id=i: self.robot_status_callback(msg, robot_id),
                10
            )
            self.robot_status_subscribers.append(sub)

        # Publish coordination commands
        self.coordination_publisher = self.create_publisher(
            String,
            '/coordination_commands',
            10
        )

    def robot_status_callback(self, msg, robot_id):
        # Process status from specific robot
        self.get_logger().info(f'Robot {robot_id} status: {msg.status}')

        # Implement coordination logic here
        if msg.status == 'idle':
            self.assign_task_to_robot(robot_id)

    def assign_task_to_robot(self, robot_id):
        # Logic to assign tasks based on robot status and capabilities
        command = f'robot{robot_id}_move_to_waypoint_1'
        self.coordination_publisher.publish(String(data=command))
```

## Custom Gazebo Plugins

Gazebo plugins extend the simulation environment with custom functionality.

### Basic Plugin Structure

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class CustomRobotPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer for convenience
      this->model = _parent;

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&CustomRobotPlugin::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply a small linear velocity to the model
      this->model->SetLinearVel(ignition::math::Vector3d(0.3, 0, 0));
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(CustomRobotPlugin)
}
```

### ROS2 Integration in Plugins

For plugins that need ROS2 communication:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace gazebo
{
  class ROS2CommunicationPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer
      this->model = _parent;

      // Initialize ROS2
      if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
      }

      // Create ROS2 node
      this->node = rclcpp::Node::make_shared("gazebo_plugin_node");

      // Create publisher
      this->publisher = this->node->create_publisher<std_msgs::msg::String>(
          "gazebo_robot_status", 10);

      // Create timer for ROS2 spinning
      auto timer_callback = [this]() -> void {
        auto message = std_msgs::msg::String();
        message.data = "Robot position: " + std::to_string(this->model->WorldPose().Pos().X());
        RCLCPP_INFO(this->node->get_logger(), "Publishing: '%s'", message.data.c_str());
        this->publisher->publish(message);
      };
      this->timer = this->node->create_wall_timer(
          std::chrono::milliseconds(500), timer_callback);

      // Connect to Gazebo update
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ROS2CommunicationPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // ROS2 spin
      rclcpp::spin_some(this->node);
    }

    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;
    private: rclcpp::Node::SharedPtr node;
    private: rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher;
    private: rclcpp::TimerBase::SharedPtr timer;
  };

  GZ_REGISTER_MODEL_PLUGIN(ROS2CommunicationPlugin)
}
```

## World Design and Modeling

### Creating Complex Environments

Gazebo worlds can be designed with complex geometry, lighting, and environmental effects:

```xml
<sdf version="1.7">
  <world name="complex_office">
    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -1</direction>
    </light>

    <!-- Models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Office furniture -->
    <include>
      <uri>model://table</uri>
      <pose>2 0 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://chair</uri>
      <pose>2.5 0.5 0 0 0 1.57</pose>
    </include>

    <!-- Custom office building -->
    <model name="office_building">
      <static>true</static>
      <link name="building_link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>model://office_building/meshes/building.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>model://office_building/meshes/building.dae</uri>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Add textures and materials -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
  </world>
</sdf>
```

### Dynamic Objects

Dynamic objects can be added to create interactive environments:

```xml
<!-- Moving platform -->
<model name="moving_platform">
  <pose>0 0 0.1 0 0 0</pose>
  <link name="platform_link">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 1 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 1 0.1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.8 0.8 0.8 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>10.0</mass>
      <inertia>
        <ixx>1.0</ixx>
        <ixy>0.0</ixy>
        <ixz>0.0</ixz>
        <iyy>1.0</iyy>
        <iyz>0.0</iyz>
        <izz>1.0</izz>
      </inertia>
    </inertial>
  </link>

  <!-- Joint to make it move -->
  <joint name="platform_slider" type="prismatic">
    <parent>world</parent>
    <child>platform_link</child>
    <axis>
      <xyz>1 0 0</xyz>
      <limit>
        <lower>-2</lower>
        <upper>2</upper>
      </limit>
    </axis>
  </joint>
</model>
```

## Advanced Sensor Integration

### Custom Sensor Plugins

Create specialized sensors for specific applications:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

namespace gazebo
{
  class CustomLaserPlugin : public SensorPlugin
  {
    public: virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/)
    {
      // Get the laser sensor
      this->laser = std::dynamic_pointer_cast<sensors::RaySensor>(_sensor);
      if (!this->laser) {
        gzerr << "CustomLaserPlugin not attached to a laser sensor\n";
        return;
      }

      // Initialize ROS2 if needed
      if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
      }

      // Create ROS2 node and publisher
      this->node = rclcpp::Node::make_shared("custom_laser_node");
      this->publisher = this->node->create_publisher<sensor_msgs::msg::LaserScan>(
          "custom_scan", 10);

      // Connect to sensor update
      this->update_connection_ = this->laser->ConnectUpdated(
          std::bind(&CustomLaserPlugin::OnScan, this));

      // Make sure the sensor is active
      this->laser->SetActive(true);
    }

    private: void OnScan()
    {
      // Get range data from the sensor
      auto ranges = this->laser->Ranges();
      auto count = this->laser->RayCount();

      // Create ROS2 message
      auto msg = sensor_msgs::msg::LaserScan();
      msg.header.stamp = this->node->get_clock()->now();
      msg.header.frame_id = "custom_laser_frame";
      msg.angle_min = this->laser->AngleMin().Radian();
      msg.angle_max = this->laser->AngleMax().Radian();
      msg.angle_increment = this->laser->AngleResolution();
      msg.time_increment = 0.0;
      msg.scan_time = 0.0;
      msg.range_min = this->laser->RangeMin();
      msg.range_max = this->laser->RangeMax();
      msg.ranges.resize(count);

      // Copy range data
      for (unsigned int i = 0; i < count; ++i) {
        msg.ranges[i] = ranges[i];
      }

      // Publish the message
      this->publisher->publish(msg);
    }

    private: sensors::RaySensorPtr laser;
    private: rclcpp::Node::SharedPtr node;
    private: rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr publisher;
    private: common::ConnectionPtr update_connection_;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomLaserPlugin)
}
```

### Perception Pipeline Integration

Connect Gazebo sensors to perception systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge

class PerceptionPipeline(Node):
    def __init__(self):
        super().__init__('perception_pipeline')

        # Subscriptions for different sensor data
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers for processed data
        self.obstacle_publisher = self.create_publisher(
            MarkerArray,
            'obstacles',
            10
        )

        self.bridge = CvBridge()
        self.obstacle_detector = ObstacleDetector()

    def scan_callback(self, msg):
        # Process laser scan data
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Detect obstacles
        obstacles = self.obstacle_detector.detect_from_scan(ranges, angles)

        # Publish visualization markers
        marker_array = self.create_obstacle_markers(obstacles)
        self.obstacle_publisher.publish(marker_array)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process image (e.g., object detection)
        detections = self.detect_objects(cv_image)

        # Visualize detections
        annotated_image = self.annotate_image(cv_image, detections)

        # Publish processed image
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        # In a real implementation, you would publish this to a topic

    def detect_objects(self, image):
        # Implement object detection (e.g., using OpenCV or a DNN)
        # This is a simplified example
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Example: simple shape detection
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({'bbox': (x, y, w, h), 'label': 'object'})

        return detections

    def annotate_image(self, image, detections):
        annotated = image.copy()
        for detection in detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated, detection['label'], (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return annotated

    def create_obstacle_markers(self, obstacles):
        marker_array = MarkerArray()
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = obstacle['x']
            marker.pose.position.y = obstacle['y']
            marker.pose.position.z = 0.5
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.3  # diameter
            marker.scale.y = 0.3  # diameter
            marker.scale.z = 1.0  # height

            marker.color.a = 0.7  # alpha
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        return marker_array
```

## Performance Optimization

### Simulation Optimization Techniques

1. **Reduce Update Rates**: Lower physics update rates for less critical simulations
2. **Simplify Models**: Use simplified collision models for performance
3. **Limit Sensor Updates**: Reduce sensor update frequency where possible
4. **Use Multi-Threading**: Enable multi-threaded physics if available

### Physics Optimization

```xml
<!-- Optimize physics for performance -->
<physics type="ode">
  <max_step_size>0.01</max_step_size>  <!-- Increase step size for performance -->
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>100</real_time_update_rate>  <!-- Lower update rate -->
  <ode>
    <solver>
      <type>quick</type>  <!-- Use faster solver -->
      <iters>10</iters>   <!-- Reduce iterations -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>      <!-- Increase ERP for stability -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Resource Management

Monitor and manage resources during simulation:

```python
import psutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimulationMonitor(Node):
    def __init__(self):
        super().__init__('simulation_monitor')

        self.monitor_publisher = self.create_publisher(
            String,
            'simulation_performance',
            10
        )

        # Timer to periodically check system resources
        self.timer = self.create_timer(1.0, self.check_resources)

    def check_resources(self):
        # Check CPU usage
        cpu_percent = psutil.cpu_percent()

        # Check memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Check disk usage
        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100

        # Create performance report
        perf_report = f"CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"

        # Log if resources are high
        if memory_percent > 80 or cpu_percent > 80:
            self.get_logger().warn(f'High resource usage detected: {perf_report}')

        # Publish performance data
        self.monitor_publisher.publish(String(data=perf_report))
```

## Lab Exercise

### Objective
Create a complex multi-robot simulation environment with custom plugins and perception systems.

### Instructions
1. Design a multi-room world with furniture and obstacles
2. Create at least 3 robots with different configurations
3. Implement a custom plugin that adds unique functionality to one robot
4. Set up a perception pipeline that processes sensor data from all robots
5. Implement basic coordination between robots

### Expected Outcome
You should have a complex simulation environment with multiple robots that can perceive their environment and coordinate their actions.

## Summary

In this chapter, we explored advanced Gazebo simulation techniques including multi-robot systems, custom plugins, complex world modeling, and performance optimization. These advanced techniques enable sophisticated robotics research and development by providing realistic, controllable simulation environments that can scale to complex multi-robot scenarios.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.