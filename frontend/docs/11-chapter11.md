---
sidebar_label: 'Chapter 11: Advanced ROS2 Concepts'
sidebar_position: 12
---

# Chapter 11: Advanced ROS2 Concepts

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement custom message and service definitions
- Create and utilize ROS2 actions for complex robotic tasks
- Implement node composition for efficient resource utilization
- Develop lifecycle nodes for robust system management
- Create sophisticated launch systems for complex deployments
- Configure and optimize ROS2 middleware for specific requirements

## Table of Contents
1. [Custom Messages and Services](#custom-messages-and-services)
2. [ROS2 Actions for Long-Running Tasks](#ros2-actions-for-long-running-tasks)
3. [Node Composition and Components](#node-composition-and-components)
4. [Lifecycle Nodes](#lifecycle-nodes)
5. [Advanced Launch Systems](#advanced-launch-systems)
6. [Middleware Configuration](#middleware-configuration)
7. [Lab Exercise](#lab-exercise)
8. [Summary](#summary)
9. [Quiz](#quiz)

## Custom Messages and Services

### Understanding ROS2 Message Types

In ROS2, communication between nodes occurs through messages, services, and actions. While ROS2 provides standard message types, creating custom message types is essential for specialized robotic applications. Custom messages allow you to define complex data structures specific to your application needs.

### Message Definition Syntax

Custom messages are defined using the `.msg` file format with a specific syntax:

```txt
# Custom message definition example
# my_robot_msgs/msg/RobotStatus.msg

# Basic data types
float64 battery_voltage
float64 battery_percentage
bool is_charging
string status_message

# Arrays
float64[] joint_positions
float64[] joint_velocities
float64[] joint_efforts

# Nested messages
std_msgs/Header header
geometry_msgs/Pose current_pose
sensor_msgs/BatteryState battery_info

# Constants
uint8 STATE_IDLE = 0
uint8 STATE_ACTIVE = 1
uint8 STATE_ERROR = 2
uint8 state
```

### Creating Custom Message Packages

To create a custom message package:

1. **Create the package**:
```bash
ros2 pkg create --build-type ament_cmake my_robot_msgs
```

2. **Create message directories**:
```bash
mkdir -p my_robot_msgs/msg
mkdir -p my_robot_msgs/srv
```

3. **Define your message files** in the appropriate directories

4. **Update CMakeLists.txt**:
```cmake
find_package(rosidl_default_generators REQUIRED)

# Specify the message files to be built
set(msg_files
  "msg/RobotStatus.msg"
  "msg/SensorData.msg"
  "msg/ControlCommand.msg"
)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)
```

5. **Update package.xml**:
```xml
<buildtool_depend>ament_cmake</buildtool_depend>
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>

<member_of_group>rosidl_interface_packages</member_of_group>
```

### Using Custom Messages in Code

**C++ Example**:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "my_robot_msgs/msg/robot_status.hpp"
#include "my_robot_msgs/msg/sensor_data.hpp"

class RobotController : public rclcpp::Node
{
public:
    RobotController() : Node("robot_controller")
    {
        // Create publisher for custom message
        status_publisher_ = this->create_publisher<my_robot_msgs::msg::RobotStatus>(
            "robot_status", 10);

        // Create subscription for custom message
        sensor_subscription_ = this->create_subscription<my_robot_msgs::msg::SensorData>(
            "sensor_data", 10,
            std::bind(&RobotController::sensor_callback, this, std::placeholders::_1));

        // Timer to periodically publish status
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&RobotController::publish_status, this));
    }

private:
    void sensor_callback(const my_robot_msgs::msg::SensorData::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(),
            "Received sensor data - Range: %.2f, Temperature: %.2f",
            msg->range_data, msg->temperature);
    }

    void publish_status()
    {
        auto msg = my_robot_msgs::msg::RobotStatus();
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";
        msg.battery_voltage = 12.6;
        msg.battery_percentage = 85.0;
        msg.is_charging = false;
        msg.status_message = "Operational";
        msg.state = my_robot_msgs::msg::RobotStatus::STATE_ACTIVE;

        status_publisher_->publish(msg);
    }

    rclcpp::Publisher<my_robot_msgs::msg::RobotStatus>::SharedPtr status_publisher_;
    rclcpp::Subscription<my_robot_msgs::msg::SensorData>::SharedPtr sensor_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotController>());
    rclcpp::shutdown();
    return 0;
}
```

**Python Example**:
```python
import rclpy
from rclpy.node import Node
from my_robot_msgs.msg import RobotStatus, SensorData
from std_msgs.msg import Header
from geometry_msgs.msg import Pose

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Create publisher for custom message
        self.status_publisher = self.create_publisher(RobotStatus, 'robot_status', 10)

        # Create subscription for custom message
        self.sensor_subscription = self.create_subscription(
            SensorData,
            'sensor_data',
            self.sensor_callback,
            10
        )

        # Timer to periodically publish status
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_status)

        self.counter = 0

    def sensor_callback(self, msg):
        self.get_logger().info(f'Received sensor data - Range: {msg.range_data:.2f}, Temperature: {msg.temperature:.2f}')

    def publish_status(self):
        msg = RobotStatus()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.battery_voltage = 12.6
        msg.battery_percentage = 85.0
        msg.is_charging = False
        msg.status_message = 'Operational'
        msg.state = RobotStatus.STATE_ACTIVE  # Use the constant from the message

        # Add joint data
        msg.joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        msg.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.joint_efforts = [1.0, 1.5, 2.0, 1.0, 1.5, 2.0]

        self.status_publisher.publish(msg)
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Custom Services

Services in ROS2 provide request-response communication patterns. Custom services are defined using `.srv` files:

```txt
# my_robot_msgs/srv/RobotControl.srv

# Request
string command
float64[] parameters
---
# Response
bool success
string message
int32 error_code
```

**Service Server Implementation**:
```python
import rclpy
from rclpy.node import Node
from my_robot_msgs.srv import RobotControl

class RobotControllerService(Node):
    def __init__(self):
        super().__init__('robot_controller_service')
        self.srv = self.create_service(
            RobotControl,
            'robot_control',
            self.control_callback
        )
        self.get_logger().info('Robot Control service is ready')

    def control_callback(self, request, response):
        self.get_logger().info(f'Received command: {request.command}')

        if request.command == 'move_to':
            if len(request.parameters) >= 3:
                x, y, theta = request.parameters[:3]
                # Execute movement command
                response.success = True
                response.message = f'Moved to position ({x}, {y}, {theta})'
                response.error_code = 0
            else:
                response.success = False
                response.message = 'Insufficient parameters for move_to command'
                response.error_code = 1

        elif request.command == 'grip':
            if len(request.parameters) >= 1:
                grip_force = request.parameters[0]
                # Execute gripping command
                response.success = True
                response.message = f'Gripped with force {grip_force}'
                response.error_code = 0
            else:
                response.success = False
                response.message = 'Grip command requires force parameter'
                response.error_code = 2

        else:
            response.success = False
            response.message = f'Unknown command: {request.command}'
            response.error_code = 3

        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Service Client Implementation**:
```python
import rclpy
from rclpy.node import Node
from my_robot_msgs.srv import RobotControl

class RobotCommander(Node):
    def __init__(self):
        super().__init__('robot_commander')
        self.cli = self.create_client(RobotControl, 'robot_control')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.send_request()

    def send_request(self):
        request = RobotControl.Request()
        request.command = 'move_to'
        request.parameters = [1.0, 2.0, 1.57]  # x, y, theta

        self.future = self.cli.call_async(request)
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        if self.future.done():
            try:
                response = self.future.result()
                self.get_logger().info(f'Response: {response.message}')
            except Exception as e:
                self.get_logger().error(f'Service call failed: {e}')
            finally:
                self.get_logger().info('Shutting down')
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RobotCommander()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

## ROS2 Actions for Long-Running Tasks

Actions are designed for long-running tasks that provide feedback and can be canceled. They're perfect for navigation, manipulation, and other complex robotic operations.

### Action Definition

Create a `.action` file:

```txt
# my_robot_msgs/action/MoveRobot.action

# Goal definition
float64[] target_pose  # [x, y, theta]
float64 max_velocity
---
# Result definition
bool success
string message
float64[] final_pose  # [x, y, theta]
---
# Feedback definition
float64[] current_pose  # [x, y, theta]
float64 distance_remaining
float64 progress_percentage
```

### Action Server Implementation

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from my_robot_msgs.action import MoveRobot
import time
import math

class MoveRobotActionServer(Node):
    def __init__(self):
        super().__init__('move_robot_action_server')
        self._action_server = ActionServer(
            self,
            MoveRobot,
            'move_robot',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        """Accept or reject a goal."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Get target pose from goal
        target_pose = goal_handle.request.target_pose
        max_velocity = goal_handle.request.max_velocity

        # Initialize feedback
        feedback_msg = MoveRobot.Feedback()
        feedback_msg.current_pose = [0.0, 0.0, 0.0]  # Starting position
        feedback_msg.distance_remaining = math.sqrt(
            (target_pose[0])**2 + (target_pose[1])**2
        )
        feedback_msg.progress_percentage = 0.0

        # Simulate robot movement
        for i in range(0, 101, 5):  # Progress from 0 to 100%
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result = MoveRobot.Result()
                result.success = False
                result.message = 'Goal was canceled'
                return result

            # Update feedback
            feedback_msg.current_pose[0] = target_pose[0] * i / 100.0
            feedback_msg.current_pose[1] = target_pose[1] * i / 100.0
            feedback_msg.current_pose[2] = target_pose[2] * i / 100.0

            distance_remaining = math.sqrt(
                (target_pose[0] - feedback_msg.current_pose[0])**2 +
                (target_pose[1] - feedback_msg.current_pose[1])**2
            )
            feedback_msg.distance_remaining = distance_remaining
            feedback_msg.progress_percentage = float(i)

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Sleep to simulate movement time
            time.sleep(0.1)

        # Goal completed
        goal_handle.succeed()
        result = MoveRobot.Result()
        result.success = True
        result.message = 'Robot reached target pose successfully'
        result.final_pose = feedback_msg.current_pose

        self.get_logger().info('Goal succeeded')
        return result

def main(args=None):
    rclpy.init(args=args)
    action_server = MoveRobotActionServer()

    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Implementation

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_msgs.action import MoveRobot

class MoveRobotClient(Node):
    def __init__(self):
        super().__init__('move_robot_client')
        self._action_client = ActionClient(
            self,
            MoveRobot,
            'move_robot')

    def send_goal(self, target_pose, max_velocity=1.0):
        goal_msg = MoveRobot.Goal()
        goal_msg.target_pose = target_pose
        goal_msg.max_velocity = max_velocity

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending goal request...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Current pose: [{feedback.current_pose[0]:.2f}, '
            f'{feedback.current_pose[1]:.2f}, {feedback.current_pose[2]:.2f}], '
            f'Distance remaining: {feedback.distance_remaining:.2f}, '
            f'Progress: {feedback.progress_percentage:.1f}%')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = MoveRobotClient()

    # Send a goal
    action_client.send_goal([1.0, 2.0, 1.57])

    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

## Node Composition and Components

Node composition allows multiple nodes to run within the same process, reducing inter-process communication overhead and improving performance.

### Component-Based Architecture

```cpp
// include/my_robot_components/velocity_controller_component.hpp
#ifndef VELOCITY_CONTROLLER_COMPONENT_HPP_
#define VELOCITY_CONTROLLER_COMPONENT_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

namespace my_robot_components
{

class VelocityControllerComponent : public rclcpp::Node
{
public:
  explicit VelocityControllerComponent(const rclcpp::NodeOptions & options);

private:
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void timer_callback();

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  double min_distance_;
  geometry_msgs::msg::Twist current_cmd_;
};

}  // namespace my_robot_components

#endif  // VELOCITY_CONTROLLER_COMPONENT_HPP_
```

```cpp
// src/velocity_controller_component.cpp
#include "my_robot_components/velocity_controller_component.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace my_robot_components
{

VelocityControllerComponent::VelocityControllerComponent(const rclcpp::NodeOptions & options)
: Node("velocity_controller_component", options)
{
  // Declare parameters
  this->declare_parameter<double>("min_distance", 1.0);
  min_distance_ = this->get_parameter("min_distance").as_double();

  // Create subscriptions and publishers
  scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "scan", 10,
    std::bind(&VelocityControllerComponent::scan_callback, this, std::placeholders::_1));

  cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
    "cmd_vel", 10);

  // Create timer
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&VelocityControllerComponent::timer_callback, this));
}

void VelocityControllerComponent::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  // Find minimum distance in scan
  double min_range = std::numeric_limits<double>::infinity();
  for (float range : msg->ranges) {
    if (std::isfinite(range) && range < min_range) {
      min_range = range;
    }
  }

  // Set velocity based on obstacle distance
  if (min_range < min_distance_) {
    current_cmd_.linear.x = 0.0;  // Stop if obstacle is too close
    current_cmd_.angular.z = 0.5;  // Turn to avoid obstacle
  } else {
    current_cmd_.linear.x = 0.5;   // Move forward
    current_cmd_.angular.z = 0.0;  // No turning
  }
}

void VelocityControllerComponent::timer_callback()
{
  cmd_vel_publisher_->publish(current_cmd_);
}

}  // namespace my_robot_components

// Register the component
RCLCPP_COMPONENTS_REGISTER_NODE(my_robot_components::VelocityControllerComponent)
```

### Composition Container

```cpp
// src/composition_demo.cpp
#include "rclcpp/rclcpp.hpp"
#include "composition/velocity_controller_component.hpp"
#include "composition/sensor_processor_component.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  // Create a container node
  auto container = rclcpp::Node::make_shared("composite_controller");

  // Create component nodes
  auto velocity_controller = std::make_shared<my_robot_components::VelocityControllerComponent>(
    rclcpp::NodeOptions());
  auto sensor_processor = std::make_shared<my_robot_components::SensorProcessorComponent>(
    rclcpp::NodeOptions());

  // Create executor and add nodes
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(velocity_controller);
  executor.add_node(sensor_processor);

  // Spin the executor
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
```

## Lifecycle Nodes

Lifecycle nodes provide a structured way to manage the state of nodes, enabling better system management and coordination.

### Lifecycle Node States

Lifecycle nodes follow a state machine with the following states:
- Unconfigured (1)
- Inactive (2)
- Active (3)
- Finalized (4)

And transitions:
- Configure (10)
- Cleanup (11)
- Shutdown (12)
- Activate (20)
- Deactivate (21)
- Error (30)

### Lifecycle Node Implementation

```cpp
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "lifecycle_msgs/msg/transition.hpp"

using namespace std::chrono_literals;

class LifecycleTalker : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit LifecycleTalker(const std::string & node_name)
  : rclcpp_lifecycle::LifecycleNode(node_name)
  {
    RCLCPP_INFO(get_logger(), "Creating Lifecycle Talker");
  }

protected:
  // Callback for configure transition
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Configuring Lifecycle Talker");

    // Create publisher
    pub_ = this->create_publisher<std_msgs::msg::String>("lifecycle_chatter", 10);

    // Create timer
    timer_ = this->create_wall_timer(
      1s, std::bind(&LifecycleTalker::on_timer, this));

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  // Callback for activate transition
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Activating Lifecycle Talker");

    // Activate publisher and timer
    pub_->on_activate();
    timer_->reset();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  // Callback for deactivate transition
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Deactivating Lifecycle Talker");

    // Deactivate publisher and timer
    pub_->on_deactivate();
    timer_->cancel();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  // Callback for cleanup transition
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Cleaning up Lifecycle Talker");

    // Destroy publisher and timer
    pub_.reset();
    timer_.reset();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  // Callback for shutdown transition
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Shutting down Lifecycle Talker");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

private:
  void on_timer()
  {
    static int count = 0;
    auto msg = std_msgs::msg::String();
    msg.data = "Lifecycle Hello World: " + std::to_string(++count);

    RCLCPP_INFO(get_logger(), "Lifecycle Talker Publishing: '%s'", msg.data.c_str());

    if (pub_->is_activated()) {
      pub_->publish(msg);
    }
  }

  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr pub_;
  std::shared_ptr<rclcpp::TimerBase> timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto lc_node = std::make_shared<LifecycleTalker>("lifecycle_talker");

  rclcpp::executors::SingleThreadedExecutor exe;
  exe.add_node(lc_node->get_node_base_interface());

  // In a real application, you would manage the lifecycle transitions
  // For this example, we'll just spin
  exe.spin();

  rclcpp::shutdown();
  return 0;
}
```

## Advanced Launch Systems

Launch files in ROS2 allow you to start multiple nodes with specific configurations in a single command, making system deployment much easier.

### Complex Launch File Structure

```python
# launch/advanced_robot_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch configuration variables
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')

    # Launch arguments
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for all nodes')

    # Include other launch files
    bringup_dir = get_package_share_directory('my_robot_bringup')
    controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup_dir, 'launch', 'controllers.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        parameters=[
            PathJoinSubstitution([bringup_dir, 'config', 'robot_params.yaml']),
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level]
    )

    # Velocity controller node
    velocity_controller_node = Node(
        package='my_robot_controllers',
        executable='velocity_controller',
        name='velocity_controller',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'min_distance': 0.5}
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level]
    )

    # Sensor processor node
    sensor_processor_node = Node(
        package='my_robot_perception',
        executable='sensor_processor',
        name='sensor_processor',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'sensor_types': ['lidar', 'camera']}
        ],
        remappings=[
            ('/input/laser_scan', '/scan'),
            ('/output/processed_data', '/sensor_data')
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level]
    )

    # Create container for composable nodes (if using composition)
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package='my_robot_perception',
                plugin='my_robot_perception::ImageProcessor',
                name='image_processor',
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=[('image_input', 'camera/image_raw')]
            ),
            ComposableNode(
                package='my_robot_perception',
                plugin='my_robot_perception::PointCloudProcessor',
                name='pointcloud_processor',
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=[('pointcloud_input', 'depth/points')]
            )
        ]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_log_level_cmd)

    # Add actions
    ld.add_action(controllers_launch)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(velocity_controller_node)
    ld.add_action(sensor_processor_node)
    ld.add_action(perception_container)

    return ld
```

### Conditional Launch

```python
# launch/conditional_launch.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_camera = LaunchConfiguration('use_camera')
    use_lidar = LaunchConfiguration('use_lidar')
    simulation_mode = LaunchConfiguration('simulation')

    declare_use_camera_cmd = DeclareLaunchArgument(
        'use_camera',
        default_value='true',
        description='Whether to launch camera nodes')

    declare_use_lidar_cmd = DeclareLaunchArgument(
        'use_lidar',
        default_value='true',
        description='Whether to launch lidar nodes')

    declare_simulation_cmd = DeclareLaunchArgument(
        'simulation',
        default_value='false',
        description='Whether to run in simulation mode')

    # Conditional nodes
    camera_node = Node(
        condition=IfCondition(use_camera),
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera',
        parameters=[{'video_device': '/dev/video0'}]
    )

    lidar_node = Node(
        condition=IfCondition(use_lidar),
        package='velodyne_driver',
        executable='velodyne_driver_node',
        name='velodyne_driver'
    )

    # Simulation-specific nodes
    simulation_nodes = IncludeLaunchDescription(
        condition=IfCondition(simulation_mode),
        launch_description_source=PythonLaunchDescriptionSource(
            PathJoinSubstitution([get_package_share_directory('my_robot_gazebo'), 'launch', 'sim.launch.py'])
        )
    )

    # Real hardware nodes
    hardware_nodes = IncludeLaunchDescription(
        condition=UnlessCondition(simulation_mode),
        launch_description_source=PythonLaunchDescriptionSource(
            PathJoinSubstitution([get_package_share_directory('my_robot_hardware'), 'launch', 'hw.launch.py'])
        )
    )

    ld = LaunchDescription()

    ld.add_action(declare_use_camera_cmd)
    ld.add_action(declare_use_lidar_cmd)
    ld.add_action(declare_simulation_cmd)

    ld.add_action(camera_node)
    ld.add_action(lidar_node)
    ld.add_action(simulation_nodes)
    ld.add_action(hardware_nodes)

    return ld
```

## Middleware Configuration

### DDS Configuration

ROS2 uses Data Distribution Service (DDS) as its middleware. Different DDS implementations offer different capabilities:

```xml
<!-- config/dds_config.xml -->
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
    <profiles xmlns="http://www.omg.org/dds/profiles">
        <!-- Participant profile for high-performance communication -->
        <participant profile="high_performance_profile">
            <rtps>
                <builtin>
                    <discovery_config>
                        <discovery_protocol>CLIENT</discovery_protocol>
                        <initial_peers>
                            <peer>
                                <address>192.168.1.10</address>
                            </peer>
                        </initial_peers>
                    </discovery_config>
                </builtin>

                <!-- Transport configuration -->
                <transport_descriptor>
                    <transport_id>UDPv4_TRANSPORT</transport_id>
                    <type>UDPv4</type>
                </transport_descriptor>

                <!-- QoS profile for reliable communication -->
                <default_builtin_reader_qos>
                    <durability>
                        <kind>TRANSIENT_LOCAL_DURABILITY_QOS</kind>
                    </durability>
                    <reliability>
                        <kind>RELIABLE_RELIABILITY_QOS</kind>
                    </reliability>
                    <history>
                        <kind>KEEP_LAST_HISTORY_QOS</kind>
                        <depth>100</depth>
                    </history>
                </default_builtin_reader_qos>
            </rtps>
        </participant>
    </profiles>
</dds>
```

### QoS Configuration

Quality of Service (QoS) settings allow you to configure communication behavior:

```python
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class QoSDemoNode(Node):
    def __init__(self):
        super().__init__('qos_demo_node')

        # QoS for sensor data (volatile, best effort)
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            sensor_qos
        )

        # QoS for control commands (reliable, transient local)
        control_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            control_qos
        )

        # QoS for configuration parameters (transient local, reliable)
        config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL
        )

        self.config_publisher = self.create_publisher(
            Parameter,
            'config',
            config_qos
        )

    def scan_callback(self, msg):
        # Process sensor data
        self.get_logger().info(f'Received scan with {len(msg.ranges)} points')

class QoSManager:
    def __init__(self):
        self.profiles = {
            'sensor': QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST
            ),
            'control': QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST
            ),
            'configuration': QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_ALL
            ),
            'telemetry': QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST
            )
        }

    def get_profile(self, purpose: str):
        """Get appropriate QoS profile for specific purpose"""
        return self.profiles.get(purpose, self.profiles['sensor'])
```

## Lab Exercise

### Objective
Implement a complete robotic system using advanced ROS2 concepts including custom messages, actions, lifecycle nodes, and composition.

### Instructions
1. Create a custom message package with robot-specific data types
2. Implement an action server for navigation tasks
3. Create a lifecycle node for system management
4. Develop a launch file that orchestrates the complete system
5. Test the system with various QoS configurations

### Expected Outcome
You should have a complete ROS2 system that demonstrates all advanced concepts with proper configuration and system management.

## Summary

In this chapter, we explored advanced ROS2 concepts including custom messages and services, actions for long-running tasks, node composition for efficient resource utilization, lifecycle nodes for system management, and advanced launch systems for deployment. These concepts are essential for building complex, professional robotic systems with proper architecture and system management.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.