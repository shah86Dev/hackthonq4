---
sidebar_label: 'Chapter 4: ROS2 Testing and Debugging'
sidebar_position: 4
---

# Chapter 4: ROS2 Testing and Debugging

## Learning Objectives

By the end of this chapter, you will be able to:
- Write unit tests for ROS2 nodes
- Create integration tests for ROS2 systems
- Use ROS2 debugging tools effectively
- Analyze system performance and identify bottlenecks
- Implement logging and monitoring for ROS2 applications

## Table of Contents
1. [ROS2 Testing Framework](#ros2-testing-framework)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [Debugging Tools](#debugging-tools)
5. [Performance Analysis](#performance-analysis)
6. [Logging and Monitoring](#logging-and-monitoring)
7. [Lab Exercise](#lab-exercise)
8. [Summary](#summary)
9. [Quiz](#quiz)

## ROS2 Testing Framework

Testing is crucial for developing reliable robotic systems. ROS2 provides several testing frameworks and tools to ensure your code works correctly.

### Testing Categories

- **Unit Tests**: Test individual functions or methods
- **Integration Tests**: Test interactions between nodes
- **System Tests**: Test complete robotic systems
- **Regression Tests**: Ensure new changes don't break existing functionality

### Testing Libraries

ROS2 uses standard testing frameworks with additional ROS2-specific extensions:
- **pytest**: For Python-based tests
- **Google Test**: For C++ tests
- **launch_testing**: For testing launch files and multi-node systems

## Unit Testing

### Basic Unit Test Example

```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_package.nodes.math_node import MathNode

class TestMathNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = MathNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_addition(self):
        # Test the addition function
        result = self.node.add(5, 3)
        self.assertEqual(result, 8)

    def test_subtraction(self):
        # Test the subtraction function
        result = self.node.subtract(10, 4)
        self.assertEqual(result, 6)
```

### Testing Publishers and Subscribers

```python
import rclpy
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from my_package.nodes.publisher_node import PublisherNode
import pytest

def test_publisher():
    rclpy.init()

    node = PublisherNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    # Wait for the publisher to be ready
    publisher = node.publisher
    assert publisher.get_subscription_count() >= 0

    # Publish a test message
    test_msg = String()
    test_msg.data = 'test'
    publisher.publish(test_msg)

    # Verify message was published
    assert publisher.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace()) is not None

    node.destroy_node()
    rclpy.shutdown()
```

## Integration Testing

Integration tests verify that multiple nodes work together correctly.

### Launch Testing Example

```python
import unittest
import launch
import launch_ros.actions
import launch_testing.actions
import pytest
from launch_testing_ros import WaitForTopics

def generate_test_description():
    """Launch the nodes under test."""
    publisher_node = launch_ros.actions.Node(
        package='my_package',
        executable='publisher_node',
        name='test_publisher'
    )

    subscriber_node = launch_ros.actions.Node(
        package='my_package',
        executable='subscriber_node',
        name='test_subscriber'
    )

    return launch.LaunchDescription([
        publisher_node,
        subscriber_node,
        launch_testing.actions.ReadyToTest()
    ])

@pytest.mark.launch_test
def test_communication():
    """Test that publisher and subscriber communicate correctly."""
    # This test verifies that messages flow between nodes
    pass

class TestCommunication(unittest.TestCase):
    def test_message_flow(self, launch_service, proc_info, proc_output):
        """Test that messages are properly exchanged between nodes."""
        with WaitForTopics([('test_topic', String)], timeout=5.0):
            # Wait for topics to become available
            pass
```

## Debugging Tools

### ROS2 Command Line Tools

ROS2 provides several command-line tools for debugging:

```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo <topic_name> <msg_type>

# Publish a message to a topic
ros2 topic pub <topic_name> <msg_type> <data>

# List all services
ros2 service list

# Call a service
ros2 service call <service_name> <srv_type> <request_data>
```

### ROS2 Doctor

The `ros2 doctor` tool helps diagnose common system issues:

```bash
# Check the entire ROS2 system
ros2 doctor

# Check specific aspects
ros2 doctor --report network
ros2 doctor --report rosdistro
```

### RQT Tools

RQT provides GUI-based debugging tools:

```bash
# General RQT interface
rqt

# Specific plugins
rqt_graph          # Visualize node graph
rqt_plot          # Plot numeric values
rqt_console       # View logs
rqt_bag           # Record and replay data
```

## Performance Analysis

### Measuring Performance

```python
import time
from rclpy.node import Node

class PerformanceNode(Node):
    def __init__(self):
        super().__init__('performance_node')
        self.times = []

    def timed_operation(self):
        start_time = time.time()

        # Perform the operation you want to measure
        result = self.expensive_computation()

        end_time = time.time()
        execution_time = end_time - start_time
        self.times.append(execution_time)

        # Log performance metrics
        self.get_logger().info(f'Operation took {execution_time:.4f} seconds')

        return result

    def expensive_computation(self):
        # Example of an expensive operation
        result = sum(i * i for i in range(10000))
        return result
```

### Using ROS2 Profiling Tools

```python
import cProfile
import pstats
from io import StringIO

class ProfilingNode(Node):
    def __init__(self):
        super().__init__('profiling_node')
        self.profiler = cProfile.Profile()

    def profile_method(self):
        self.profiler.enable()

        # Code to profile
        self.method_to_profile()

        self.profiler.disable()

        # Print profiling results
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        self.get_logger().info(s.getvalue())

    def method_to_profile(self):
        # Method to analyze for performance
        pass
```

## Logging and Monitoring

### Advanced Logging

```python
from rclpy.node import Node
from rclpy.logging import LoggingSeverity

class LoggingNode(Node):
    def __init__(self):
        super().__init__('logging_node')

        # Set up different log levels
        self.get_logger().set_level(LoggingSeverity.INFO)

    def perform_operation(self):
        self.get_logger().debug('Starting operation')

        try:
            result = self.risky_operation()
            self.get_logger().info(f'Operation completed successfully: {result}')
        except Exception as e:
            self.get_logger().error(f'Operation failed: {str(e)}')
            raise
        finally:
            self.get_logger().debug('Operation completed')

    def risky_operation(self):
        # Some operation that might fail
        return "success"
```

### System Monitoring

```python
import psutil
from rclpy.node import Node

class MonitoringNode(Node):
    def __init__(self):
        super().__init__('monitoring_node')

        # Create timer to periodically check system resources
        self.timer = self.create_timer(1.0, self.check_resources)

    def check_resources(self):
        # Monitor CPU usage
        cpu_percent = psutil.cpu_percent()

        # Monitor memory usage
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        # Monitor disk usage
        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100

        # Log resource usage
        self.get_logger().info(
            f'CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%'
        )

        # Check if resources are getting low
        if memory_percent > 80:
            self.get_logger().warn('High memory usage detected')
        if cpu_percent > 80:
            self.get_logger().warn('High CPU usage detected')
```

## Lab Exercise

### Objective
Create a comprehensive test suite and debugging setup for a ROS2 navigation system.

### Instructions
1. Create a navigation node with basic movement commands
2. Write unit tests for the navigation algorithms
3. Create integration tests that verify communication between navigation and sensor nodes
4. Implement performance monitoring in the navigation node
5. Set up logging with different severity levels
6. Use ROS2 debugging tools to analyze the system

### Expected Outcome
You should have a well-tested navigation system with comprehensive debugging and monitoring capabilities.

## Summary

In this chapter, we covered essential testing and debugging techniques for ROS2 systems. We explored unit testing, integration testing, debugging tools, performance analysis, and logging strategies. These practices are crucial for developing reliable and maintainable robotic systems.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.