---
sidebar_label: 'Chapter 2: ROS2 Nodes and Communication'
sidebar_position: 2
---

# Chapter 2: ROS2 Nodes and Communication

## Learning Objectives

By the end of this chapter, you will be able to:
- Create and manage ROS2 nodes
- Implement publisher-subscriber communication patterns
- Use services for request-response communication
- Implement action servers and clients for long-running tasks
- Debug ROS2 communication issues

## Table of Contents
1. [ROS2 Nodes](#ros2-nodes)
2. [Publisher-Subscriber Pattern](#publisher-subscriber-pattern)
3. [Services](#services)
4. [Actions](#actions)
5. [Lab Exercise](#lab-exercise)
6. [Summary](#summary)
7. [Quiz](#quiz)

## ROS2 Nodes

A node is a process that performs computation. Nodes are the fundamental building blocks of ROS2 applications. They are designed to be modular and reusable components that can be combined to create complex robotic systems.

### Node Structure

A basic ROS2 node consists of:
- Node class inheritance
- Lifecycle management
- Communication interfaces (publishers, subscribers, services, actions)
- Spin loop for processing callbacks

### Creating a Node

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Initialize node components here
        self.get_logger().info('MyNode has been started')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publisher-Subscriber Pattern

The publish-subscribe pattern is the most common communication method in ROS2. Publishers send messages to topics, and subscribers receive messages from topics.

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

## Services

Services provide a request-response communication pattern. A service client sends a request and waits for a response from the service server.

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response
```

### Creating a Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
```

## Actions

Actions are used for long-running tasks that require feedback and the ability to cancel. They provide a more sophisticated communication pattern than services.

### Action Structure

An action has three components:
- Goal: Request to start a long-running task
- Feedback: Periodic updates during task execution
- Result: Final outcome of the task

### Creating an Action Server

```python
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        # Accept or reject a goal
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept or reject a cancel request
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        # Execute the goal and provide feedback
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Lab Exercise

### Objective
Create a ROS2 system with a publisher, subscriber, service, and action to control a simulated robot's movement.

### Instructions
1. Create a publisher that sends velocity commands to move a robot
2. Create a subscriber that receives position updates from the robot
3. Create a service that accepts a goal position and returns the distance to it
4. Create an action that moves the robot to a specified goal position with feedback
5. Implement a client node that uses all communication patterns

### Expected Outcome
You should have a complete ROS2 system that demonstrates all communication patterns working together to control a simulated robot.

## Summary

In this chapter, we explored the fundamental communication patterns in ROS2: nodes, publishers/subscribers, services, and actions. Each pattern serves different purposes and is chosen based on the requirements of the robotic application. Understanding these patterns is crucial for developing effective robotic systems with ROS2.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.