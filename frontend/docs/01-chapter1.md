---
sidebar_label: 'Chapter 1: Introduction to ROS2'
sidebar_position: 2
---

# Chapter 1: Introduction to ROS2

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the fundamental concepts of ROS2
- Describe the architecture of ROS2 systems
- Identify the key differences between ROS1 and ROS2
- Set up a basic ROS2 development environment
- Create and run your first ROS2 package

## Table of Contents
1. [What is ROS2?](#what-is-ros2)
2. [ROS2 Architecture](#ros2-architecture)
3. [ROS1 vs ROS2](#ros1-vs-ros2)
4. [Setting Up ROS2](#setting-up-ros2)
5. [Your First ROS2 Package](#your-first-ros2-package)
6. [Lab Exercise](#lab-exercise)
7. [Summary](#summary)
8. [Quiz](#quiz)

## What is ROS2?

Robot Operating System 2 (ROS2) is not an operating system but rather a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Features of ROS2

- **Distributed Computing**: ROS2 allows different processes to communicate seamlessly, whether they're running on the same machine or across a network.
- **Language Independence**: ROS2 supports multiple programming languages including C++, Python, and more.
- **Real-time Support**: Unlike ROS1, ROS2 has real-time capabilities essential for robotics applications.
- **Improved Security**: ROS2 includes security features out of the box.

## ROS2 Architecture

ROS2 is built on Data Distribution Service (DDS), which provides a publish-subscribe communication pattern. The architecture consists of:

- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Messages**: ROS data types used when publishing or subscribing to a Topic
- **Services**: Synchronous request/response communication between nodes
- **Actions**: Asynchronous communication for long-running tasks

### Communication Patterns

1. **Publish-Subscribe**: Used for streaming data
2. **Request-Response**: Used for single requests
3. **Action**: Used for long-running tasks with feedback

## ROS1 vs ROS2

| Feature | ROS1 | ROS2 |
|---------|------|------|
| Communication | Custom | DDS-based |
| Middleware | roscore | DDS implementation |
| Real-time Support | Limited | Full support |
| Security | Add-ons | Built-in |
| Cross-platform | Linux-focused | Multi-platform |
| Lifecycle Management | Manual | Built-in |

## Setting Up ROS2

### Prerequisites

- Ubuntu 22.04 (Jammy Jellyfish) or equivalent
- At least 4GB RAM
- 20GB free disk space

### Installation Steps

1. **Set up locale**
   ```bash
   locale  # Check for UTF-8
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8
   ```

2. **Setup Sources**
   ```bash
   sudo apt install software-properties-common
   sudo add-apt-repository universe
   ```

3. **Add ROS2 GPG key**
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   ```

4. **Add ROS2 repository**
   ```bash
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

5. **Install ROS2 packages**
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   ```

6. **Environment setup**
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

## Your First ROS2 Package

Let's create your first ROS2 package:

```bash
# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Create a package
colcon build
source install/setup.bash

# Create a simple publisher package
cd src
ros2 pkg create --build-type ament_python my_publisher
```

This will create a basic ROS2 package structure that you can extend.

## Lab Exercise

### Objective
Create a simple publisher-subscriber system in ROS2 that publishes "Hello, ROS2!" messages.

### Instructions
1. Create a new ROS2 package named `hello_ros2`
2. Implement a publisher node that publishes messages every 2 seconds
3. Implement a subscriber node that receives and prints the messages
4. Test the system using ROS2 tools

### Expected Outcome
You should see the subscriber node receiving messages from the publisher node.

## Summary

In this chapter, we introduced ROS2, its architecture, and key differences from ROS1. We also walked through the installation process and created our first ROS2 package. ROS2 provides a robust framework for developing complex robotic systems with improved real-time capabilities and security compared to its predecessor.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.