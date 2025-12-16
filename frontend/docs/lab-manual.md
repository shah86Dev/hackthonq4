---
sidebar_label: 'Lab Manual'
sidebar_position: 103
---

# Lab Manual

This manual contains all the laboratory exercises for the Physical AI & Humanoid Robotics textbook. Each lab exercise is designed to provide hands-on experience with the concepts covered in the theoretical chapters.

## Lab Exercise Guidelines

### Before Starting a Lab

1. **Read the entire lab exercise** carefully before beginning
2. **Ensure all prerequisites** are met (software installed, hardware ready)
3. **Review safety guidelines** if working with physical robots
4. **Set up your workspace** with all necessary materials

### During the Lab

1. **Follow instructions sequentially** unless otherwise specified
2. **Take notes** of observations and unexpected behaviors
3. **Document any errors** and how you resolved them
4. **Ask questions** if you're unsure about any step

### After Completing a Lab

1. **Clean up your workspace** and properly store equipment
2. **Record your results** and any insights gained
3. **Complete the lab report** with your findings
4. **Reflect on what you learned** and how it connects to theory

## Module 1: ROS2 Labs

### Lab 1.1: ROS2 Environment Setup

**Objective**: Set up and verify a complete ROS2 development environment.

**Prerequisites**:
- Ubuntu 22.04 or equivalent Linux distribution
- Internet connection
- Administrator access to install packages

**Estimated Duration**: 60 minutes

**Steps**:
1. Verify system requirements
2. Install ROS2 Humble Hawksbill
3. Set up environment variables
4. Test basic ROS2 commands
5. Create and run a simple ROS2 package

**Expected Outcome**: A fully functional ROS2 environment with basic commands working.

### Lab 1.2: Publisher-Subscriber Communication

**Objective**: Create a publisher-subscriber system to understand ROS2 communication.

**Prerequisites**:
- Completed Lab 1.1
- Basic Python knowledge

**Estimated Duration**: 90 minutes

**Steps**:
1. Create a publisher node that sends messages
2. Create a subscriber node that receives messages
3. Test the communication between nodes
4. Modify message content and frequency
5. Use ROS2 command-line tools to monitor communication

**Expected Outcome**: Two nodes successfully communicating via ROS2 topics.

### Lab 1.3: Services and Actions

**Objective**: Implement service and action patterns for different communication needs.

**Prerequisites**:
- Completed Lab 1.1 and Lab 1.2
- Understanding of ROS2 nodes and topics

**Estimated Duration**: 120 minutes

**Steps**:
1. Create a service server and client
2. Create an action server and client
3. Compare the use cases for each communication pattern
4. Test service response times vs. action feedback
5. Implement error handling for each pattern

**Expected Outcome**: Working examples of all three communication patterns with understanding of when to use each.

### Lab 1.4: Parameters and Launch Files

**Objective**: Use parameters and launch files to configure complex ROS2 systems.

**Prerequisites**:
- Completed previous labs
- Understanding of ROS2 nodes and communication

**Estimated Duration**: 120 minutes

**Steps**:
1. Create a parameterized node
2. Write a YAML configuration file
3. Create a launch file that loads parameters
4. Test different configurations using launch arguments
5. Implement parameter validation and callbacks

**Expected Outcome**: A configurable ROS2 system that can be launched with different parameters.

## Module 2: Gazebo & Unity Labs

### Lab 2.1: Gazebo Simulation Environment

**Objective**: Learn to use Gazebo for robot simulation.

**Prerequisites**:
- ROS2 installation
- Basic understanding of physics simulation

**Estimated Duration**: 150 minutes

**Steps**:
1. Install Gazebo Garden
2. Launch a simple simulation environment
3. Spawn a robot model in the simulation
4. Control the robot using ROS2 commands
5. Add sensors to the robot model

**Expected Outcome**: Ability to create and control robots in Gazebo simulation.

### Lab 2.2: Unity Robotics Simulation

**Objective**: Explore Unity as a robotics simulation platform.

**Prerequisites**:
- Unity Hub and Unity 2022.3 LTS installed
- Unity Robotics Hub package
- ROS2 installation

**Estimated Duration**: 180 minutes

**Steps**:
1. Set up Unity Robotics package
2. Create a simple robot model in Unity
3. Connect Unity to ROS2 using ROS TCP Connector
4. Implement robot control from ROS2
5. Add physics and sensors to the Unity environment

**Expected Outcome**: Working connection between Unity simulation and ROS2.

## Module 3: NVIDIA Isaac Labs

### Lab 3.1: Isaac Sim Introduction

**Objective**: Set up and explore NVIDIA Isaac Sim for robotics simulation.

**Prerequisites**:
- NVIDIA GPU with CUDA support
- Isaac Sim installed
- Omniverse account

**Estimated Duration**: 180 minutes

**Steps**:
1. Install Isaac Sim and Omniverse
2. Launch Isaac Sim and explore the interface
3. Load a sample robot and environment
4. Control the robot using Isaac Sim tools
5. Connect to ROS2 for external control

**Expected Outcome**: Basic proficiency with Isaac Sim interface and robot control.

### Lab 3.2: Isaac ROS Integration

**Objective**: Integrate Isaac Sim with ROS2 for advanced robotics development.

**Prerequisites**:
- Completed Lab 3.1
- ROS2 knowledge
- Isaac Sim proficiency

**Estimated Duration**: 240 minutes

**Steps**:
1. Set up Isaac ROS bridge
2. Configure robot sensors for ROS2
3. Implement perception pipeline in Isaac Sim
4. Test SLAM algorithms in simulation
5. Deploy navigation stack in Isaac Sim environment

**Expected Outcome**: Complete robot perception and navigation pipeline in Isaac Sim.

## Module 4: VLA Models Labs

### Lab 4.1: Vision-Language-Action Model Integration

**Objective**: Integrate VLA models with robotic systems for embodied AI.

**Prerequisites**:
- Access to VLA model APIs
- Robot simulation environment (Isaac Sim or Gazebo)
- Python programming skills

**Estimated Duration**: 300 minutes

**Steps**:
1. Set up VLA model access
2. Create interface between VLA model and robot
3. Implement vision processing pipeline
4. Connect language understanding to action planning
5. Test complete VLA system with simulated robot

**Expected Outcome**: Working VLA system that can perceive, understand commands, and execute actions.

## Safety Guidelines

### Physical Robot Safety
- Always maintain clear sight lines to emergency stops
- Never bypass safety systems
- Ensure adequate space around operating robots
- Follow all manufacturer safety guidelines

### Software Safety
- Test code in simulation before real hardware deployment
- Implement proper error handling and timeouts
- Use version control for all code changes
- Document all safety-critical parameters

## Troubleshooting Common Issues

### ROS2 Communication Issues
- Verify ROS_DOMAIN_ID is the same for all nodes
- Check network connectivity if using multiple machines
- Ensure proper environment setup (source setup.bash)

### Simulation Performance
- Reduce simulation step size if experiencing instability
- Close unnecessary applications to free up resources
- Check GPU and CPU usage during simulation

### Hardware Connection Problems
- Verify all cables are properly connected
- Check power supply to all components
- Test individual components before system integration

## Lab Report Template

Each lab should be documented with the following sections:

1. **Objective**: What you were trying to accomplish
2. **Setup**: Hardware, software, and environment configuration
3. **Procedure**: Step-by-step description of what you did
4. **Results**: What happened during the experiment
5. **Analysis**: What the results mean and any insights gained
6. **Issues**: Problems encountered and how they were resolved
7. **Conclusion**: What you learned and how it connects to theory

## Assessment Rubric

Labs will be assessed based on:
- **Completion** (40%): Successfully completing all required steps
- **Documentation** (30%): Quality of lab report and observations
- **Understanding** (20%): Demonstration of understanding of concepts
- **Problem-solving** (10%): Ability to troubleshoot and resolve issues