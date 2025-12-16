---
sidebar_label: 'Chapter 7: Unity for Robotics Simulation'
sidebar_position: 8
---

# Chapter 7: Unity for Robotics Simulation

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up Unity for robotics simulation using Unity Robotics Hub
- Create robot models and environments in Unity
- Integrate ROS2 with Unity for bidirectional communication
- Implement physics simulation and sensor systems in Unity
- Deploy Unity robotics applications for various platforms

## Table of Contents
1. [Introduction to Unity for Robotics](#introduction-to-unity-for-robotics)
2. [Unity Robotics Setup](#unity-robotics-setup)
3. [Robot Modeling in Unity](#robot-modeling-in-unity)
4. [ROS2 Integration](#ros2-integration)
5. [Physics and Sensors in Unity](#physics-and-sensors-in-unity)
6. [Lab Exercise](#lab-exercise)
7. [Summary](#summary)
8. [Quiz](#quiz)

## Introduction to Unity for Robotics

Unity is a powerful game engine that has been adapted for robotics simulation through the Unity Robotics Hub. It offers high-quality graphics, flexible physics simulation, and cross-platform deployment capabilities that make it attractive for robotics research and development.

### Advantages of Unity for Robotics

- **High-Quality Graphics**: Advanced rendering capabilities for realistic visual simulation
- **Flexible Physics**: Customizable physics engine with good performance
- **Cross-Platform**: Deploy to various platforms including VR/AR
- **Asset Store**: Extensive library of 3D models and components
- **Scripting**: C# scripting with access to Unity's powerful APIs
- **Community**: Large community with extensive documentation and tutorials

### Unity vs Other Simulation Platforms

| Feature | Unity | Gazebo | V-REP/CoppeliaSim |
|---------|-------|--------|-------------------|
| Graphics Quality | Excellent | Good | Good |
| Physics Accuracy | Good | Excellent | Excellent |
| ROS Integration | Good (with plugins) | Native | Good |
| Learning Curve | Moderate | Steep | Steep |
| Deployment Options | Excellent | Limited | Limited |
| Cost | Free (personal use) | Free | Free/Commercial |

## Unity Robotics Setup

### Prerequisites

- Unity Hub (latest version)
- Unity 2021.3 LTS or later
- ROS2 Humble Hawksbill installed
- Visual Studio or Visual Studio Code
- Git for Windows (if on Windows)

### Installation Steps

1. **Install Unity Hub**:
   - Download from https://unity.com/download
   - Install Unity Hub to manage Unity versions

2. **Install Unity Editor**:
   - Open Unity Hub
   - Click "Installs" → "Add"
   - Select version 2021.3 LTS or later
   - Select modules: Android Build Support (if needed), Visual Studio Tools

3. **Install Unity Robotics Hub**:
   - Open Unity Hub
   - Go to "Projects" → "New"
   - Select "3D Core" template
   - In the Unity Editor, go to Window → Package Manager
   - Click the "+" icon → "Add package from git URL"
   - Add these packages:
     - `com.unity.robotics.ros-tcp-connector` (for ROS2 communication)
     - `com.unity.robotics.urdf-importer` (for importing URDF files)

4. **Install ROS TCP Connector**:
   - This package enables communication between Unity and ROS2
   - It provides a TCP connection for bidirectional data transfer

### Setting Up the Project

```bash
# Create a new Unity project
# In Unity Hub: New Project → 3D (Core) → Name: UnityRoboticsProject

# Once in Unity Editor:
# 1. Create folders: Assets/Scripts, Assets/Models, Assets/Materials
# 2. Import ROS TCP Connector package
# 3. Set up a basic scene with lighting and basic objects
```

## Robot Modeling in Unity

### Creating a Simple Robot

Unity uses GameObjects to represent objects in the scene. For robotics, we typically create a hierarchy of linked objects:

```
Robot (Root GameObject)
├── BaseLink
│   ├── Chassis
│   ├── LeftWheel
│   ├── RightWheel
│   └── Camera
└── Sensors
    ├── Lidar
    └── IMU
```

### Basic Robot Controller Script

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    [SerializeField] private float linearVelocity = 1.0f;
    [SerializeField] private float angularVelocity = 1.0f;

    private ROSConnection ros;
    private string robotTopic = "cmd_vel";

    private Rigidbody rb;

    void Start()
    {
        // Connect to ROS
        ros = ROSConnection.instance;

        // Get rigidbody component
        rb = GetComponent<Rigidbody>();

        // Subscribe to command topic
        ros.Subscribe<TwistMsg>(robotTopic, CmdVelCallback);
    }

    void CmdVelCallback(TwistMsg cmdVel)
    {
        // Convert ROS velocity commands to Unity physics
        Vector3 linear = new Vector3((float)cmdVel.linear.y, 0, (float)cmdVel.linear.x);
        float angular = (float)cmdVel.angular.z;

        // Apply movement
        transform.Translate(linear * linearVelocity * Time.deltaTime);
        transform.Rotate(Vector3.up, angular * angularVelocity * Time.deltaTime);
    }

    void Update()
    {
        // Additional movement logic if needed
    }
}
```

### Importing URDF Models

Unity provides a URDF importer that allows you to import robot models from ROS:

1. Download the URDF Importer package from Package Manager
2. Place your URDF files in Assets/URDF folder
3. Use the URDF Importer window (Window → URDF Importer)
4. Select your URDF file and import with appropriate settings

```csharp
// Example of programmatically loading a URDF
using Unity.Robotics.UrdfImporter;
using UnityEngine;

public class UrdfLoader : MonoBehaviour
{
    [SerializeField] private string urdfPath;

    void Start()
    {
        // Load URDF file
        var robot = UrdfRobotExtensions.CreateRobotFromUrdf(urdfPath);

        // Position the robot in the scene
        robot.transform.position = Vector3.zero;
    }
}
```

### Animation and Joint Control

For articulated robots, you'll need to control joints:

```csharp
using UnityEngine;

public class JointController : MonoBehaviour
{
    [SerializeField] private ConfigurableJoint joint;
    [SerializeField] private float targetAngle = 0f;
    [SerializeField] private float moveSpeed = 100f;

    private JointDrive jointDrive;

    void Start()
    {
        jointDrive = joint.angularXDrive;
    }

    void Update()
    {
        // Update joint position
        jointDrive.targetPosition = targetAngle;
        joint.angularXDrive = jointDrive;
    }

    public void SetTargetAngle(float angle)
    {
        targetAngle = angle;
    }
}
```

## ROS2 Integration

### ROS TCP Connector Setup

The ROS TCP Connector enables communication between Unity and ROS2:

1. **In Unity**: Add ROS TCP Connector component to your scene
2. **Configure IP and Port**: Set to match your ROS2 network settings
3. **Publish/Subscribe**: Use the connector to send/receive messages

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using UnityEngine;

public class UnityRosBridge : MonoBehaviour
{
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;

        // Set ROS IP (usually localhost for local development)
        ros.rosIPAddress = "127.0.0.1";
        ros.rosPort = 10000; // Default port for ROS TCP Connector

        // Start publishers and subscribers
        InitializeRosComms();
    }

    void InitializeRosComms()
    {
        // Publisher for robot state
        ros.RegisterPublisher<OdometryMsg>("robot_odom");

        // Subscriber for velocity commands
        ros.Subscribe<TwistMsg>("cmd_vel", VelCmdCallback);

        // Publisher for sensor data
        ros.RegisterPublisher<LaserScanMsg>("scan");
    }

    void VelCmdCallback(TwistMsg cmd)
    {
        // Process velocity command
        Debug.Log($"Received velocity command: linear={cmd.linear.x}, angular={cmd.angular.z}");
    }

    void PublishOdometry()
    {
        var odomMsg = new OdometryMsg();
        odomMsg.header = new std_msgs.Header();
        odomMsg.header.stamp = new builtin_interfaces.Time();
        odomMsg.header.frame_id = "odom";

        // Set position and orientation from Unity transform
        odomMsg.pose.pose.position = new geometry_msgs.Point(
            transform.position.x,
            transform.position.y,
            transform.position.z
        );

        odomMsg.pose.pose.orientation = new geometry_msgs.Quaternion(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        );

        // Publish message
        ros.Publish("robot_odom", odomMsg);
    }
}
```

### Custom ROS Messages

Unity can work with custom ROS messages:

```csharp
// Define a custom message class
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using System;

[Serializable]
public class CustomRobotStatusMsg : Message
{
    public const string k_RosMessageName = "custom_msgs/RobotStatus";
    public override string RosMessageName => k_RosMessageName;

    public Header header;
    public string robot_name;
    public float battery_level;
    public bool is_charging;
    public string[] active_tasks;

    public CustomRobotStatusMsg()
    {
        header = new Header();
        robot_name = "";
        battery_level = 0.0f;
        is_charging = false;
        active_tasks = new string[0];
    }

    public CustomRobotStatusMsg(Header header, string robot_name, float battery_level, bool is_charging, string[] active_tasks)
    {
        this.header = header;
        this.robot_name = robot_name;
        this.battery_level = battery_level;
        this.is_charging = is_charging;
        this.active_tasks = active_tasks;
    }
}
```

### Synchronization Considerations

Unity runs at its own frame rate, which may differ from ROS2's update rate:

```csharp
using UnityEngine;

public class RateController : MonoBehaviour
{
    [SerializeField] private float targetRate = 50f; // Hz
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        updateInterval = 1.0f / targetRate;
        lastUpdateTime = Time.time;
    }

    bool ShouldUpdate()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            lastUpdateTime = Time.time;
            return true;
        }
        return false;
    }

    void Update()
    {
        if (ShouldUpdate())
        {
            // Update ROS communication at target rate
            UpdateRosComms();
        }
    }

    void UpdateRosComms()
    {
        // Handle ROS communication here
    }
}
```

## Physics and Sensors in Unity

### Physics Configuration

Unity's physics engine can be configured for robotics applications:

```csharp
using UnityEngine;

public class PhysicsConfig : MonoBehaviour
{
    void Start()
    {
        // Configure physics settings
        Physics.gravity = new Vector3(0, -9.81f, 0); // Earth gravity
        Physics.defaultSolverIterations = 10; // Increase for stability
        Physics.defaultSolverVelocityIterations = 2; // Increase for stability

        // Set fixed time step for consistent physics
        Time.fixedDeltaTime = 0.02f; // 50 Hz physics update
    }
}
```

### Implementing Sensors

Unity can simulate various types of sensors:

**Camera Sensor**:
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class CameraSensor : MonoBehaviour
{
    [SerializeField] private Camera sensorCamera;
    [SerializeField] private int imageWidth = 640;
    [SerializeField] private int imageHeight = 480;
    [SerializeField] private string topicName = "camera/image_raw";

    private RenderTexture renderTexture;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;

        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        sensorCamera.targetTexture = renderTexture;
    }

    void Update()
    {
        // Capture image and publish to ROS
        if (Time.frameCount % 30 == 0) // Publish every 30 frames
        {
            PublishCameraImage();
        }
    }

    void PublishCameraImage()
    {
        // Render texture to texture2D
        RenderTexture.active = renderTexture;
        Texture2D texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to ROS message (simplified)
        // In practice, you'd convert the image data to sensor_msgs/Image format
    }
}
```

**Lidar Sensor**:
```csharp
using UnityEngine;
using System.Collections.Generic;

public class LidarSensor : MonoBehaviour
{
    [SerializeField] private int rayCount = 360;
    [SerializeField] private float maxRange = 10.0f;
    [SerializeField] private float minRange = 0.1f;
    [SerializeField] private float angleMin = -Mathf.PI;
    [SerializeField] private float angleMax = Mathf.PI;

    private List<float> ranges;

    void Start()
    {
        ranges = new List<float>(new float[rayCount]);
    }

    void Update()
    {
        // Update lidar readings
        UpdateLidarReadings();
    }

    void UpdateLidarReadings()
    {
        float angleStep = (angleMax - angleMin) / rayCount;

        for (int i = 0; i < rayCount; i++)
        {
            float angle = angleMin + (i * angleStep);

            // Calculate ray direction
            Vector3 direction = new Vector3(
                Mathf.Cos(angle),
                0,
                Mathf.Sin(angle)
            );

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(direction), out hit, maxRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = maxRange;
            }
        }
    }

    public float[] GetRanges()
    {
        return ranges.ToArray();
    }
}
```

### Environment Simulation

Create realistic environments with proper lighting and materials:

```csharp
using UnityEngine;

public class EnvironmentSimulator : MonoBehaviour
{
    [SerializeField] private Material[] materials;
    [SerializeField] private GameObject[] environmentPrefabs;

    void Start()
    {
        // Set up environment
        SetupLighting();
        SetupMaterials();
        InstantiateEnvironment();
    }

    void SetupLighting()
    {
        // Configure main light (sun)
        Light mainLight = FindObjectOfType<Light>();
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.intensity = 1.0f;
            mainLight.color = Color.white;
            mainLight.transform.rotation = Quaternion.Euler(50, -30, 0);
        }

        // Enable shadows
        RenderSettings.shadowDistance = 100f;
        RenderSettings.shadowResolution = ShadowResolution.High;
    }

    void SetupMaterials()
    {
        // Apply realistic materials to environment objects
        foreach (Material mat in materials)
        {
            // Configure material properties for physics simulation
            mat.EnableKeyword("_NORMALMAP");
        }
    }

    void InstantiateEnvironment()
    {
        // Place environment objects
        foreach (GameObject prefab in environmentPrefabs)
        {
            // Instantiate at appropriate positions
            GameObject envObj = Instantiate(prefab);
            envObj.transform.SetParent(transform);
        }
    }
}
```

## Lab Exercise

### Objective
Create a Unity robotics simulation with ROS2 integration and sensor systems.

### Instructions
1. Set up a Unity project with ROS TCP Connector
2. Create a simple differential drive robot model
3. Implement ROS2 communication for velocity commands
4. Add a camera sensor and publish images to ROS
5. Implement a basic lidar sensor using raycasting
6. Test the simulation with ROS2 nodes

### Expected Outcome
You should have a Unity scene with a controllable robot that publishes sensor data to ROS2 topics.

## Summary

In this chapter, we explored Unity as a robotics simulation platform, covering setup, robot modeling, ROS2 integration, and sensor implementation. Unity provides high-quality graphics and flexible physics simulation, making it suitable for applications where visual realism is important. The integration with ROS2 through the TCP connector enables bidirectional communication between Unity and ROS2 systems.

## Quiz

Test your understanding of this chapter by taking the quiz. You can access the quiz at [Quiz Link] or through the navigation menu.