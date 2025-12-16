---
id: 1-sensorimotor-coordination
title: Sensorimotor Coordination
sidebar_label: Sensorimotor Coordination
---

# Sensorimotor Coordination

Perception and Action in Physical Systems

## Learning Outcomes

- Understand sensorimotor coordination principles

- Implement basic sensorimotor loops

- Analyze feedback control systems

## Prerequisites

- chapter1


## Perception-Action Loops

Sensorimotor coordination is fundamental to Physical AI. It involves the continuous cycle of perception, decision-making, and action. In humanoid robotics, this loop must operate in real-time to enable stable locomotion, manipulation, and interaction with the environment.


### Sensorimotor Loop Architecture

Diagram showing perception, decision, and action components in a feedback loop


### Isaac Sim: Basic Sensor Integration

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.range_sensor import _range_sensor

# Initialize Isaac Sim world
world = World(stage_units_in_meters=1.0)

# Create a robot with sensors
robot = world.scene.add(
    Robot(prim_path="/World/Robot", name="my_robot")
)

# Access sensor data
range_sensor = _range_sensor.acquire_range_sensor_interface()
point_cloud = range_sensor.get_point_cloud_data(
    prim_path="/World/Robot/lidar"
)
```

