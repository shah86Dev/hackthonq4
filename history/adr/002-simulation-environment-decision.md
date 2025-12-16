# ADR-002: Simulation Environment for Robotics Education

## Status
Accepted

## Date
2025-12-16

## Context
The Physical AI & Humanoid Robotics textbook requires a simulation environment that can:
- Support ROS2 integration for lab exercises
- Provide realistic physics for humanoid robotics
- Be accessible to students for educational purposes
- Integrate with Isaac (NVIDIA's simulation platform)
- Support both Gazebo and Isaac Sim based on different use cases

## Decision
We will use a dual simulation environment approach with:
- **Primary**: Isaac Sim for advanced humanoid robotics simulation
- **Secondary**: Gazebo for ROS2 compatibility and broader robotics concepts
- Integration with ROS2 workspace for lab exercises
- Containerized environments for consistent student experience

## Alternatives Considered
1. **Webots**: Rejected due to limited ROS2 integration capabilities
2. **PyBullet**: Rejected due to less realistic physics for humanoid robotics applications
3. **Custom Unity simulation**: Rejected due to increased development time and lack of robotics-specific features
4. **Only Gazebo**: Rejected as Isaac Sim provides better humanoid simulation capabilities
5. **Only Isaac Sim**: Rejected as Gazebo offers broader ROS2 compatibility

## Consequences
### Positive
- Isaac Sim provides state-of-the-art humanoid robotics simulation
- Gazebo ensures broad ROS2 compatibility and community support
- Students gain experience with industry-standard simulation tools
- Containerized environments ensure consistent lab experience
- Supports both basic and advanced robotics concepts

### Negative
- Increased complexity in maintaining two simulation environments
- Higher hardware requirements for students to run Isaac Sim
- Need for more complex documentation and setup guides
- Potential licensing considerations for Isaac Sim in educational setting

## References
- specs/1-physical-ai-textbook/plan.md
- specs/1-physical-ai-textbook/research.md
- specs/1-physical-ai-textbook/data-model.md