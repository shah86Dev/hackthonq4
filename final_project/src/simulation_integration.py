"""
Simulation Integration for Physical AI & Humanoid Robotics
Handles integration with Isaac Sim, Gazebo, and other simulation environments
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math
import time

logger = logging.getLogger(__name__)


class SimulatorType(Enum):
    """Supported simulation platforms"""
    ISAAC_SIM = "isaac_sim"
    GAZEBO = "gazebo"
    UNITY = "unity"
    CUSTOM = "custom"


class SimulationState(Enum):
    """Simulation execution states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    RESET = "reset"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration for simulation environment"""
    simulator_type: SimulatorType
    world_file: str
    robot_model_path: str
    physics_engine: str = "PhysX"
    time_step: float = 0.001
    gravity: List[float] = None  # [x, y, z] - defaults to Earth gravity
    rendering_enabled: bool = True
    max_steps: int = 1000000
    real_time_factor: float = 1.0

    def __post_init__(self):
        if self.gravity is None:
            self.gravity = [0.0, 0.0, -9.81]  # Earth gravity


@dataclass
class RobotState:
    """State of robot in simulation"""
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]
    joint_efforts: Dict[str, float]
    base_pose: List[float]  # [x, y, z, qx, qy, qz, qw]
    base_twist: List[float]  # [vx, vy, vz, wx, wy, wz]
    imu_data: Dict[str, float]
    force_torque_data: Dict[str, List[float]]
    contact_points: List[Dict[str, Any]]
    simulation_time: float


@dataclass
class SimulationResponse:
    """Response from simulation operations"""
    success: bool
    message: str
    execution_time: float
    robot_state: Optional[RobotState] = None
    error_details: Optional[Dict[str, Any]] = None


class IsaacSimInterface:
    """Interface to NVIDIA Isaac Sim"""

    def __init__(self):
        self.is_connected = False
        self.simulation_state = SimulationState.STOPPED
        self.robot_prim_path = None
        self.simulation_context = None
        self.world = None
        self.is_initialized = False

    async def initialize(self, config: SimulationConfig):
        """Initialize Isaac Sim interface"""
        logger.info(f"Initializing Isaac Sim interface with config: {config.world_file}")

        try:
            # In a real implementation, this would connect to Isaac Sim
            # For now, simulate the connection process
            await asyncio.sleep(0.1)  # Simulate connection time

            self.simulation_config = config
            self.robot_prim_path = config.robot_model_path
            self.is_connected = True
            self.is_initialized = True

            logger.info("Isaac Sim interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Isaac Sim interface: {str(e)}")
            raise

    async def load_robot(self, robot_usd_path: str) -> SimulationResponse:
        """Load robot model into Isaac Sim"""
        if not self.is_initialized:
            raise RuntimeError("Isaac Sim interface not initialized")

        start_time = time.time()

        try:
            # In real implementation, this would load the robot USD into Isaac Sim
            # For now, simulate the process
            await asyncio.sleep(0.05)  # Simulate loading time

            # Create mock robot state
            mock_state = RobotState(
                joint_positions={'hip_l': 0.0, 'hip_r': 0.0, 'knee_l': 0.0, 'knee_r': 0.0},
                joint_velocities={'hip_l': 0.0, 'hip_r': 0.0, 'knee_l': 0.0, 'knee_r': 0.0},
                joint_efforts={'hip_l': 0.0, 'hip_r': 0.0, 'knee_l': 0.0, 'knee_r': 0.0},
                base_pose=[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                base_twist=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                imu_data={'orientation_x': 0.0, 'orientation_y': 0.0, 'orientation_z': 0.0, 'orientation_w': 1.0,
                         'angular_velocity_x': 0.0, 'angular_velocity_y': 0.0, 'angular_velocity_z': 0.0,
                         'linear_acceleration_x': 0.0, 'linear_acceleration_y': 0.0, 'linear_acceleration_z': 9.81},
                force_torque_data={'left_foot': [0.0, 0.0, -400.0, 0.0, 0.0, 0.0],
                                  'right_foot': [0.0, 0.0, -400.0, 0.0, 0.0, 0.0]},
                contact_points=[],
                simulation_time=0.0
            )

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Robot loaded from {robot_usd_path}",
                execution_time=execution_time,
                robot_state=mock_state
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to load robot: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def set_joint_positions(self, joint_commands: Dict[str, float]) -> SimulationResponse:
        """Set joint positions in simulation"""
        if not self.is_connected:
            raise RuntimeError("Isaac Sim interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would set joint positions in Isaac Sim
            # For now, simulate the process
            await asyncio.sleep(0.005)  # Simulate command execution time

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Joint positions set: {list(joint_commands.keys())}",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to set joint positions: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def get_robot_state(self) -> SimulationResponse:
        """Get current robot state from simulation"""
        if not self.is_connected:
            raise RuntimeError("Isaac Sim interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would get robot state from Isaac Sim
            # For now, return mock state with realistic values
            mock_state = RobotState(
                joint_positions={'hip_l': 0.1, 'hip_r': -0.1, 'knee_l': 0.5, 'knee_r': 0.5},
                joint_velocities={'hip_l': 0.01, 'hip_r': -0.01, 'knee_l': 0.02, 'knee_r': 0.02},
                joint_efforts={'hip_l': 10.5, 'hip_r': 11.2, 'knee_l': 15.8, 'knee_r': 16.1},
                base_pose=[0.01, 0.0, 0.79, 0.0, 0.0, 0.01, 0.9999],
                base_twist=[0.05, 0.0, 0.01, 0.0, 0.01, 0.0],
                imu_data={'orientation_x': 0.001, 'orientation_y': 0.002, 'orientation_z': 0.01, 'orientation_w': 0.9999,
                         'angular_velocity_x': 0.01, 'angular_velocity_y': 0.02, 'angular_velocity_z': 0.05,
                         'linear_acceleration_x': 0.1, 'linear_acceleration_y': 0.05, 'linear_acceleration_z': 9.75},
                force_torque_data={'left_foot': [5.2, 3.1, -395.8, 0.1, 0.2, 0.05],
                                  'right_foot': [-2.1, -1.8, -398.2, -0.05, 0.1, 0.02]},
                contact_points=[
                    {'body1': 'left_foot', 'body2': 'ground', 'position': [0.0, 0.1, 0.0]},
                    {'body1': 'right_foot', 'body2': 'ground', 'position': [0.0, -0.1, 0.0]}
                ],
                simulation_time=time.time()
            )

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message="Robot state retrieved successfully",
                execution_time=execution_time,
                robot_state=mock_state
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to get robot state: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def step_simulation(self, num_steps: int = 1) -> SimulationResponse:
        """Step the simulation forward"""
        if not self.is_connected:
            raise RuntimeError("Isaac Sim interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would step the Isaac Sim physics
            # For now, simulate the process
            await asyncio.sleep(0.001 * num_steps)  # Simulate physics step time

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Stepped simulation by {num_steps} steps",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to step simulation: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def reset_simulation(self) -> SimulationResponse:
        """Reset the simulation to initial state"""
        if not self.is_connected:
            raise RuntimeError("Isaac Sim interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would reset Isaac Sim
            # For now, simulate the process
            await asyncio.sleep(0.02)  # Simulate reset time

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message="Simulation reset successfully",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to reset simulation: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def get_sensor_data(self, sensor_names: List[str]) -> SimulationResponse:
        """Get data from specified sensors in simulation"""
        if not self.is_connected:
            raise RuntimeError("Isaac Sim interface not connected")

        start_time = time.time()

        try:
            sensor_data = {}

            for sensor_name in sensor_names:
                if 'camera' in sensor_name.lower():
                    # Mock camera data
                    sensor_data[sensor_name] = {
                        'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                        'depth': np.random.random((480, 640)).astype(np.float32) * 10.0,
                        'timestamp': time.time()
                    }
                elif 'lidar' in sensor_name.lower():
                    # Mock LiDAR data
                    sensor_data[sensor_name] = {
                        'ranges': np.random.random(360).astype(np.float32) * 10.0,
                        'intensities': np.random.random(360).astype(np.float32),
                        'timestamp': time.time()
                    }
                elif 'imu' in sensor_name.lower():
                    # Mock IMU data
                    sensor_data[sensor_name] = {
                        'orientation': [0.0, 0.0, 0.0, 1.0],
                        'angular_velocity': [0.01, 0.02, 0.05],
                        'linear_acceleration': [0.1, 0.05, 9.75],
                        'timestamp': time.time()
                    }
                else:
                    # Generic sensor data
                    sensor_data[sensor_name] = {
                        'values': np.random.random(3).astype(np.float32),
                        'timestamp': time.time()
                    }

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Retrieved data from {len(sensor_names)} sensors",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to get sensor data: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def apply_external_force(self, link_name: str, force: List[float],
                                 position: List[float]) -> SimulationResponse:
        """Apply external force to a link in the simulation"""
        if not self.is_connected:
            raise RuntimeError("Isaac Sim interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would apply force in Isaac Sim
            # For now, simulate the process
            await asyncio.sleep(0.001)

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Applied force {force} to {link_name} at {position}",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to apply external force: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def shutdown(self):
        """Shutdown Isaac Sim interface"""
        logger.info("Shutting down Isaac Sim interface")
        self.is_connected = False
        self.is_initialized = False
        logger.info("Isaac Sim interface shutdown complete")


class GazeboInterface:
    """Interface to Gazebo simulation"""

    def __init__(self):
        self.is_connected = False
        self.simulation_state = SimulationState.STOPPED
        self.model_names = []
        self.is_initialized = False

    async def initialize(self, config: SimulationConfig):
        """Initialize Gazebo interface"""
        logger.info(f"Initializing Gazebo interface with config: {config.world_file}")

        try:
            # In a real implementation, this would connect to Gazebo
            # For now, simulate the connection process
            await asyncio.sleep(0.1)  # Simulate connection time

            self.simulation_config = config
            self.is_connected = True
            self.is_initialized = True

            logger.info("Gazebo interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gazebo interface: {str(e)}")
            raise

    async def spawn_robot(self, robot_model_path: str, pose: List[float]) -> SimulationResponse:
        """Spawn robot model in Gazebo"""
        if not self.is_initialized:
            raise RuntimeError("Gazebo interface not initialized")

        start_time = time.time()

        try:
            # In real implementation, this would spawn the robot in Gazebo
            # For now, simulate the process
            await asyncio.sleep(0.05)  # Simulate spawning time

            # Add model name to tracking list
            model_name = f"robot_{int(time.time())}"
            self.model_names.append(model_name)

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Robot spawned as {model_name} at pose {pose}",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to spawn robot: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def set_model_state(self, model_name: str, pose: List[float],
                            twist: List[float]) -> SimulationResponse:
        """Set model state (pose and velocity) in Gazebo"""
        if not self.is_connected:
            raise RuntimeError("Gazebo interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would set model state in Gazebo
            # For now, simulate the process
            await asyncio.sleep(0.002)  # Simulate command execution time

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Set state for model {model_name}",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to set model state: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def get_model_state(self, model_name: str) -> SimulationResponse:
        """Get model state from Gazebo"""
        if not self.is_connected:
            raise RuntimeError("Gazebo interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would get model state from Gazebo
            # For now, return mock state
            mock_state = RobotState(
                joint_positions={'hip_l': 0.05, 'hip_r': -0.05, 'knee_l': 0.3, 'knee_r': 0.3},
                joint_velocities={'hip_l': 0.0, 'hip_r': 0.0, 'knee_l': 0.0, 'knee_r': 0.0},
                joint_efforts={'hip_l': 5.0, 'hip_r': 5.0, 'knee_l': 8.0, 'knee_r': 8.0},
                base_pose=[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                base_twist=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                imu_data={'orientation_x': 0.0, 'orientation_y': 0.0, 'orientation_z': 0.0, 'orientation_w': 1.0,
                         'angular_velocity_x': 0.0, 'angular_velocity_y': 0.0, 'angular_velocity_z': 0.0,
                         'linear_acceleration_x': 0.0, 'linear_acceleration_y': 0.0, 'linear_acceleration_z': 9.81},
                force_torque_data={'left_foot': [0.0, 0.0, -200.0, 0.0, 0.0, 0.0],
                                  'right_foot': [0.0, 0.0, -200.0, 0.0, 0.0, 0.0]},
                contact_points=[],
                simulation_time=time.time()
            )

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Retrieved state for model {model_name}",
                execution_time=execution_time,
                robot_state=mock_state
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to get model state: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def apply_joint_effort(self, model_name: str, joint_name: str,
                               effort: float, duration: float) -> SimulationResponse:
        """Apply effort to a joint in Gazebo"""
        if not self.is_connected:
            raise RuntimeError("Gazebo interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would apply joint effort in Gazebo
            # For now, simulate the process
            await asyncio.sleep(duration)  # Simulate application duration

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Applied {effort}N effort to {joint_name} on {model_name} for {duration}s",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to apply joint effort: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def get_physics_properties(self) -> SimulationResponse:
        """Get current physics properties from Gazebo"""
        if not self.is_connected:
            raise RuntimeError("Gazebo interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would get physics properties from Gazebo
            physics_props = {
                'gravity': self.simulation_config.gravity,
                'time_step': self.simulation_config.time_step,
                'real_time_factor': self.simulation_config.real_time_factor,
                'max_step_size': 0.001,
                'sor_iterations': 50,
                'contact_surface_layer': 0.001
            }

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message="Retrieved physics properties",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to get physics properties: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def set_physics_properties(self, gravity: List[float] = None,
                                   time_step: float = None,
                                   real_time_factor: float = None) -> SimulationResponse:
        """Set physics properties in Gazebo"""
        if not self.is_connected:
            raise RuntimeError("Gazebo interface not connected")

        start_time = time.time()

        try:
            # In real implementation, this would set physics properties in Gazebo
            # For now, simulate the process
            await asyncio.sleep(0.01)  # Simulate configuration time

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message="Physics properties updated successfully",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to set physics properties: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def shutdown(self):
        """Shutdown Gazebo interface"""
        logger.info("Shutting down Gazebo interface")
        self.is_connected = False
        self.is_initialized = False
        logger.info("Gazebo interface shutdown complete")


class SimulationManager:
    """Manages multiple simulation environments and provides unified interface"""

    def __init__(self):
        self.isaac_sim_interface = IsaacSimInterface()
        self.gazebo_interface = GazeboInterface()
        self.active_simulator = None
        self.is_running = False
        self.health_status = "unknown"

    async def initialize(self, config: SimulationConfig):
        """Initialize simulation manager with specified configuration"""
        logger.info(f"Initializing Simulation Manager for {config.simulator_type.value}")

        # Initialize the appropriate simulator interface
        if config.simulator_type == SimulatorType.ISAAC_SIM:
            await self.isaac_sim_interface.initialize(config)
            self.active_simulator = self.isaac_sim_interface
        elif config.simulator_type == SimulatorType.GAZEBO:
            await self.gazebo_interface.initialize(config)
            self.active_simulator = self.gazebo_interface
        else:
            raise ValueError(f"Unsupported simulator type: {config.simulator_type}")

        self.is_running = True
        self.health_status = "healthy"
        logger.info(f"Simulation Manager initialized for {config.simulator_type.value}")

    async def load_robot_model(self, robot_path: str, initial_pose: List[float] = None) -> SimulationResponse:
        """Load robot model into active simulation"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        if initial_pose is None:
            initial_pose = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]  # Default standing pose

        if self.active_simulator == self.isaac_sim_interface:
            return await self.isaac_sim_interface.load_robot(robot_path)
        elif self.active_simulator == self.gazebo_interface:
            return await self.gazebo_interface.spawn_robot(robot_path, initial_pose)
        else:
            raise RuntimeError("No active simulator")

    async def get_robot_state(self) -> SimulationResponse:
        """Get robot state from active simulation"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        if self.active_simulator == self.isaac_sim_interface:
            return await self.isaac_sim_interface.get_robot_state()
        elif self.active_simulator == self.gazebo_interface:
            # For Gazebo, we need to know the model name
            # This would be tracked by the manager in a real implementation
            return await self.gazebo_interface.get_model_state("robot_0")  # Mock model name
        else:
            raise RuntimeError("No active simulator")

    async def set_joint_commands(self, joint_commands: Dict[str, float]) -> SimulationResponse:
        """Send joint commands to robot in active simulation"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        if self.active_simulator == self.isaac_sim_interface:
            return await self.isaac_sim_interface.set_joint_positions(joint_commands)
        elif self.active_simulator == self.gazebo_interface:
            # In Gazebo, this would involve sending joint commands through ROS2 controllers
            # For now, return a mock response
            await asyncio.sleep(0.005)  # Simulate command processing
            return SimulationResponse(
                success=True,
                message="Joint commands sent to Gazebo simulation",
                execution_time=0.005
            )
        else:
            raise RuntimeError("No active simulator")

    async def step_simulation(self, num_steps: int = 1) -> SimulationResponse:
        """Step the active simulation"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        if self.active_simulator == self.isaac_sim_interface:
            return await self.isaac_sim_interface.step_simulation(num_steps)
        elif self.active_simulator == self.gazebo_interface:
            # In Gazebo, this would be handled differently
            # For now, simulate the stepping
            await asyncio.sleep(0.001 * num_steps)  # Simulate physics time
            return SimulationResponse(
                success=True,
                message=f"Stepped Gazebo simulation by {num_steps} steps",
                execution_time=0.001 * num_steps
            )
        else:
            raise RuntimeError("No active simulator")

    async def reset_simulation(self) -> SimulationResponse:
        """Reset the active simulation"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        if self.active_simulator == self.isaac_sim_interface:
            return await self.isaac_sim_interface.reset_simulation()
        elif self.active_simulator == self.gazebo_interface:
            # In Gazebo, reset would be different
            await asyncio.sleep(0.02)  # Simulate reset time
            return SimulationResponse(
                success=True,
                message="Gazebo simulation reset",
                execution_time=0.02
            )
        else:
            raise RuntimeError("No active simulator")

    async def get_sensor_data(self, sensor_names: List[str]) -> SimulationResponse:
        """Get sensor data from active simulation"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        if self.active_simulator == self.isaac_sim_interface:
            return await self.isaac_sim_interface.get_sensor_data(sensor_names)
        elif self.active_simulator == self.gazebo_interface:
            # For Gazebo, this would involve getting sensor data through ROS2 topics
            # For now, return mock data
            sensor_data = {}
            for name in sensor_names:
                if 'camera' in name.lower():
                    sensor_data[name] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                elif 'lidar' in name.lower():
                    sensor_data[name] = np.random.random(360).astype(np.float32) * 10.0
                else:
                    sensor_data[name] = np.random.random(3).astype(np.float32)

            return SimulationResponse(
                success=True,
                message=f"Retrieved data from {len(sensor_names)} sensors in Gazebo",
                execution_time=0.002
            )
        else:
            raise RuntimeError("No active simulator")

    async def apply_external_disturbance(self, force: List[float], position: List[float],
                                       link_name: str = "base_link") -> SimulationResponse:
        """Apply external disturbance to robot in simulation"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        if self.active_simulator == self.isaac_sim_interface:
            return await self.isaac_sim_interface.apply_external_force(link_name, force, position)
        elif self.active_simulator == self.gazebo_interface:
            # In Gazebo, apply external force differently
            await asyncio.sleep(0.001)  # Simulate force application
            return SimulationResponse(
                success=True,
                message=f"Applied external force to {link_name} in Gazebo",
                execution_time=0.001
            )
        else:
            raise RuntimeError("No active simulator")

    def get_active_simulator_type(self) -> SimulatorType:
        """Get the currently active simulator type"""
        if self.active_simulator == self.isaac_sim_interface:
            return SimulatorType.ISAAC_SIM
        elif self.active_simulator == self.gazebo_interface:
            return SimulatorType.GAZEBO
        else:
            return None

    def get_health_status(self) -> str:
        """Get current health status of simulation manager"""
        return self.health_status

    async def switch_simulator(self, new_simulator_type: SimulatorType,
                             config: SimulationConfig) -> SimulationResponse:
        """Switch to a different simulator"""
        if not self.is_running:
            raise RuntimeError("Simulation Manager not running")

        start_time = time.time()

        try:
            # Shutdown current simulator
            if self.active_simulator:
                await self.active_simulator.shutdown()

            # Initialize new simulator
            if new_simulator_type == SimulatorType.ISAAC_SIM:
                await self.isaac_sim_interface.initialize(config)
                self.active_simulator = self.isaac_sim_interface
            elif new_simulator_type == SimulatorType.GAZEBO:
                await self.gazebo_interface.initialize(config)
                self.active_simulator = self.gazebo_interface
            else:
                raise ValueError(f"Unsupported simulator type: {new_simulator_type}")

            execution_time = time.time() - start_time
            return SimulationResponse(
                success=True,
                message=f"Switched to {new_simulator_type.value} simulator",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SimulationResponse(
                success=False,
                message=f"Failed to switch simulator: {str(e)}",
                execution_time=execution_time,
                error_details={'exception': str(e)}
            )

    async def shutdown(self):
        """Shutdown simulation manager and all interfaces"""
        logger.info("Shutting down Simulation Manager")

        if self.active_simulator:
            await self.active_simulator.shutdown()

        self.is_running = False
        self.health_status = "shutdown"
        logger.info("Simulation Manager shutdown complete")