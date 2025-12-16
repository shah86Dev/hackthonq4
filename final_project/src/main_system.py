"""
Main System Orchestrator for Physical AI & Humanoid Robotics Textbook Project
This file implements the complete humanoid robot system integrating all components
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

from .system_architecture import (
    Component, RobotState, TaskResult, Command, TaskPriority,
    PerceptionSystem, AIDecisionMaker, ControlSystem,
    HumanInterface, HardwareInterface, TaskPlanner, MotionPlanner,
    BehaviorTreeSystem, IsaacSimInterface
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system state"""
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING_TASK = "executing_task"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    task_success_rate: float = 0.0
    uptime: float = 0.0
    error_count: int = 0


class HumanoidRobotSystem:
    """Main system orchestrator that integrates all components"""

    def __init__(self):
        self.system_state = SystemState.INITIALIZING
        self.start_time = time.time()

        # Initialize components
        self.perception_system = PerceptionSystem()
        self.ai_decision_maker = AIDecisionMaker()
        self.control_system = ControlSystem()
        self.human_interface = HumanInterface()
        self.hardware_interface = HardwareInterface()
        self.task_planner = TaskPlanner()
        self.motion_planner = MotionPlanner()
        self.behavior_system = BehaviorTreeSystem()
        self.simulation_interface = IsaacSimInterface()

        # System queues
        self.command_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.feedback_queue = queue.Queue()

        # Performance monitoring
        self.metrics = SystemMetrics()
        self.performance_monitor = PerformanceMonitor()

        # Task management
        self.active_tasks = {}
        self.completed_tasks = []

        # Health monitoring
        self.component_health = {}
        self.system_health = "unknown"

    async def initialize(self):
        """Initialize the complete humanoid robot system"""
        logger.info("Initializing Humanoid Robot System")

        # Initialize all components in parallel
        initialization_tasks = [
            self.perception_system.initialize(),
            self.ai_decision_maker.initialize(),
            self.control_system.initialize(),
            self.human_interface.initialize(),
            self.hardware_interface.initialize(),
            self.task_planner.initialize(),
            self.motion_planner.initialize(),
            self.behavior_system.initialize(),
            self.simulation_interface.initialize()
        ]

        # Run all initializations concurrently
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

        # Check for initialization errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component_names = [
                    'PerceptionSystem', 'AIDecisionMaker', 'ControlSystem',
                    'HumanInterface', 'HardwareInterface', 'TaskPlanner',
                    'MotionPlanner', 'BehaviorSystem', 'SimulationInterface'
                ]
                logger.error(f"Initialization failed for {component_names[i]}: {result}")
                raise result

        # Verify all components are healthy
        await self._verify_component_health()

        # Start system monitoring
        self._start_system_monitoring()

        self.system_state = SystemState.READY
        logger.info("Humanoid Robot System initialized successfully")

    async def _verify_component_health(self):
        """Verify all components are healthy after initialization"""
        components = {
            'perception_system': self.perception_system,
            'ai_decision_maker': self.ai_decision_maker,
            'control_system': self.control_system,
            'human_interface': self.human_interface,
            'hardware_interface': self.hardware_interface,
            'task_planner': self.task_planner,
            'motion_planner': self.motion_planner,
            'behavior_system': self.behavior_system,
            'simulation_interface': self.simulation_interface
        }

        for name, component in components.items():
            health = component.get_health_status()
            self.component_health[name] = health
            if health != "healthy":
                logger.warning(f"Component {name} is not healthy: {health}")

    def _start_system_monitoring(self):
        """Start system monitoring in background thread"""
        monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        monitor_thread.start()

    def _monitor_system(self):
        """Monitor system performance and health"""
        while self.system_state != SystemState.SHUTTING_DOWN:
            try:
                # Collect system metrics
                self.metrics.cpu_usage = self._get_cpu_usage()
                self.metrics.memory_usage = self._get_memory_usage()

                # Check component health
                self._check_component_health()

                # Update system health status
                self._update_system_health()

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                logger.error(f"Error in system monitoring: {str(e)}")
                time.sleep(1.0)

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        import psutil
        return psutil.cpu_percent()

    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        import psutil
        return psutil.virtual_memory().percent

    def _check_component_health(self):
        """Check health of all components"""
        components = {
            'perception_system': self.perception_system,
            'ai_decision_maker': self.ai_decision_maker,
            'control_system': self.control_system,
            'human_interface': self.human_interface,
            'hardware_interface': self.hardware_interface,
            'task_planner': self.task_planner,
            'motion_planner': self.motion_planner,
            'behavior_system': self.behavior_system,
            'simulation_interface': self.simulation_interface
        }

        for name, component in components.items():
            try:
                health = component.get_health_status()
                self.component_health[name] = health
            except Exception as e:
                logger.error(f"Error checking health of {name}: {str(e)}")
                self.component_health[name] = "error"

    def _update_system_health(self):
        """Update overall system health based on component health"""
        healthy_count = sum(1 for status in self.component_health.values() if status == "healthy")
        total_count = len(self.component_health)

        if healthy_count == total_count:
            self.system_health = "healthy"
        elif healthy_count >= total_count * 0.7:  # 70% healthy
            self.system_health = "degraded"
        else:
            self.system_health = "critical"

    async def process_command(self, command_text: str, source: str = "human") -> Dict[str, Any]:
        """Process a command from any source"""
        if self.system_state != SystemState.READY:
            return {
                'success': False,
                'message': f'System not ready, current state: {self.system_state.value}',
                'error': 'system_not_ready'
            }

        start_time = time.time()

        try:
            # Update system state
            self.system_state = SystemState.EXECUTING_TASK

            # If command is from human, process through human interface
            if source == "human":
                processed_command = await self.human_interface.process("text", command_text)
            else:
                processed_command = command_text

            # Get current perception data
            perception_data = await self.perception_system.process(None)  # Will get real data in implementation

            # Get current robot state
            robot_state = await self._get_robot_state()

            # Generate action plan using AI decision maker
            action_plan = await self.ai_decision_maker.process(
                processed_command,
                perception_data,
                robot_state
            )

            # Plan tasks sequence
            task_sequence = await self.task_planner.plan_task_sequence(
                action_plan,
                perception_data,
                robot_state
            )

            # Generate motion plan
            motion_plan = await self.motion_planner.generate_motion_plan(
                task_sequence,
                perception_data
            )

            # Execute through behavior system
            execution_result = await self.behavior_system.execute_behavior(
                task_sequence,
                motion_plan
            )

            # Execute on hardware/simulation
            hardware_result = await self.hardware_interface.process(action_plan)

            # Update metrics
            response_time = time.time() - start_time
            self.metrics.response_time = response_time

            # Update system state
            self.system_state = SystemState.READY

            return {
                'success': True,
                'message': 'Command executed successfully',
                'execution_result': execution_result,
                'hardware_result': hardware_result,
                'response_time': response_time,
                'task_id': action_plan.get('task_id', 'unknown')
            }

        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            self.metrics.error_count += 1

            # Update system state
            self.system_state = SystemState.READY

            return {
                'success': False,
                'message': f'Error processing command: {str(e)}',
                'error': str(e),
                'response_time': time.time() - start_time
            }

    async def _get_robot_state(self) -> RobotState:
        """Get current robot state"""
        # In a real implementation, this would get state from hardware/simulation
        # For now, return a mock state
        return RobotState(
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
            joint_states={'hip_l': 0.0, 'hip_r': 0.0, 'knee_l': 0.0, 'knee_r': 0.0},
            battery_level=0.85,
            is_connected=True,
            is_balanced=True
        )

    async def run_main_loop(self):
        """Main system execution loop"""
        logger.info("Starting main system loop")

        while self.system_state != SystemState.SHUTTING_DOWN:
            try:
                # Process any queued commands
                await self._process_command_queue()

                # Perform periodic system maintenance
                await self._perform_maintenance()

                # Check for system events
                await self._check_system_events()

                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(0.01)  # 10ms

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(0.1)  # Brief pause before continuing

    async def _process_command_queue(self):
        """Process commands from the command queue"""
        try:
            while not self.command_queue.empty():
                command_item = self.command_queue.get_nowait()
                await self.process_command(command_item['command'], command_item['source'])
        except queue.Empty:
            pass  # Queue is empty, continue

    async def _perform_maintenance(self):
        """Perform periodic system maintenance"""
        # This could include:
        # - Clearing old task data
        # - Updating system metrics
        # - Checking for component updates
        # - Managing system resources

        current_time = time.time()
        self.metrics.uptime = current_time - self.start_time

    async def _check_system_events(self):
        """Check for and process system events"""
        try:
            while not self.event_queue.empty():
                event = self.event_queue.get_nowait()
                await self._handle_system_event(event)
        except queue.Empty:
            pass

    async def _handle_system_event(self, event: Dict[str, Any]):
        """Handle a system event"""
        event_type = event.get('type', 'unknown')

        if event_type == 'emergency_stop':
            await self.emergency_stop()
        elif event_type == 'system_alert':
            await self.handle_system_alert(event)
        elif event_type == 'component_failure':
            await self.handle_component_failure(event)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    async def emergency_stop(self):
        """Perform emergency stop of all systems"""
        logger.warning("Emergency stop initiated!")

        self.system_state = SystemState.EMERGENCY_STOP

        # Stop all active tasks
        for task_id in list(self.active_tasks.keys()):
            await self._stop_task(task_id)

        # Stop all motion
        await self.control_system.stop_all_motion()

        # Pause all components
        await self._pause_all_components()

    async def handle_system_alert(self, alert: Dict[str, Any]):
        """Handle system alert"""
        alert_level = alert.get('level', 'info')
        alert_message = alert.get('message', 'Unknown alert')

        if alert_level == 'error':
            logger.error(f"System alert: {alert_message}")
        elif alert_level == 'warning':
            logger.warning(f"System alert: {alert_message}")
        else:
            logger.info(f"System alert: {alert_message}")

    async def handle_component_failure(self, failure_event: Dict[str, Any]):
        """Handle component failure event"""
        component_name = failure_event.get('component', 'unknown')
        error_message = failure_event.get('error', 'Unknown error')

        logger.error(f"Component failure: {component_name} - {error_message}")

        # Attempt to restart the failed component
        await self._attempt_component_recovery(component_name)

    async def _attempt_component_recovery(self, component_name: str):
        """Attempt to recover from component failure"""
        logger.info(f"Attempting to recover component: {component_name}")

        # This would implement component-specific recovery logic
        # For now, just log the attempt
        recovery_success = True  # Placeholder

        if recovery_success:
            logger.info(f"Successfully recovered component: {component_name}")
        else:
            logger.error(f"Failed to recover component: {component_name}")

    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Initiating system shutdown...")

        self.system_state = SystemState.SHUTTING_DOWN

        # Stop all active tasks
        for task_id in list(self.active_tasks.keys()):
            await self._stop_task(task_id)

        # Shutdown all components
        shutdown_tasks = [
            self.perception_system.shutdown(),
            self.ai_decision_maker.shutdown(),
            self.control_system.shutdown(),
            self.human_interface.shutdown(),
            self.hardware_interface.shutdown(),
            self.simulation_interface.shutdown()
        ]

        try:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during component shutdown: {str(e)}")

        logger.info("Humanoid Robot System shutdown complete")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'system_state': self.system_state.value,
            'system_health': self.system_health,
            'component_health': self.component_health,
            'metrics': {
                'cpu_usage': self.metrics.cpu_usage,
                'memory_usage': self.metrics.memory_usage,
                'response_time': self.metrics.response_time,
                'uptime': self.metrics.uptime,
                'error_count': self.metrics.error_count
            },
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'timestamp': time.time()
        }

    async def add_command(self, command: str, source: str = "human"):
        """Add command to processing queue"""
        self.command_queue.put({
            'command': command,
            'source': source,
            'timestamp': time.time()
        })

    async def _stop_task(self, task_id: str):
        """Stop a specific task"""
        if task_id in self.active_tasks:
            # Implement task stopping logic
            del self.active_tasks[task_id]
            logger.info(f"Stopped task: {task_id}")


class PerformanceMonitor:
    """Monitor system performance metrics"""

    def __init__(self):
        self.metrics_history = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': [],
            'throughput': [],
            'error_rates': []
        }
        self.window_size = 100  # Keep last 100 measurements

    def record_cpu_usage(self, usage: float):
        """Record CPU usage metric"""
        self.metrics_history['cpu_usage'].append(usage)
        if len(self.metrics_history['cpu_usage']) > self.window_size:
            self.metrics_history['cpu_usage'].pop(0)

    def record_memory_usage(self, usage: float):
        """Record memory usage metric"""
        self.metrics_history['memory_usage'].append(usage)
        if len(self.metrics_history['memory_usage']) > self.window_size:
            self.metrics_history['memory_usage'].pop(0)

    def record_response_time(self, time_ms: float):
        """Record response time metric"""
        self.metrics_history['response_times'].append(time_ms)
        if len(self.metrics_history['response_times']) > self.window_size:
            self.metrics_history['response_times'].pop(0)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        metrics = {}

        for metric_name, values in self.metrics_history.items():
            if values:
                metrics[f'{metric_name}_avg'] = sum(values) / len(values)
                metrics[f'{metric_name}_min'] = min(values)
                metrics[f'{metric_name}_max'] = max(values)
                metrics[f'{metric_name}_current'] = values[-1] if values else 0.0
            else:
                metrics[f'{metric_name}_avg'] = 0.0
                metrics[f'{metric_name}_min'] = 0.0
                metrics[f'{metric_name}_max'] = 0.0
                metrics[f'{metric_name}_current'] = 0.0

        return metrics

    def get_performance_trend(self, metric_name: str) -> str:
        """Get trend for a specific metric"""
        if metric_name not in self.metrics_history:
            return "unknown"

        values = self.metrics_history[metric_name]
        if len(values) < 2:
            return "stable"

        recent_avg = sum(values[-5:]) / min(5, len(values))
        earlier_avg = sum(values[:5]) / min(5, len(values))

        if recent_avg > earlier_avg * 1.1:  # 10% increase
            return "increasing"
        elif recent_avg < earlier_avg * 0.9:  # 10% decrease
            return "decreasing"
        else:
            return "stable"

    def check_performance_thresholds(self) -> List[str]:
        """Check if performance is within acceptable thresholds"""
        alerts = []

        # CPU usage threshold
        if self.metrics_history['cpu_usage']:
            avg_cpu = sum(self.metrics_history['cpu_usage']) / len(self.metrics_history['cpu_usage'])
            if avg_cpu > 80:
                alerts.append(f"High CPU usage: {avg_cpu:.1f}%")

        # Memory usage threshold
        if self.metrics_history['memory_usage']:
            avg_memory = sum(self.metrics_history['memory_usage']) / len(self.metrics_history['memory_usage'])
            if avg_memory > 85:
                alerts.append(f"High memory usage: {avg_memory:.1f}%")

        # Response time threshold
        if self.metrics_history['response_times']:
            avg_response = sum(self.metrics_history['response_times']) / len(self.metrics_history['response_times'])
            if avg_response > 1000:  # More than 1 second
                alerts.append(f"Slow response times: {avg_response:.1f}ms")

        return alerts


class TaskManager:
    """Manage execution of complex tasks"""

    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_queue = asyncio.Queue()

    async def create_task(self, task_spec: Dict[str, Any]) -> str:
        """Create a new task with the given specification"""
        task_id = f"task_{int(time.time())}_{len(self.active_tasks)}"

        task = {
            'id': task_id,
            'spec': task_spec,
            'status': 'created',
            'created_at': time.time(),
            'started_at': None,
            'completed_at': None,
            'progress': 0.0,
            'result': None,
            'error': None
        }

        self.active_tasks[task_id] = task
        await self.task_queue.put(task_id)

        return task_id

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a specific task"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.active_tasks[task_id]
        task['status'] = 'executing'
        task['started_at'] = time.time()

        try:
            # Execute the task based on its specification
            result = await self._execute_task_logic(task['spec'])

            task['status'] = 'completed'
            task['completed_at'] = time.time()
            task['result'] = result
            task['progress'] = 1.0

            # Move to completed tasks
            self.completed_tasks[task_id] = self.active_tasks.pop(task_id)

            return {
                'success': True,
                'task_id': task_id,
                'result': result,
                'execution_time': task['completed_at'] - task['started_at']
            }

        except Exception as e:
            task['status'] = 'failed'
            task['completed_at'] = time.time()
            task['error'] = str(e)
            task['progress'] = 0.0

            # Move to completed tasks
            self.completed_tasks[task_id] = self.active_tasks.pop(task_id)

            return {
                'success': False,
                'task_id': task_id,
                'error': str(e),
                'execution_time': task['completed_at'] - task['started_at'] if task['started_at'] else 0
            }

    async def _execute_task_logic(self, task_spec: Dict[str, Any]) -> Any:
        """Execute the actual task logic based on specification"""
        task_type = task_spec.get('type', 'unknown')

        if task_type == 'navigation':
            return await self._execute_navigation_task(task_spec)
        elif task_type == 'manipulation':
            return await self._execute_manipulation_task(task_spec)
        elif task_type == 'perception':
            return await self._execute_perception_task(task_spec)
        elif task_type == 'communication':
            return await self._execute_communication_task(task_spec)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _execute_navigation_task(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute navigation task"""
        target_pose = task_spec.get('target_pose', [0.0, 0.0, 0.0])
        max_velocity = task_spec.get('max_velocity', 0.5)

        # In a real implementation, this would interface with navigation stack
        # For now, simulate navigation
        await asyncio.sleep(2.0)  # Simulate navigation time

        return {
            'success': True,
            'final_pose': target_pose,
            'path_executed': True,
            'obstacles_avoided': 0,
            'navigation_time': 2.0
        }

    async def _execute_manipulation_task(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manipulation task"""
        target_object = task_spec.get('target_object', 'unknown')
        action = task_spec.get('action', 'grasp')

        # In a real implementation, this would interface with manipulation stack
        # For now, simulate manipulation
        await asyncio.sleep(3.0)  # Simulate manipulation time

        return {
            'success': True,
            'action_performed': action,
            'object_manipulated': target_object,
            'manipulation_time': 3.0
        }

    async def _execute_perception_task(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute perception task"""
        perception_type = task_spec.get('perception_type', 'object_detection')

        # In a real implementation, this would interface with perception system
        # For now, simulate perception
        await asyncio.sleep(1.0)  # Simulate perception time

        if perception_type == 'object_detection':
            return {
                'success': True,
                'objects_detected': ['cube', 'sphere'],
                'detection_time': 1.0
            }
        else:
            return {
                'success': True,
                'perception_result': 'completed',
                'perception_time': 1.0
            }

    async def _execute_communication_task(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute communication task"""
        message = task_spec.get('message', 'Hello')
        recipient = task_spec.get('recipient', 'user')

        # In a real implementation, this would interface with communication system
        # For now, simulate communication
        await asyncio.sleep(0.5)  # Simulate communication time

        return {
            'success': True,
            'message_sent': message,
            'recipient': recipient,
            'communication_time': 0.5
        }

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task['status'],
                'progress': task['progress'],
                'created_at': task['created_at'],
                'started_at': task['started_at'],
                'estimated_completion': self._estimate_completion_time(task)
            }
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task['status'],
                'result': task['result'],
                'error': task['error'],
                'created_at': task['created_at'],
                'started_at': task['started_at'],
                'completed_at': task['completed_at'],
                'execution_time': task['completed_at'] - task['started_at'] if task['started_at'] else 0
            }
        else:
            raise ValueError(f"Task {task_id} not found")

    def _estimate_completion_time(self, task: Dict[str, Any]) -> float:
        """Estimate time to completion for active task"""
        if task['started_at'] is None:
            return -1.0  # Not started yet

        elapsed = time.time() - task['started_at']
        if task['progress'] > 0:
            estimated_total = elapsed / task['progress']
            return estimated_total - elapsed
        else:
            return 60.0  # Default estimate of 60 seconds


# Main execution function
async def main():
    """Main function to run the complete system"""
    system = HumanoidRobotSystem()

    try:
        # Initialize the system
        await system.initialize()

        # Example: Process a command
        result = await system.process_command("Move forward by 1 meter", source="human")
        print(f"Command result: {result}")

        # Example: Add multiple commands to queue
        await system.add_command("Turn left 90 degrees", source="human")
        await system.add_command("Pick up the red cube", source="human")

        # Run main loop for a short time
        start_time = time.time()
        while time.time() - start_time < 10:  # Run for 10 seconds
            await asyncio.sleep(0.1)

        # Get system status
        status = system.get_system_status()
        print(f"System status: {status}")

    except KeyboardInterrupt:
        print("\nShutting down system...")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())