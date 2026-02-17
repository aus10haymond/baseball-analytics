"""
Base agent class providing common functionality for all agents.

All specialized agents (orchestrator, data quality, etc.) inherit from this base.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from shared import (
    AgentType,
    AgentTask,
    AgentResult,
    AgentAlert,
    TaskStatus,
    TaskPriority,
    AlertSeverity,
    settings,
    message_queue,
    get_agent_logger,
)


class BaseAgent(ABC):
    """Base class for all Diamond Mind agents."""
    
    def __init__(self, agent_id: AgentType):
        """
        Initialize the agent.
        
        Args:
            agent_id: Type of agent (from AgentType enum)
        """
        self.agent_id = agent_id
        self.logger = get_agent_logger(agent_id.value)
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.tasks_completed = 0
        self.tasks_failed = 0
        self._heartbeat_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the agent."""
        self.logger.info(f"Starting agent: {self.agent_id.value}")
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Agent-specific initialization
        await self.initialize()
        
        # Start main loop
        await self._run_loop()
    
    async def stop(self):
        """Stop the agent."""
        self.logger.info(f"Stopping agent: {self.agent_id.value}")
        self.is_running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Agent-specific cleanup
        await self.cleanup()
    
    @abstractmethod
    async def initialize(self):
        """
        Initialize agent-specific resources.
        
        Override this method to set up agent-specific resources,
        load models, connect to databases, etc.
        """
        pass
    
    @abstractmethod
    async def cleanup(self):
        """
        Clean up agent-specific resources.
        
        Override this method to clean up resources when agent stops.
        """
        pass
    
    @abstractmethod
    async def handle_task(self, task: AgentTask) -> AgentResult:
        """
        Handle a specific task.
        
        Override this method to implement agent-specific task handling logic.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        pass
    
    async def _run_loop(self):
        """Main agent run loop."""
        self.logger.info(f"Agent {self.agent_id.value} entering main loop")
        
        while self.is_running:
            try:
                # Check for tasks
                task = await message_queue.consume_task(timeout=5)
                
                if task and task.agent_id == self.agent_id:
                    await self._execute_task(task)
                
                # Allow other coroutines to run
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _execute_task(self, task: AgentTask):
        """
        Execute a task with error handling and result publishing.
        
        Args:
            task: Task to execute
        """
        start_time = datetime.now()
        self.logger.info(f"Executing task {task.task_id}: {task.task_type}")
        
        try:
            # Handle the task
            result = await self.handle_task(task)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            result.duration_seconds = duration
            
            # Publish result
            await message_queue.publish_result(result)
            
            self.tasks_completed += 1
            self.logger.info(f"Task {task.task_id} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            self.tasks_failed += 1
            self.logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
            
            # Create failure result
            duration = (datetime.now() - start_time).total_seconds()
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                duration_seconds=duration
            )
            await message_queue.publish_result(result)
            
            # Publish alert for critical failures
            if task.priority == TaskPriority.CRITICAL:
                await self.publish_alert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"Critical task {task.task_id} failed: {e}",
                    related_task_id=task.task_id
                )
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.is_running:
            try:
                await message_queue.update_agent_heartbeat(self.agent_id.value)
                await asyncio.sleep(settings.heartbeat_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(settings.heartbeat_interval_seconds)
    
    async def publish_alert(
        self,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        requires_action: bool = False,
        suggested_actions: Optional[list[str]] = None,
        related_task_id: Optional[str] = None
    ):
        """
        Publish an alert.
        
        Args:
            severity: Alert severity
            message: Alert message
            details: Additional details
            requires_action: Whether manual action is required
            suggested_actions: List of suggested actions
            related_task_id: Related task ID if applicable
        """
        alert = AgentAlert(
            alert_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            severity=severity,
            message=message,
            details=details or {},
            requires_action=requires_action,
            suggested_actions=suggested_actions or [],
            related_task_id=related_task_id
        )
        await message_queue.publish_alert(alert)
    
    async def publish_task(
        self,
        target_agent: AgentType,
        task_type: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> str:
        """
        Publish a task for another agent.
        
        Args:
            target_agent: Target agent to execute the task
            task_type: Type of task
            parameters: Task parameters
            priority: Task priority
            
        Returns:
            Task ID
        """
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_id=target_agent,
            task_type=task_type,
            priority=priority,
            parameters=parameters
        )
        await message_queue.publish_task(task)
        return task.task_id
    
    def get_uptime_seconds(self) -> float:
        """Get agent uptime in seconds."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0
    
    def get_error_rate(self) -> float:
        """Get task error rate."""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 0.0
        return self.tasks_failed / total_tasks
    
    @asynccontextmanager
    async def task_context(self, task_id: str):
        """
        Context manager for task execution with logging.
        
        Usage:
            async with self.task_context(task.task_id):
                # Do work
                pass
        """
        self.logger.info(f"Starting task {task_id}")
        start_time = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Task {task_id} completed in {duration:.2f}s")
