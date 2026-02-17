"""
Redis-based messaging system for agent communication.

Provides pub/sub and task queue functionality for agents to
communicate asynchronously.
"""

import json
import asyncio
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import redis.asyncio as redis

from shared.config import settings, get_redis_url
from shared.schemas import AgentTask, AgentResult, AgentAlert, TaskStatus
from shared.logging_utils import get_agent_logger

logger = get_agent_logger("messaging")


class MessageQueue:
    """Async message queue using Redis."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._pubsub = None
        
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = await redis.from_url(
                get_redis_url(),
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    # Task Queue Operations
    async def publish_task(self, task: AgentTask) -> bool:
        """
        Publish a task to the task queue.
        
        Args:
            task: AgentTask to publish
            
        Returns:
            True if successful
        """
        try:
            task_json = task.model_dump_json()
            await self.redis_client.lpush(settings.task_queue_name, task_json)
            logger.info(f"Published task {task.task_id} for agent {task.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish task: {e}")
            return False
    
    async def consume_task(self, timeout: int = 5) -> Optional[AgentTask]:
        """
        Consume a task from the queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            AgentTask if available, None otherwise
        """
        try:
            result = await self.redis_client.brpop(
                settings.task_queue_name,
                timeout=timeout
            )
            if result:
                _, task_json = result
                task_dict = json.loads(task_json)
                return AgentTask(**task_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to consume task: {e}")
            return None
    
    async def get_task_by_id(self, task_id: str) -> Optional[AgentTask]:
        """Get a specific task by ID."""
        # In a real implementation, you'd store tasks in a hash
        # This is a simplified version
        pass
    
    # Result Operations
    async def publish_result(self, result: AgentResult) -> bool:
        """Publish a task result."""
        try:
            result_json = result.model_dump_json()
            await self.redis_client.lpush(settings.result_queue_name, result_json)
            
            # Also store in a hash for easy retrieval
            await self.redis_client.hset(
                f"results:{result.task_id}",
                mapping={"data": result_json, "timestamp": datetime.now().isoformat()}
            )
            logger.info(f"Published result for task {result.task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")
            return False
    
    async def get_result(self, task_id: str) -> Optional[AgentResult]:
        """Get result for a specific task."""
        try:
            result_data = await self.redis_client.hget(f"results:{task_id}", "data")
            if result_data:
                result_dict = json.loads(result_data)
                return AgentResult(**result_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to get result: {e}")
            return None
    
    # Alert Operations
    async def publish_alert(self, alert: AgentAlert) -> bool:
        """Publish an alert."""
        try:
            alert_json = alert.model_dump_json()
            await self.redis_client.lpush(settings.alert_queue_name, alert_json)
            
            # Also publish to pub/sub for real-time notifications
            await self.redis_client.publish("alerts", alert_json)
            
            logger.warning(f"Published alert {alert.alert_id} from {alert.agent_id}: {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            return False
    
    async def consume_alerts(self, callback: Callable[[AgentAlert], None]):
        """
        Subscribe to alerts and call callback for each.
        
        Args:
            callback: Function to call with each alert
        """
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("alerts")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    alert_dict = json.loads(message["data"])
                    alert = AgentAlert(**alert_dict)
                    callback(alert)
        except Exception as e:
            logger.error(f"Error consuming alerts: {e}")
    
    # Pub/Sub for General Messages
    async def publish_message(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish a message to a channel."""
        try:
            message_json = json.dumps(message)
            await self.redis_client.publish(channel, message_json)
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to a channel."""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    message_dict = json.loads(message["data"])
                    callback(message_dict)
        except Exception as e:
            logger.error(f"Error subscribing to {channel}: {e}")
    
    # Health & Monitoring
    async def update_agent_heartbeat(self, agent_id: str):
        """Update agent heartbeat timestamp."""
        await self.redis_client.hset(
            "agent_heartbeats",
            agent_id,
            datetime.now().isoformat()
        )
    
    async def get_agent_heartbeat(self, agent_id: str) -> Optional[datetime]:
        """Get last heartbeat time for an agent."""
        heartbeat_str = await self.redis_client.hget("agent_heartbeats", agent_id)
        if heartbeat_str:
            return datetime.fromisoformat(heartbeat_str)
        return None
    
    async def get_queue_depth(self, queue_name: str) -> int:
        """Get current depth of a queue."""
        return await self.redis_client.llen(queue_name)
    
    async def clear_queue(self, queue_name: str):
        """Clear all messages from a queue."""
        await self.redis_client.delete(queue_name)


# Global message queue instance
message_queue = MessageQueue()


async def init_messaging():
    """Initialize the messaging system."""
    await message_queue.connect()


async def shutdown_messaging():
    """Shutdown the messaging system."""
    await message_queue.disconnect()
