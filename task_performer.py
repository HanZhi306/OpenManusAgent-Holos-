import asyncio
from asyncio import Task as AsyncTask
import time
import base64
import uuid
from typing import Dict, Any, Optional, List
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_agent_parts_message
from a2a.types import (
    TaskStatusUpdateEvent, TaskStatus, TaskState, 
    TaskArtifactUpdateEvent, Artifact, Part, TextPart, FilePart, FileWithBytes, Task
)
from a2a.types import Task as A2ATask
from logging_config import get_logger
from app.agent.manus import Manus

logger = get_logger('task_performer')

class TaskPerformer:
    """Handles the execution of different types of tasks using coroutines."""
    
    def __init__(self):
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.a2a_tasks: Dict[str, A2ATask] = {}
        self.a2a_tasks_events: Dict[str, List] = {}
        
    def get_a2a_task(self, task_id: str) -> A2ATask:
        if task_id not in self.a2a_tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.a2a_tasks[task_id]
        
    def start_a2a_task(self, task: A2ATask) -> None:
        """Start a new task as a coroutine."""
        if task.id in self.active_tasks:
            logger.warning(f"Task {task.id} is already running")
            return
        self.a2a_tasks[task.id] = task
        self.a2a_tasks_events[task.id] = []
            
        # Create and start the coroutine task
        async_task = asyncio.create_task(self._execute_task(task), name=f"task_{task.id}")
        self.active_tasks[task.id] = async_task
        
        # Add callback to clean up when task completes
        async_task.add_done_callback(lambda t: self._cleanup_async_task(task.id, t))

        logger.info(f"Started task {task.id} as coroutine")
        
    def _cleanup_async_task(self, task_id: str, async_task: AsyncTask) -> None:
        """Clean up completed task."""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        # Handle any exceptions that occurred
        try:
            if async_task.cancelled():
                logger.info(f"Task {task_id} was cancelled")
            elif async_task.exception():
                logger.error(f"Task {task_id} failed with exception: {async_task.exception()}")
            else:
                logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            logger.error(f"Error during task cleanup for {task_id}: {e}")
                
    async def _execute_task(self, task: A2ATask) -> None:
        """Execute the actual task logic."""
        try:
            logger.info(f"Executing task {task.id}: {task.history[0].parts[0].root.text}")
            '''
            self.tracer.submit_task_tracing(
                task=task,
                flow_role="consumer"
            )
            '''
            # Send initial status
            task.status = TaskStatus(
                state=TaskState.working,
                message=new_agent_text_message(f"Starting task: {task.history[0].parts[0].root.text}", task.context_id, task.id),
            )
            task.history.append(task.status.message)
            self.a2a_tasks_events[task.id].append(TaskStatusUpdateEvent(
                status=task.status,
                final=False,
                context_id=task.context_id,
                taskId=task.id,
            ))
            
            logger.info(f"Performing task: {task.history[0].parts[0].root.text}")
            prompt = task.history[0].parts[0].root.text
            agent = await Manus.create()
            new_message_text = await agent.run(prompt)
            new_message_parts = [TextPart(text=new_message_text)]
            task.history.append(new_agent_parts_message(new_message_parts, task.context_id, task.id))
            

            # Send completion status
            task.status = TaskStatus(
                state=TaskState.completed,
                message=new_agent_text_message(f"Task completed", task.context_id, task.id),
            )
            task.history.append(task.status.message)
            self.a2a_tasks_events[task.id].append(TaskStatusUpdateEvent(
                status=task.status,
                final=True,
                context_id=task.context_id,
                taskId=task.id,
            ))
            

        except Exception as e:
            logger.error(f"Task execution error: {e}", exc_info=True)
            try:
                task.status = TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(f"Task failed: {str(e)}", task.context_id, task.id),
                )
                task.history.append(task.status.message)
                self.a2a_tasks_events[task.id].append(TaskStatusUpdateEvent(
                    status=task.status,
                    final=True,
                    context_id=task.context_id,
                    taskId=task.id,
                ))
            except Exception as send_error:
                logger.error(f"Failed to send task error message: {send_error}")
            
       
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task {task_id}")
                return True
            else:
                logger.info(f"Task {task_id} was already completed")
                return True
        return False