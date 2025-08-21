import asyncio
from asyncio import Task as AsyncTask
import time
import uuid
from typing import Dict, Any, Optional, List
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_agent_parts_message, new_task
from a2a.types import (
    TaskStatusUpdateEvent, TaskStatus, TaskState, 
    TaskArtifactUpdateEvent, Artifact, Part, TextPart, FilePart, FileWithBytes, Task
)
from a2a.types import Task as A2ATask
from holos_types import TaskPlan
from logging_config import get_logger
from plant_tracer import PlantTracer

logger = get_logger('task_plan_performer')

class TaskPlanPerformer:
    """Handles the execution of task plans using coroutines."""
    
    def __init__(self, tracer: PlantTracer = None):
        self.active_task_plans: Dict[str, AsyncTask] = {}
        self.a2a_task_plans: Dict[str, TaskPlan] = {}
        self.a2a_task_plans_events: Dict[str, List] = {}
        self.tracer = tracer
        
    def get_a2a_task_plan(self, task_plan_id: str) -> TaskPlan:
        if task_plan_id not in self.a2a_task_plans:
            raise ValueError(f"Task plan {task_plan_id} not found")
        return self.a2a_task_plans[task_plan_id]
        
    def start_a2a_task_plan(self, task_plan: TaskPlan, a2a_task: Task) -> None:
        """Start a new task plan as a coroutine."""
        # Generate a unique ID for the task plan if not provided
        if not task_plan.id:
            task_plan.id = f"task_plan_{uuid.uuid4().hex[:8]}"
            
        if task_plan.id in self.active_task_plans:
            logger.warning(f"Task plan {task_plan.id} is already running")
            return

        self.a2a_task_plans[task_plan.id] = task_plan
        self.a2a_task_plans_events[task_plan.id] = []
            
        # Create and start the coroutine task
        async_task = asyncio.create_task(self._execute_task_plan(task_plan, a2a_task), name=f"task_plan_{task_plan.id}")
        self.active_task_plans[task_plan.id] = async_task

        # Add callback to clean up when task plan completes
        async_task.add_done_callback(lambda t: self._cleanup_async_task_plan(task_plan, a2a_task, t))

        logger.info(f"Started task plan {task_plan.id} as coroutine")
        
    def _cleanup_async_task_plan(self, task_plan: TaskPlan, a2a_task: Task, async_task: AsyncTask) -> None:
        a2a_task.status.state = TaskState.completed
        if task_plan.id in self.active_task_plans:
            del self.active_task_plans[task_plan.id]
        
        # Handle any exceptions that occurred
        try:
            if async_task.cancelled():
                logger.info(f"Task plan {task_plan.id} was cancelled")
            elif async_task.exception():
                logger.error(f"Task plan {task_plan.id} failed with exception: {async_task.exception()}")
            else:
                logger.info(f"Task plan {task_plan.id} completed successfully")
        except Exception as e:
            logger.error(f"Error during task plan cleanup for {task_plan.id}: {e}")
                
    async def _execute_task_plan(self, task_plan: TaskPlan, a2a_task: Task) -> Task:
        """Execute the actual task plan logic."""
        try:
            logger.info(f"Executing task plan {task_plan.id}: {task_plan.goal}, depend_plans: {len(task_plan.depend_plans)}")
            self.tracer.submit_task_plan_tracing(
                task_plan=task_plan,
                flow_role="consumer"
            )

            if not task_plan.depend_plans:
                logger.info(f"No depend plans, executing goal")
                current_task_result = await self._execute_goal(task_plan, a2a_task)
            else:
                logger.info(f"Executing {len(task_plan.depend_plans)} depend plans")
                task_results = []
                for i, depend_plan in enumerate(task_plan.depend_plans):
                    logger.info(f"Executing depend plan {i+1}/{len(task_plan.depend_plans)}: {depend_plan}")
                    task_result = await self._execute_task_plan(depend_plan, a2a_task)
                    task_results.append(task_result)
                logger.info(f"All depend plans completed, starting task plan {task_plan.goal}")
                current_task_result = await self._execute_goal(task_plan, a2a_task, task_results)

            logger.info(f"Task plan {task_plan.id} completed successfully")
            return current_task_result
        except Exception as e:
            logger.error(f"Task plan execution error: {e}", exc_info=True)
            return None

    async def _execute_goal(self, task_plan: TaskPlan, a2a_task: Task, depend_task_results: List[Task] = []) -> Task:
        """input a2a_task is the final task returned to client, output task is the task for current goal""" 
        logger.info(f"Executing goal: {task_plan.goal}, depend_task_results: {len(depend_task_results)}")
        result_task = new_task(new_agent_text_message(task_plan.goal))
        result_task.metadata = {"task_plan_id": task_plan.id}
        await asyncio.sleep(2)
        result_task.history.append(new_agent_text_message(f"Goal achieved: {task_plan.goal}"))
        self.tracer.submit_task_tracing(task=result_task, flow_role="producer_consumer")
        a2a_task.history.append(result_task.history[-1])
        return result_task

    def cancel_task_plan(self, task_plan_id: str) -> bool:
        """Cancel a running task plan."""
        if task_plan_id in self.active_task_plans:
            async_task = self.active_task_plans[task_plan_id]
            if not async_task.done():
                async_task.cancel()
                logger.info(f"Cancelled task plan {task_plan_id}")
                return True
            else:
                logger.info(f"Task plan {task_plan_id} was already completed")
                return False
        else:
            logger.warning(f"Task plan {task_plan_id} not found")
            return False 