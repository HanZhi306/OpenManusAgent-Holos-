import time
import asyncio
from typing import Optional, AsyncGenerator

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_task
from a2a.types import TaskStatusUpdateEvent, TaskStatus, TaskState, Task, FilePart, TextPart, Message, TaskArtifactUpdateEvent
from a2a.utils.telemetry import trace_function, SpanKind
from llm_client import LLMClient
from config import LLM_MODEL, API_BASE_URL
from logging_config import get_logger
from task_performer import TaskPerformer
from task_plan_performer import TaskPlanPerformer
from typing import Union
from holos_types import TaskPlan
from plant_tracer import PlantTracer

logger = get_logger('OpenManus_executor')

A2AResult = Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent, Task, Message]


class SimpleTestAgentExecutor(AgentExecutor):
    """OpenManus Agent Executor Implementation."""

    def __init__(self):
        self.llm_client = LLMClient()
        self.task_performer = TaskPerformer()
        
        self.tracer = PlantTracer(
            role="server",
            creator_id="simple-test-agent",
            base_url=API_BASE_URL
        )
        
        self.task_plan_performer = TaskPlanPerformer(tracer=self.tracer)
        self.system_prompt = """You are OpenManus Agent. 
You are helpful and provide clear, concise responses.
If the user asks for something you cannot do, explain what you can help with instead."""

        logger.info(f"OpenManusAgentExecutor initialized with model: {LLM_MODEL}")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        start_time = time.time()
        logger.info(f"Executing context: {context.context_id}")
        logger.info(f"context.call_context: {context.call_context}")
        logger.info(f"context._params: {context._params}")

        self.tracer.submit_message_tracing(
            message=context.message,
            flow_role="consumer"
        )

        try:
            user_input = context.get_user_input()
            logger.info(f"Input: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
            is_streaming = False
            if context.configuration and context.configuration.blocking is not None:
                is_streaming = not context.configuration.blocking

            if not is_streaming:
                logger.info("Processing non-streaming response")

                a2a_result: A2AResult = await self._handle_non_streaming_response(user_input, event_queue, context)
                if a2a_result:
                    await event_queue.enqueue_event(a2a_result)
            else:
                logger.info("Processing streaming response")
                a2a_result_generator = await self._handle_streaming_response(user_input, event_queue, context)
                async for a2a_result in a2a_result_generator:
                    await event_queue.enqueue_event(a2a_result)

            execution_time = time.time() - start_time
            logger.info(f"Completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Execution error after {execution_time:.2f}s: {e}", exc_info=True)
            
            try:
                error_message = f"I apologize, but I encountered an error while processing your request: {str(e)}"
                logger.info("Sending error response")
                await event_queue.enqueue_event(new_agent_text_message(error_message))
            except Exception as enqueue_error:
                logger.error(f"Failed to send error response: {enqueue_error}", exc_info=True)


    async def _call_llm(self, user_input: str) -> str:
        """Call LLM to generate a response."""
        if not self.llm_client.is_available():
            return f"I'm sorry, but I don't have access to advanced AI capabilities at the moment. However, I can still help you. Please provide more details about what you'd like to accomplish."
        try:
            response = await self.llm_client.simple_chat(user_input, self.system_prompt)
            return response
            
        except Exception as e:
            logger.error(f"LLM API error: {e}", exc_info=True)
            return f"I encountered an error while processing your request with AI. Please provide more details about what you'd like to accomplish, and I'll do my best to help you."

    async def _create_a2a_task(self, task_description: str, event_queue: EventQueue, context: RequestContext) -> Task|Message:
        try:
            if not task_description:
                error_message = "Please provide a description for the task"
                a2a_result = new_agent_text_message(error_message)
                return a2a_result
            
            task = new_task(context.message)
            task.status.message = new_agent_text_message(f"Task created: {task_description}", task.context_id, task.id)
            
            if not task.metadata:
                task.metadata = {}
            task.metadata['task_description'] = task_description
            task.metadata['created_by'] = 'simple_test_agent'
            
            logger.info(f"Created task {task.id} with description: {task_description}")
            return task
            
        except Exception as e:
            logger.error(f"Task creation error: {e}", exc_info=True)
            error_message = f"I encountered an error while creating the task: {str(e)}"
            await event_queue.enqueue_event(new_agent_text_message(error_message))
            return None

    async def _create_task_plan(self, task_description: str, event_queue: EventQueue, context: RequestContext) -> TaskPlan:
        task_plan = TaskPlan(goal="/task 7月13号AI领域有什么进展")

        task_plan11 = TaskPlan(goal="从arxiv收集7月13号AI领域论文")
        task_plan12 = TaskPlan(goal="理解论文内容形成前沿解读")
        task_plan13 = TaskPlan(goal="根据创新性对论文排序")
        task_plan14 = TaskPlan(goal="总结形成最终报告")
        task_plan.depend_plans = [task_plan11, task_plan12, task_plan13, task_plan14]
        task_plan11.parent_plan_id = task_plan.id
        task_plan12.parent_plan_id = task_plan.id
        task_plan13.parent_plan_id = task_plan.id
        task_plan14.parent_plan_id = task_plan.id

        task_plan121 = TaskPlan(goal="对比同作者之前工作")
        task_plan122 = TaskPlan(goal="对比不同作者类似工作")
        task_plan12.depend_plans = [task_plan121, task_plan122]
        task_plan121.parent_plan_id = task_plan12.id
        task_plan122.parent_plan_id = task_plan12.id
        
        return task_plan

    async def _handle_non_streaming_response(self, user_input: str, event_queue: EventQueue, context: RequestContext) -> A2AResult:
        if user_input.lower().startswith('/task '):
            logger.info("Creating new task based on user input")
            a2a_task = await self._create_a2a_task(user_input[6:].strip(), event_queue, context)
            if isinstance(a2a_task, Task):
                self.tracer.submit_task_tracing(task=a2a_task, flow_role="producer")
                #The task_performer don't need the event_queue,
                #  DefaultRequestHandler's task_manager share the same task object with the task_performer
                #  so we don't need the event_queue to update task's status, just modify it directly
                logger.info(f"Starting task execution as coroutine for task {a2a_task.id}")
                self.task_performer.start_a2a_task(a2a_task)
            return a2a_task
        elif user_input.lower().startswith('/sleep '):
            await asyncio.sleep(float(user_input[6:].strip()))
            return new_agent_text_message("I'm back!")
        elif user_input.lower().startswith('/plan '):
            a2a_task_plan = await self._create_task_plan(user_input[6:].strip(), event_queue, context)
            a2a_task = new_task(context.message)
            a2a_task.history.append(new_agent_text_message(f"Task goal: {a2a_task_plan.goal}"))
            if isinstance(a2a_task_plan, TaskPlan):
                self.tracer.submit_task_plan_tracing(task_plan=a2a_task_plan, flow_role="producer")
                logger.info(f"Starting task plan execution as coroutine for {a2a_task_plan}")
                self.task_plan_performer.start_a2a_task_plan(a2a_task_plan, a2a_task)
            self.tracer.submit_task_tracing(task=a2a_task, flow_role="producer")
            return a2a_task
        else:
            result = ""
            for part in context.message.parts:
                if isinstance(part.root, FilePart):
                    result += f"Received file: {part.root.file.name} ({part.root.file.mimeType}, {len(part.root.file.bytes)} bytes)\n"
            result += await self._call_llm(user_input)
            logger.info(f"Response: {result[:50]}...")
            a2a_message = new_agent_text_message(result)
            return a2a_message






    async def _call_llm_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Call LLM to generate a streaming response."""
        if not self.llm_client.is_available():
            yield f"I'm sorry, but I don't have access to advanced AI capabilities at the moment. However, I can still help you. Please provide more details about what you'd like to accomplish."
            return

        try:
            async for chunk in self.llm_client.simple_chat_stream(user_input, self.system_prompt):
                yield chunk
        except Exception as e:
            logger.error(f"LLM streaming error: {e}", exc_info=True)
            yield f"I encountered an error while processing your request with AI. Please provide more details about what you'd like to accomplish, and I'll do my best to help you."

    async def _handle_streaming_response(self, user_input: str, event_queue: EventQueue, context: RequestContext) -> AsyncGenerator[A2AResult, None]:
        """Handle streaming response from the LLM."""
        try:
            chunk_count = 0
            async for chunk in self._call_llm_stream(user_input):
                chunk_count += 1
                if chunk:
                    logger.info(f"Chunk #{chunk_count}: {len(chunk)} chars")
                    a2a_result = TaskStatusUpdateEvent(
                            status=TaskStatus(
                                state=TaskState.working,
                                message=new_agent_text_message(chunk, context.context_id, context.task_id),
                            ),
                            final=False,
                            contextId=context.context_id,
                            taskId=context.task_id,
                        )
                    yield a2a_result
            logger.info(f"Streaming completed: {chunk_count} chunks")
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_message = f"I encountered an error while streaming the response: {str(e)}"
            a2a_result = new_agent_text_message(error_message, context.context_id, context.task_id)
            yield a2a_result


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel ongoing tasks for this context."""
        logger.info(f"Cancel requested for context: {context.context_id}")
        
        try:
            # Try to cancel regular task
            cancelled = self.task_performer.cancel_task(context.task_id)
            if cancelled:
                logger.info(f"Successfully cancelled task {context.task_id}")
                await event_queue.enqueue_event(
                    new_agent_text_message(f"Task {context.task_id} has been cancelled.")
                )
                return
                
            # Try to cancel task plan
            cancelled_plan = self.task_plan_performer.cancel_task_plan(context.task_id)
            if cancelled_plan:
                logger.info(f"Successfully cancelled task plan {context.task_id}")
                await event_queue.enqueue_event(
                    new_agent_text_message(f"Task plan {context.task_id} has been cancelled.")
                )
                return
                
            logger.info(f"Task/Task plan {context.task_id} was not found or already completed")
                
        except Exception as e:
            logger.error(f"Error during cancellation: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error during cancellation: {str(e)}")
            ) 
