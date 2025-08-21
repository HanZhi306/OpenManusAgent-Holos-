"""
PlantTracer - A wrapper class for tracing functionality

This module provides a PlantTracer class that simplifies tracing operations
by allowing configuration of default values during initialization.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Literal
from a2a.types import Message, Task, TaskStatusUpdateEvent, AgentCard
from holos_types import TaskPlan

import requests


class PlantTracer:
    """
    A wrapper class for plant tracing functionality.
    
    This class allows you to configure default values for tracing operations,
    reducing the need to specify common parameters in each tracing call.
    """
    
    def __init__(
        self,
        base_url: str,
        role: Literal['client', 'server'],
        creator_id: str,
        agent_id: Optional[str] = None,
        agent_card: Optional[AgentCard] = None,
    ):
        """
        Initialize the PlantTracer with default values.
        
        Args:
            base_url: API base URL for tracing operations
            role: Default role ('client' or 'server')
            creator_id: Default creator ID
            agent_id: Optional default agent ID
            agent_card: Optional agent card (will use agent_card.url as agent_id if provided)
        """
        self.role = role
        self.creator_id = creator_id
        self.agent_id = agent_id
        self.api_base_url = base_url
        
        # Extract agent_id from agent_card if provided and agent_id not already set
        if agent_card and not self.agent_id:
            self.agent_id = agent_card.url
    
    def _convert_to_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Convert an object to a dictionary, handling both A2A objects and regular dicts.
        
        Args:
            obj: Object to convert (A2A object or dict)
        
        Returns:
            Dictionary representation of the object
        """
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(exclude_none=True)
        elif isinstance(obj, dict):
            return obj
        else:
            return obj
    
    def _create_tracing_data_entry(
        self,
        object_kind: str,
        object_data: Any,
        event_name: str,
        flow_role: Literal['producer', 'consumer', 'producer_consumer'],
        entry_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a tracing data entry with default values from the tracer.
        
        Args:
            object_kind: Type of object being traced
            object_data: The actual object data being traced
            event_name: Name of the event
            flow_role: Flow role ('producer', 'consumer', or 'producer_consumer')
            entry_id: Optional unique identifier (will generate if not provided)
            parent_id: Optional parent ID for hierarchical relationships
            timestamp: Optional timestamp (will use current time if not provided)
        
        Returns:
            Dict containing the formatted tracing data entry
        """
        if not entry_id:
            entry_id = str(uuid.uuid4())
        
        if not timestamp:
            timestamp = datetime.utcnow().isoformat()
        
        return {
            "id": entry_id,
            "creator_id": self.creator_id,
            "parent_id": parent_id,
            "object_kind": object_kind,
            "object": self._convert_to_dict(object_data),
            "event_name": event_name,
            "role": self.role,
            "flow_role": flow_role,
            "agent_id": self.agent_id,
            "timestamp": timestamp
        }
    
    def submit_tracing_data(self, tracing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit tracing data to the API.
        
        Args:
            tracing_data: Dictionary containing tracing data
        
        Returns:
            Dict containing the response with id, event_name, timestamp, status, and message
        """
        url = f"{self.api_base_url}/plant/trace"
        
        try:
            response = requests.post(url, json=tracing_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to submit tracing data: {str(e)}"
            }
    
    def submit_task_plan_tracing(
        self,
        task_plan: TaskPlan,
        flow_role: Literal['producer', 'consumer', 'producer_consumer'],
        event_name: str = "Create task plan",
        entry_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit tracing data for a task plan.
        
        Args:
            task_plan: Dictionary containing task plan data with goal, depend_plans, task_id, parent_plan_id
            flow_role: Flow role ('producer', 'consumer', or 'producer_consumer')
            event_name: Name of the event
            entry_id: Optional unique identifier
            parent_id: Optional parent ID
        
        Returns:
            Dict containing the response from the tracing API
        """
        tracing_data = self._create_tracing_data_entry(
            object_kind="task_plan",
            object_data=task_plan,
            event_name=event_name,
            flow_role=flow_role,
            entry_id=entry_id,
            parent_id=parent_id,
        )
        
        return self.submit_tracing_data(tracing_data)
    
    def submit_message_tracing(
        self,
        message: Message,
        flow_role: Literal['producer', 'consumer', 'producer_consumer'],
        event_name: str = "Send message",
        entry_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit tracing data for a message.
        
        Args:
            message: A2A Message object
            flow_role: Flow role ('producer', 'consumer', or 'producer_consumer')
            event_name: Name of the event
            entry_id: Optional unique identifier
            parent_id: Optional parent ID
        
        Returns:
            Dict containing the response from the tracing API
        """
        tracing_data = self._create_tracing_data_entry(
            object_kind="message",
            object_data=message,
            event_name=event_name,
            flow_role=flow_role,
            entry_id=entry_id,
            parent_id=parent_id,
        )
        
        return self.submit_tracing_data(tracing_data)
    
    def submit_task_tracing(
        self,
        task: Task,
        flow_role: Literal['producer', 'consumer', 'producer_consumer'],
        event_name: str = "Create task",
        entry_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit tracing data for a task.
        
        Args:
            task: A2A Task object
            flow_role: Flow role ('producer', 'consumer', or 'producer_consumer')
            event_name: Name of the event
            entry_id: Optional unique identifier
            parent_id: Optional parent ID
        
        Returns:
            Dict containing the response from the tracing API
        """
        tracing_data = self._create_tracing_data_entry(
            object_kind="task",
            object_data=task,
            event_name=event_name,
            flow_role=flow_role,
            entry_id=entry_id,
            parent_id=parent_id,
        )
        
        return self.submit_tracing_data(tracing_data)
    
    def submit_task_update_event_tracing(
        self,
        task_update_event: TaskStatusUpdateEvent,
        flow_role: Literal['producer', 'consumer', 'producer_consumer'],
        event_name: str = "Update task status",
        entry_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit tracing data for a task update event.
        
        Args:
            task_update_event: A2A TaskStatusUpdateEvent object
            flow_role: Flow role ('producer', 'consumer', or 'producer_consumer')
            event_name: Name of the event
            entry_id: Optional unique identifier
            parent_id: Optional parent ID
        
        Returns:
            Dict containing the response from the tracing API
        """
        tracing_data = self._create_tracing_data_entry(
            object_kind="task_update_event",
            object_data=task_update_event,
            event_name=event_name,
            flow_role=flow_role,
            entry_id=entry_id,
            parent_id=parent_id,
        )
        
        return self.submit_tracing_data(tracing_data)
    
    def get_tracing_data(self, tracing_id: str) -> Dict[str, Any]:
        """
        Get tracing data by ID.
        
        Args:
            tracing_id: The ID of the tracing data entry to retrieve
        
        Returns:
            Dict containing the tracing data entry or error information
        """
        url = f"{self.api_base_url}/plant/trace/{tracing_id}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to get tracing data: {str(e)}"
            }
    
    def get_tracing_data_list(
        self,
        skip: int = 0,
        limit: int = 100,
        object_kind: Optional[str] = None,
        parent_id: Optional[str] = None,
        include_objects: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Get list of all tracing data with pagination and filtering.
        
        Args:
            skip: Number of items to skip (default: 0)
            limit: Maximum number of items to return (default: 100, max: 100)
            object_kind: Filter by object kind (e.g., 'task_plan', 'message', 'task', 'task_update_event')
            parent_id: Filter by parent ID for hierarchical relationships
            include_objects: List of object types to include in response
        
        Returns:
            Dict containing paginated tracing data with items, total, page, size, and pages
        """
        url = f"{self.api_base_url}/plant/trace"
        params = {
            "skip": skip,
            "limit": limit
        }
        
        if object_kind:
            params["object_kind"] = object_kind
        if parent_id:
            params["parent_id"] = parent_id
        if include_objects:
            params["include_objects"] = include_objects
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to get tracing data list: {str(e)}"
            }
    
    def get_tracing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tracing data.
        
        Returns:
            Dict containing tracing statistics including counts and relationships
        """
        url = f"{self.api_base_url}/plant/stats"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to get tracing stats: {str(e)}"
            } 