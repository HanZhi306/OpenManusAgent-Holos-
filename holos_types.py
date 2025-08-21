import uuid
from pydantic import BaseModel, Field
from typing import Optional, Union, List


class TaskPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    depend_plans: List[Union[str, 'TaskPlan']] = Field(default_factory=list)
    task_id: Optional[str] = None
    parent_plan_id: Optional[str] = None
