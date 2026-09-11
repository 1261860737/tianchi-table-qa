"""把任务计划路由到职责单一的专家 Agent。"""

from __future__ import annotations

from typing import Any, Protocol

from table_qa_agent.schemas import SpecialistName, TaskPlan


class Specialist(Protocol):
    def run(self, *args: Any, **kwargs: Any) -> Any: ...


class AgentRouter:
    def __init__(self, specialists: dict[SpecialistName, Specialist]) -> None:
        missing = {"extract", "compute", "structure", "visual_attribute"} - set(specialists)
        if missing:
            raise ValueError(f"缺少专家 Agent: {sorted(missing)}")
        self._specialists = specialists

    def route(self, plan: TaskPlan) -> Specialist:
        return self._specialists[plan.specialist]
