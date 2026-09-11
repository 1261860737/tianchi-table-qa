"""首次识别失败后，用视觉模型定位少量高价值区域。"""

from __future__ import annotations

from json_repair import loads as repair_json_loads

from table_qa_agent.client import OpenAICompatibleVLClient
from table_qa_agent.schemas import QuestionRecord, RegionRef, TaskPlan, TokenUsage

LOCATOR_PROMPT = """你是文档表格区域定位器。根据问题和页面缩略图，返回最可能包含
目标字段、相关行头和列头的候选区域。bbox 使用 [x1,y1,x2,y2] 归一化坐标，范围 0 到 1。
区域要保留足够行列上下文，不要只框数字。最多返回指定数量，只输出 JSON：
{"regions":[{"page":1,"bbox":[0.0,0.0,1.0,1.0],"reason":"...","confidence":0.8}]}。
"""


class RegionLocationError(ValueError):
    """候选区域输出不可用。"""

    def __init__(self, message: str, *, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class PageRegionLocator:
    def __init__(self, client: OpenAICompatibleVLClient, *, max_regions: int = 3) -> None:
        self.client = client
        self.max_regions = max_regions

    def locate(
        self,
        question: QuestionRecord,
        plan: TaskPlan,
        image_content: list[dict[str, object]],
        *,
        failure_reason: str,
    ) -> tuple[list[RegionRef], TokenUsage]:
        user_text = (
            f"问题：{question.question}\n"
            f"任务计划：{plan.model_dump_json()}\n"
            f"首次失败：{failure_reason}\n"
            f"最多返回 {self.max_regions} 个区域。"
        )
        completion = self.client.complete(
            system_prompt=LOCATOR_PROMPT,
            user_text=user_text,
            image_content=image_content,
        )
        try:
            parsed = repair_json_loads(completion.text)
            raw_regions = parsed.get("regions") if isinstance(parsed, dict) else None
            if not isinstance(raw_regions, list):
                raise ValueError("regions 不是数组")
            regions = [RegionRef.model_validate(item) for item in raw_regions[: self.max_regions]]
            if not regions:
                raise ValueError("没有候选区域")
        except Exception as exc:
            raise RegionLocationError("区域定位输出无法解析", usage=completion.usage) from exc
        return regions, completion.usage
