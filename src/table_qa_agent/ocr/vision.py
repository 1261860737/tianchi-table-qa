"""使用 OpenAI 兼容视觉模型生成带位置的 OCR 候选。"""

from __future__ import annotations

from json_repair import loads as repair_json_loads

from table_qa_agent.client import OpenAICompatibleVLClient
from table_qa_agent.schemas import OCRResult, QuestionRecord, TokenUsage

OCR_PROMPT = """你是表格 OCR 工具。逐字识别候选区域中的文字和数字，保留标点、负号、
百分号、括号、单位和大小写。不要回答问题，不要计算。bbox 使用当前候选图的归一化坐标。
只输出 JSON：{"blocks":[{"text":"...","bbox":[0,0,1,1],"confidence":0.9}]}。
"""


class VisionOCRBackend:
    def __init__(self, client: OpenAICompatibleVLClient) -> None:
        self.client = client

    def recognize(
        self,
        question: QuestionRecord,
        image_content: list[dict[str, object]],
    ) -> tuple[OCRResult, TokenUsage]:
        completion = self.client.complete(
            system_prompt=OCR_PROMPT,
            user_text=f"识别与下列问题相关的全部可见文本，但不要回答：{question.question}",
            image_content=image_content,
        )
        parsed = repair_json_loads(completion.text)
        return OCRResult.model_validate(parsed), completion.usage
