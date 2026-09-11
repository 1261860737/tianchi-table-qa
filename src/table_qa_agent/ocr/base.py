"""OCR 后端公共接口。"""

from __future__ import annotations

from typing import Protocol

from table_qa_agent.schemas import OCRResult, QuestionRecord, TokenUsage


class OCRBackend(Protocol):
    def recognize(
        self,
        question: QuestionRecord,
        image_content: list[dict[str, object]],
    ) -> tuple[OCRResult, TokenUsage]: ...
