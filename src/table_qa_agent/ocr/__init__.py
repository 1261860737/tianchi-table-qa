"""可插拔 OCR 后端。"""

from table_qa_agent.ocr.base import OCRBackend
from table_qa_agent.ocr.vision import VisionOCRBackend

__all__ = ["OCRBackend", "VisionOCRBackend"]
