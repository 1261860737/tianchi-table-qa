"""比赛工作簿读取、数据清洗与文档路径解析。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from table_qa_agent.schemas import QuestionRecord

REQUIRED_COLUMNS = {
    "id",
    "file_name",
    "question_type",
    "question",
    "table_hint",
    "answer_format",
}

# 原始测试表第 63 行的 question_type 被误填成了问题正文。
KNOWN_QUESTION_TYPE_FIXES: dict[int, str] = {63: "extract"}
# 第 32 题要求恢复表格结构，但原始 answer_format 被误标成 number。
KNOWN_ANSWER_FORMAT_FIXES: dict[int, str] = {32: "json"}


class DatasetError(ValueError):
    """题目表结构或内容不符合预期。"""


class DocumentNotFoundError(FileNotFoundError):
    """无法为题目定位输入文档。"""


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def read_questions(path: Path | str) -> list[QuestionRecord]:
    """读取并验证 tests.xlsx，修复数据集中已确认的单行错位。"""

    workbook_path = Path(path)
    frame = pd.read_excel(workbook_path, dtype=object)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetError(f"题目表缺少字段: {sorted(missing)}")

    records: list[QuestionRecord] = []
    for row in frame.to_dict(orient="records"):
        question_id = int(row["id"])
        question_type = KNOWN_QUESTION_TYPE_FIXES.get(
            question_id, str(row["question_type"]).strip()
        )
        answer_format = KNOWN_ANSWER_FORMAT_FIXES.get(
            question_id, str(row["answer_format"]).strip()
        )
        records.append(
            QuestionRecord(
                id=question_id,
                file_name=str(row["file_name"]).strip(),
                question_type=question_type,
                question=str(row["question"]).strip(),
                table_hint=_optional_text(row.get("table_hint")),
                answer_format=answer_format,
            )
        )

    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise DatasetError(f"题目 id 重复: {duplicates}")
    return records


class DocumentResolver:
    """解析脏文件名，支持 58.pdf / 0060.pdf 等数字零填充差异。"""

    def __init__(self, files_dir: Path | str) -> None:
        self.files_dir = Path(files_dir)
        if not self.files_dir.is_dir():
            raise NotADirectoryError(f"文档目录不存在: {self.files_dir}")
        self._by_numeric_key = self._index_numeric_files()

    def _index_numeric_files(self) -> dict[tuple[int, str], list[Path]]:
        index: dict[tuple[int, str], list[Path]] = {}
        for path in self.files_dir.iterdir():
            if path.is_file() and path.stem.isdigit():
                key = (int(path.stem), path.suffix.lower())
                index.setdefault(key, []).append(path)
        return index

    def resolve(self, file_name: str) -> Path:
        exact = self.files_dir / file_name
        if exact.is_file():
            return exact

        source = Path(file_name)
        if source.stem.isdigit():
            candidates = self._by_numeric_key.get((int(source.stem), source.suffix.lower()), [])
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                names = ", ".join(path.name for path in candidates)
                raise DocumentNotFoundError(f"文件名 {file_name!r} 对应多个候选: {names}")

        raise DocumentNotFoundError(f"无法在 {self.files_dir} 中找到题目文件: {file_name}")


def select_questions(
    questions: Iterable[QuestionRecord],
    *,
    ids: set[int] | None = None,
    limit: int | None = None,
) -> list[QuestionRecord]:
    """按 ID 与数量选择子集，便于低成本冒烟。"""

    selected = [record for record in questions if ids is None or record.id in ids]
    if ids is not None:
        found = {record.id for record in selected}
        missing = sorted(ids - found)
        if missing:
            raise DatasetError(f"未找到题目 ID: {missing}")
    if limit is not None:
        if limit <= 0:
            raise DatasetError("limit 必须大于 0")
        selected = selected[:limit]
    return selected


def summarize_dataset(
    questions: list[QuestionRecord], resolver: DocumentResolver
) -> dict[str, Any]:
    """输出数据检查摘要，不调用模型。"""

    resolved: dict[str, str] = {}
    unresolved: dict[str, str] = {}
    corrected_names: dict[str, str] = {}
    for file_name in sorted({question.file_name for question in questions}):
        try:
            path = resolver.resolve(file_name)
            resolved[file_name] = str(path)
            if path.name != file_name:
                corrected_names[file_name] = path.name
        except DocumentNotFoundError as exc:
            unresolved[file_name] = str(exc)

    return {
        "question_count": len(questions),
        "document_count": len({question.file_name for question in questions}),
        "question_types": dict(Counter(item.question_type for item in questions)),
        "answer_formats": dict(Counter(item.answer_format for item in questions)),
        "resolved_document_count": len(resolved),
        "unresolved_documents": unresolved,
        "corrected_file_names": corrected_names,
        "known_question_type_fixes": KNOWN_QUESTION_TYPE_FIXES,
        "known_answer_format_fixes": KNOWN_ANSWER_FORMAT_FIXES,
    }
