import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from table_qa_agent.agent import AgentOutput
from table_qa_agent.client import ModelToolCall
from table_qa_agent.config import DocumentConfig, RuntimeConfig
from table_qa_agent.dataset import DocumentResolver
from table_qa_agent.documents import DocumentProcessor
from table_qa_agent.normalizer import validate_answer_text
from table_qa_agent.pipeline import (
    BaselinePipeline,
    JsonlRunStore,
    SubmissionBuildError,
    find_error_derived_empty_ids,
    force_complete_results,
    publish_submission,
    select_submission_results,
    validate_submission,
    write_submission,
)
from table_qa_agent.schemas import (
    EvidenceItem,
    EvidenceResponse,
    OCRResult,
    OCRTextBlock,
    OperationSpec,
    QuestionRecord,
    RegionRef,
    RunResult,
    TokenUsage,
)


class FakeEvidenceAgent:
    def run(self, question: QuestionRecord, image_content: list[dict]) -> AgentOutput:
        assert question.id == 1
        assert image_content
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="thinking",
                evidence=[
                    EvidenceItem(value_raw="150", value=150, page=1),
                    EvidenceItem(value_raw="120", value=120, page=1),
                ],
                operation=OperationSpec(name="subtract", arguments={"a": 150, "b": 120}),
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=10),
        )


class LiteralAnswerAgent:
    """实验模式下返回不与 Evidence 绑定的直接答案。"""

    def run(self, question: QuestionRecord, image_content: list[dict]) -> AgentOutput:
        assert image_content
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="extract",
                evidence=[EvidenceItem(value_raw="图中候选", value="图中候选", page=1)],
                operation=OperationSpec(name="lookup", arguments={"value": 42}),
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


class EmptyAnswerAgent:
    """模拟模型在 format-only 策略下仍返回空字符串。"""

    def run(self, question: QuestionRecord, image_content: list[dict]) -> AgentOutput:
        assert image_content
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="extract",
                evidence=[],
                operation=OperationSpec(name="lookup", arguments={"value": ""}),
                output={"type": "string"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


class DirectExtractAgent:
    def run(self, question: QuestionRecord, image_content: list[dict], **_: object) -> AgentOutput:
        assert image_content
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="extract",
                evidence=[EvidenceItem(value_raw="42", value=42, page=1)],
                direct_answer=42,
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


class ToolCallingExtractAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        question: QuestionRecord,
        image_content: list[dict],
        **kwargs: object,
    ) -> AgentOutput:
        self.calls += 1
        assert image_content
        if self.calls == 1:
            return AgentOutput(
                evidence=EvidenceResponse(
                    status="insufficient",
                    question_type="extract",
                    reason="需要读取高清表格",
                ),
                raw_text="",
                usage=TokenUsage(total_tokens=1),
                tool_calls=[
                    ModelToolCall(
                        id="call_1",
                        name="inspect_table_region",
                        arguments={"query": "目标字段", "reason": "原图文字太小"},
                    )
                ],
            )
        assert kwargs["ocr_text"] == "目标字段 42"
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="extract",
                evidence=[EvidenceItem(value_raw="42", value=42, page=1)],
                direct_answer=42,
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


class SoftRiskRepairAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *_: object, **__: object) -> AgentOutput:
        self.calls += 1
        if self.calls == 1:
            return AgentOutput(
                evidence=EvidenceResponse(
                    status="success",
                    question_type="extract",
                    evidence=[EvidenceItem(value_raw="40", value=40, page=1)],
                    operation=OperationSpec(name="lookup", arguments={"value": 41}),
                    output={"type": "number"},
                ),
                raw_text="{}",
                usage=TokenUsage(total_tokens=1),
            )
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="extract",
                evidence=[EvidenceItem(value_raw="42", value=42, page=1)],
                direct_answer=42,
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


class CountingFakeEvidenceAgent:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def run(self, question: QuestionRecord, image_content: list[dict]) -> AgentOutput:
        assert image_content
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            return AgentOutput(
                evidence=EvidenceResponse(
                    status="success",
                    question_type="extract",
                    evidence=[EvidenceItem(value_raw=str(question.id), value=question.id, page=1)],
                    operation=OperationSpec(name="lookup", arguments={"value": question.id}),
                    output={"type": "number"},
                ),
                raw_text="{}",
                usage=TokenUsage(total_tokens=1),
            )
        finally:
            with self.lock:
                self.active -= 1


class StructureFakeEvidenceAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, question: QuestionRecord, image_content: list[dict]) -> AgentOutput:
        self.calls += 1
        structure = {
            "row_count": 2,
            "col_count": 2,
            "cells": [
                {"text": "项目", "row": 0, "col": 0, "rowspan": 2, "colspan": 1},
                {"text": "金额", "row": 0, "col": 1, "rowspan": 1, "colspan": 1},
            ],
        }
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="structure",
                evidence=[EvidenceItem(value_raw="局部结构", value=structure, page=1)],
                operation=OperationSpec(name="lookup", arguments={"value": {"evidence_index": 0}}),
                output={"type": "json"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


class RecoveringFakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        question: QuestionRecord,
        image_content: list[dict],
        **kwargs: object,
    ) -> AgentOutput:
        self.calls += 1
        assert image_content
        if self.calls == 1:
            return AgentOutput(
                evidence=EvidenceResponse(
                    status="insufficient",
                    question_type="extract",
                    reason="文字过小",
                ),
                raw_text="{}",
                usage=TokenUsage(total_tokens=1),
            )
        assert kwargs["ocr_text"] == "目标字段 42"
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="extract",
                evidence=[EvidenceItem(id="answer", value_raw="42", value=42, page=1)],
                operation=OperationSpec(
                    name="lookup",
                    arguments={"value": {"evidence_id": "answer"}},
                ),
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


class FakeLocator:
    def locate(self, *_: object, **__: object) -> tuple[list[RegionRef], TokenUsage]:
        return [RegionRef(page=1, bbox=(0.1, 0.1, 0.9, 0.9))], TokenUsage(total_tokens=1)


class FakeOCR:
    def recognize(self, *_: object, **__: object) -> tuple[OCRResult, TokenUsage]:
        return OCRResult(blocks=[OCRTextBlock(text="目标字段 42")]), TokenUsage(total_tokens=1)


class RepairingOperationAgent:
    def run(self, *_: object, **__: object) -> AgentOutput:
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="thinking",
                evidence=[
                    EvidenceItem(id="left", value_raw="1", value=1),
                    EvidenceItem(id="right", value_raw="2", value=2),
                ],
                operation=OperationSpec(name="add", arguments={"a": 1, "b": 99}),
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )

    def repair_operation(self, *_: object, **__: object) -> AgentOutput:
        return AgentOutput(
            evidence=EvidenceResponse(
                status="success",
                question_type="thinking",
                evidence=[
                    EvidenceItem(id="left", value_raw="1", value=1),
                    EvidenceItem(id="right", value_raw="2", value=2),
                ],
                operation=OperationSpec(
                    name="add",
                    arguments={
                        "a": {"evidence_id": "left"},
                        "b": {"evidence_id": "right"},
                    },
                ),
                output={"type": "number"},
            ),
            raw_text="{}",
            usage=TokenUsage(total_tokens=1),
        )


def test_submission_round_trip(tmp_path: Path) -> None:
    questions = [
        QuestionRecord(
            id=1,
            file_name="001.pdf",
            question_type="extract",
            question="测试",
            answer_format="number",
        )
    ]
    output = tmp_path / "submission.xlsx"
    write_submission(
        [
            RunResult(
                question_id=1,
                question="测试",
                document_id="001.pdf",
                final_answer="25",
                status="success",
            )
        ],
        output,
    )

    assert validate_submission(output, questions) == {
        "is_valid": True,
        "row_count": 1,
        "empty_answer_count": 0,
        "empty_answer_ids": [],
        "format_error_count": 0,
        "format_errors": [],
    }


def test_submission_validator_checks_structure_contract(tmp_path: Path) -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="structure",
        question="恢复表格结构",
        answer_format="json",
    )
    output = tmp_path / "submission.xlsx"
    write_submission(
        [
            RunResult(
                question_id=1,
                question=question.question,
                document_id=question.file_name,
                final_answer='["项目","金额"]',
                status="success",
            )
        ],
        output,
    )

    report = validate_submission(output, [question])

    assert report["is_valid"] is False
    assert report["format_error_count"] == 1
    assert report["format_errors"][0]["id"] == 1


def test_submission_validator_allows_blank_answer(tmp_path: Path) -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="structure",
        question="恢复表格结构",
        answer_format="json",
    )
    output = tmp_path / "submission.xlsx"
    write_submission(
        [RunResult(question_id=1, question="测试", document_id="001.png")],
        output,
        [question],
    )

    report = validate_submission(output, [question])

    assert report["is_valid"] is True
    assert report["empty_answer_ids"] == [1]
    frame = pd.read_excel(output, dtype=object, keep_default_na=False)
    assert frame.loc[0, "answer"] == '{"row_count":0,"col_count":0,"cells":[]}'


def test_submission_uses_array_placeholder_for_blank_json_array(tmp_path: Path) -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="thinking",
        question="列出结果",
        answer_format="json_array",
    )
    output = tmp_path / "submission.xlsx"
    write_submission(
        [RunResult(question_id=1, question=question.question, document_id="001.png")],
        output,
        [question],
    )

    frame = pd.read_excel(output, dtype=object, keep_default_na=False)
    assert frame.loc[0, "answer"] == "[]"
    assert validate_submission(output, [question])["empty_answer_ids"] == [1]


def test_submission_selection_keeps_older_valid_success_after_new_failure() -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="值是多少？",
        answer_format="number",
    )
    success = RunResult(
        question_id=1,
        question=question.question,
        document_id=question.file_name,
        final_answer="42",
        status="success",
    )
    failure = RunResult(
        question_id=1,
        question=question.question,
        document_id=question.file_name,
        status="error",
        error_type="E11_OUTPUT_PROTOCOL_ERROR",
    )

    selected = select_submission_results([question], [[success, failure]])

    assert selected[0].final_answer == "42"
    assert find_error_derived_empty_ids(selected, [question]) == []


def test_error_derived_empty_is_reported_but_insufficient_empty_is_allowed() -> None:
    questions = [
        QuestionRecord(
            id=index,
            file_name="001.png",
            question_type="extract",
            question=f"问题 {index}",
            answer_format="string",
        )
        for index in (1, 2)
    ]
    results = [
        RunResult(
            question_id=1,
            question="问题 1",
            document_id="001.png",
            status="error",
        ),
        RunResult(
            question_id=2,
            question="问题 2",
            document_id="001.png",
            status="insufficient",
        ),
    ]

    assert find_error_derived_empty_ids(results, questions) == [1]


def test_publish_submission_replaces_existing_file_after_validation(tmp_path: Path) -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="值是多少？",
        answer_format="number",
    )
    output = tmp_path / "submission.xlsx"
    write_submission(
        [
            RunResult(
                question_id=1,
                question=question.question,
                document_id=question.file_name,
                final_answer="7",
                status="success",
            )
        ],
        output,
        [question],
    )

    report = publish_submission(
        [
            RunResult(
                question_id=1,
                question=question.question,
                document_id=question.file_name,
                final_answer="42",
                status="success",
            )
        ],
        output,
        [question],
    )

    frame = pd.read_excel(output, dtype=object, keep_default_na=False)
    assert frame.loc[0, "answer"] == "42"
    assert report["is_valid"] is True
    assert report["error_derived_empty_count"] == 0
    assert not list(tmp_path.glob(".*.candidate.xlsx"))


def test_publish_submission_preserves_existing_file_on_run_error(tmp_path: Path) -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="值是多少？",
        answer_format="number",
    )
    output = tmp_path / "submission.xlsx"
    previous = RunResult(
        question_id=1,
        question=question.question,
        document_id=question.file_name,
        final_answer="7",
        status="success",
    )
    write_submission([previous], output, [question])

    with pytest.raises(SubmissionBuildError) as captured:
        publish_submission(
            [
                RunResult(
                    question_id=1,
                    question=question.question,
                    document_id=question.file_name,
                    status="error",
                    error_type="E11_OUTPUT_PROTOCOL_ERROR",
                )
            ],
            output,
            [question],
        )

    frame = pd.read_excel(output, dtype=object, keep_default_na=False)
    assert frame.loc[0, "answer"] == "7"
    assert captured.value.report["error_derived_empty_ids"] == [1]
    assert not list(tmp_path.glob(".*.candidate.xlsx"))


def test_publish_submission_preserves_existing_file_on_format_error(tmp_path: Path) -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="值是多少？",
        answer_format="number",
    )
    output = tmp_path / "submission.xlsx"
    previous = RunResult(
        question_id=1,
        question=question.question,
        document_id=question.file_name,
        final_answer="7",
        status="success",
    )
    write_submission([previous], output, [question])

    with pytest.raises(SubmissionBuildError) as captured:
        publish_submission(
            [
                RunResult(
                    question_id=1,
                    question=question.question,
                    document_id=question.file_name,
                    final_answer="not-a-number",
                    status="success",
                )
            ],
            output,
            [question],
        )

    frame = pd.read_excel(output, dtype=object, keep_default_na=False)
    assert frame.loc[0, "answer"] == "7"
    assert captured.value.report["format_error_count"] == 1
    assert not list(tmp_path.glob(".*.candidate.xlsx"))


def test_force_complete_results_produces_non_empty_format_safe_answers() -> None:
    formats = ["string", "number", "json_array", "json"]
    questions = [
        QuestionRecord(
            id=index,
            file_name="001.png",
            question_type="structure" if answer_format == "json" else "extract",
            question=f"问题 {index}",
            answer_format=answer_format,  # type: ignore[arg-type]
        )
        for index, answer_format in enumerate(formats, start=1)
    ]
    results = [
        RunResult(
            question_id=question.id,
            question=question.question,
            document_id=question.file_name,
            status="insufficient",
        )
        for question in questions
    ]

    completed, forced_ids = force_complete_results(results, questions)

    assert forced_ids == [1, 2, 3, 4]
    assert [result.final_answer for result in completed] == [
        "未知",
        "0",
        '["未知"]',
        (
            '{"row_count":1,"col_count":1,"cells":'
            '[{"text":"未知","row":0,"col":0,"rowspan":1,"colspan":1}]}'
        ),
    ]
    for result, question in zip(completed, questions, strict=True):
        assert validate_answer_text(result.final_answer, question, allow_blank=False) == []


def test_end_to_end_pipeline_without_external_api(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="thinking",
        question="差额是多少？",
        answer_format="number",
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=FakeEvidenceAgent(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )
    store = JsonlRunStore(tmp_path / "logs" / "run.jsonl")

    results = pipeline.run_batch([question], store=store)

    assert results[0].status == "success"
    assert results[0].final_answer == "30"
    assert store.load_latest()[1].final_answer == "30"


def test_extract_workflow_can_submit_direct_answer_without_operation(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="目标值是多少？",
        answer_format="number",
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=DirectExtractAgent(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "42"
    assert result.evidence is not None and result.evidence.operation is None


def test_extract_agent_can_call_shared_locator_crop_ocr_tool(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (1000, 500), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="请提取目标字段",
        answer_format="number",
    )
    agent = ToolCallingExtractAgent()
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=agent,  # type: ignore[arg-type]
        locator=FakeLocator(),  # type: ignore[arg-type]
        ocr_backend=FakeOCR(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "42"
    assert agent.calls == 2
    assert any("Agent 主动调用 inspect_table_region" in item for item in result.warnings)


def test_list_result_does_not_receive_object_fields_projection(tmp_path: Path) -> None:
    from table_qa_agent.schemas import AnswerProjection, TaskPlan

    plan = TaskPlan(
        task_kind="extract_multi", specialist="extract", mode="multi_field",
        preferred_operation="list",
        answer_projection=AnswerProjection(mode="fields", fields=["a", "b"]),
    )
    question = QuestionRecord(
        id=1, file_name="001.png", question_type="extract",
        question="依次提取 a、b", answer_format="json_array",
    )
    output = AgentOutput(
        evidence=EvidenceResponse(
            question_type="extract",
            evidence=[EvidenceItem(value=10), EvidenceItem(value=20)],
            operation=OperationSpec(name="list", arguments={"values": [
                {"evidence_index": 1}, {"evidence_index": 0},
            ]}),
            answer_projection=plan.answer_projection,
        ), raw_text="{}", usage=TokenUsage(),
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(tmp_path),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=DirectExtractAgent(), runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )
    result = RunResult(question_id=1, question=question.question, document_id="001.png")
    pipeline._execute_agent_output(
        result=result, question=question, plan=plan, agent=pipeline.agent, agent_output=output,
    )
    assert result.final_answer == "[20,10]"
    assert result.recovery_attempts == []


def test_pipeline_preference_accepts_single_step_list(tmp_path: Path) -> None:
    """第 22 题：通用 pipeline 建议不能触发对合法 list 的强制修复。"""
    from table_qa_agent.schemas import AnswerProjection, TaskPlan

    plan = TaskPlan(
        task_kind="compute_boolean", specialist="compute", mode="multi_field",
        preferred_operation="pipeline",
        answer_projection=AnswerProjection(mode="fields", fields=["item"]),
    )
    question = QuestionRecord(
        id=22, file_name="005.pdf", question_type="thinking",
        question="那些项，今年有增减变动？", answer_format="json_array",
    )
    output = AgentOutput(
        evidence=EvidenceResponse(
            question_type="thinking",
            evidence=[EvidenceItem(value=10, row_header="利润分配")],
            operation=OperationSpec(name="list", arguments={"values": [
                {"evidence_index": 0, "field": "row_header"},
            ]}),
            answer_projection=AnswerProjection(mode="identity"),
        ), raw_text="{}", usage=TokenUsage(),
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(tmp_path),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=DirectExtractAgent(), runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )
    result = RunResult(question_id=22, question=question.question, document_id="005.pdf")
    pipeline._execute_agent_output(
        result=result, question=question, plan=plan, agent=pipeline.agent, agent_output=output,
    )
    assert result.final_answer == '["利润分配"]'
    assert result.recovery_attempts == []


def test_soft_evidence_risk_is_recorded_without_overwriting_answer(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (1000, 500), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="请提取目标字段",
        answer_format="number",
    )
    agent = SoftRiskRepairAgent()
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=agent,  # type: ignore[arg-type]
        locator=FakeLocator(),  # type: ignore[arg-type]
        ocr_backend=FakeOCR(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
        validate_evidence=False,
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "41"
    assert agent.calls == 1
    assert result.recovery_attempts == []
    assert any("Evidence 软检查" in item for item in result.warnings)
    assert any("保留原答案" in item for item in result.warnings)


def test_soft_evidence_risk_without_locator_keeps_original_answer(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="请提取目标字段",
        answer_format="number",
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=SoftRiskRepairAgent(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
        validate_evidence=False,
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "41"
    assert any("保留原答案" in item for item in result.warnings)


def test_experimental_pipeline_skips_operation_grounding(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="值是多少？",
        answer_format="number",
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=LiteralAnswerAgent(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
        validate_evidence=False,
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "42"


def test_format_only_pipeline_recovers_empty_string_as_non_empty_answer(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="请提取不存在的字段",
        answer_format="string",
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=EmptyAnswerAgent(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
        validate_evidence=False,
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "未知"
    assert result.error_type is None
    assert any("format-only 本地恢复" in warning for warning in result.warnings)


def test_resume_reruns_old_success_that_violates_current_contract(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="structure",
        question="恢复前一行和第一列",
        answer_format="json",
    )
    store = JsonlRunStore(tmp_path / "logs" / "run.jsonl")
    store.append(
        RunResult(
            question_id=1,
            question=question.question,
            document_id=question.file_name,
            final_answer='["项目","金额"]',
            status="success",
        )
    )
    agent = StructureFakeEvidenceAgent()
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=agent,  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )

    result = pipeline.run_batch([question], store=store, resume=True)[0]

    assert agent.calls == 1
    assert result.status == "success"
    assert result.final_answer is not None
    assert '"row_count":2' in result.final_answer


def test_json_array_scalar_is_wrapped_and_warned(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="thinking",
        question="差额是多少？",
        answer_format="json_array",
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=FakeEvidenceAgent(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )

    result = pipeline.run_one(question)

    assert result.final_answer == "[30]"
    assert result.warnings == ["题目要求 json_array，但工具返回标量；已按单元素数组包装"]


def test_run_batch_uses_eight_workers_and_keeps_order(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    questions = [
        QuestionRecord(
            id=index,
            file_name="001.png",
            question_type="extract",
            question=f"问题 {index}",
            answer_format="number",
        )
        for index in range(1, 17)
    ]
    agent = CountingFakeEvidenceAgent()
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=agent,  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs", max_workers=8),
    )
    store = JsonlRunStore(tmp_path / "logs" / "concurrent.jsonl")

    results = pipeline.run_batch(questions, store=store, resume=False)

    assert agent.max_active == 8
    assert [result.question_id for result in results] == list(range(1, 17))
    assert len(store.load_latest()) == 16


def test_pipeline_recovers_insufficient_with_locator_crop_and_ocr(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (1000, 500), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="请提取目标字段",
        answer_format="number",
    )
    agent = RecoveringFakeAgent()
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=agent,  # type: ignore[arg-type]
        locator=FakeLocator(),  # type: ignore[arg-type]
        ocr_backend=FakeOCR(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "42"
    assert agent.calls == 2
    assert result.ocr_text == "目标字段 42"
    assert result.regions[0].page == 1
    assert result.recovery_attempts[0].action == "locate_crop_ocr"
    assert result.recovery_attempts[0].succeeded is True
    assert result.token_usage.total_tokens == 4


def test_pipeline_repairs_only_operation_when_evidence_is_available(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    Image.new("RGB", (200, 100), "white").save(files_dir / "001.png")
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="thinking",
        question="两项之和是多少？",
        answer_format="number",
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        agent=RepairingOperationAgent(),  # type: ignore[arg-type]
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )

    result = pipeline.run_one(question)

    assert result.status == "success"
    assert result.final_answer == "3"
    assert result.recovery_attempts[0].action == "repair_operation"
    assert result.recovery_attempts[0].succeeded is True
    assert result.token_usage.total_tokens == 2
