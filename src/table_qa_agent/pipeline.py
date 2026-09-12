"""分层专家批处理 Pipeline 与可恢复运行日志。"""

from __future__ import annotations

import inspect
import json
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from loguru import logger
from tqdm import tqdm

from table_qa_agent.agent import (
    AgentOutput,
    EvidenceAgent,
    EvidenceValidationError,
    parse_evidence_response,
)
from table_qa_agent.answer import AnswerProjectionError, project_answer
from table_qa_agent.capabilities import validate_tool_call
from table_qa_agent.config import (
    OCRConfig,
    RecoveryConfig,
    RetrievalConfig,
    RuntimeConfig,
)
from table_qa_agent.dataset import DocumentNotFoundError, DocumentResolver
from table_qa_agent.documents import DocumentProcessingError, DocumentProcessor
from table_qa_agent.executor import (
    ArgumentGroundingError,
    OperationExecutionError,
    can_safely_replay_count,
    execute_operation,
    operation_selects_answer,
    validate_operation_grounding,
)
from table_qa_agent.normalizer import (
    AnswerNormalizationError,
    normalize_answer,
    validate_answer_text,
)
from table_qa_agent.ocr.base import OCRBackend
from table_qa_agent.orchestration import (
    AgentRouter,
    FunctionCallingIntentPlanner,
    PlanningResult,
    TaskPlanner,
)
from table_qa_agent.retrieval import PageRegionLocator
from table_qa_agent.schemas import (
    AnswerFormat,
    QuestionRecord,
    RecoveryAttempt,
    RunResult,
    TableStructureAnswer,
    TaskPlan,
    TokenUsage,
)
from table_qa_agent.structure import StructureRepairError, repair_structure
from table_qa_agent.structure.contract import structure_recovery_content
from table_qa_agent.structure.patch import apply_structure_patch
from table_qa_agent.verification import assess_evidence

# xlsx 中零长度文本常被读取器还原成 null/NaN。按 answer_format 写入合法的
# 显式占位值，既避免单元格缺失，也能通过 JSON 类型校验。
EMPTY_ANSWER_BY_FORMAT: dict[AnswerFormat, str] = {
    "string": '""',
    "number": '""',
    "json_array": "[]",
    "json": '{"row_count":0,"col_count":0,"cells":[]}',
}

# format-only 策略的最终兜底。它们保证类型协议与非空性，不代表有视觉证据支持。
FORCED_ANSWER_BY_FORMAT: dict[AnswerFormat, str] = {
    "string": "未知",
    "number": "0",
    "json_array": '["未知"]',
    "json": (
        '{"row_count":1,"col_count":1,"cells":'
        '[{"text":"未知","row":0,"col":0,"rowspan":1,"colspan":1}]}'
    ),
}


class JsonlRunStore:
    """逐题追加日志，进程中断后仍可恢复已完成结果。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def append(self, result: RunResult) -> None:
        payload = result.model_dump(mode="json")
        with self._write_lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()

    def load_latest(self) -> dict[int, RunResult]:
        results: dict[int, RunResult] = {}
        for result in self.load_all():
            results[result.question_id] = result
        return results

    def load_all(self) -> list[RunResult]:
        """按日志顺序读取全部记录，供历史成功答案回退和审计。"""

        if not self.path.exists():
            return []
        ordered: list[RunResult] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    result = RunResult.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(
                        f"日志第 {line_number} 行不是合法 RunResult: {self.path}"
                    ) from exc
                ordered.append(result)
        return ordered


class SubmissionBuildError(ValueError):
    """提交产物未通过发布前门禁。"""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class BaselinePipeline:
    """Question + Document -> Evidence -> Tool -> Normalized Answer。"""

    def __init__(
        self,
        *,
        resolver: DocumentResolver,
        processor: DocumentProcessor,
        agent: EvidenceAgent | None = None,
        planner: Any | None = None,
        router: AgentRouter | None = None,
        locator: PageRegionLocator | None = None,
        ocr_backend: OCRBackend | None = None,
        runtime: RuntimeConfig,
        retrieval: RetrievalConfig | None = None,
        ocr: OCRConfig | None = None,
        recovery: RecoveryConfig | None = None,
        validate_evidence: bool = True,
    ) -> None:
        if agent is None and router is None:
            raise ValueError("必须提供默认 agent 或专家 router")
        self.resolver = resolver
        self.processor = processor
        self.agent = agent
        self.planner = planner or TaskPlanner()
        self.router = router
        self.locator = locator
        self.ocr_backend = ocr_backend
        self.runtime = runtime
        self.retrieval = retrieval or RetrievalConfig()
        self.ocr = ocr or OCRConfig()
        self.recovery = recovery or RecoveryConfig()
        self.validate_evidence = validate_evidence
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0

    @staticmethod
    def _add_usage(result: RunResult, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        current = result.token_usage
        result.token_usage = TokenUsage(
            prompt_tokens=(current.prompt_tokens or 0) + (usage.prompt_tokens or 0),
            completion_tokens=(current.completion_tokens or 0) + (usage.completion_tokens or 0),
            total_tokens=(current.total_tokens or 0) + (usage.total_tokens or 0),
        )

    @staticmethod
    def _call_agent(
        agent: Any,
        question: QuestionRecord,
        image_content: list[dict[str, object]],
        *,
        plan: TaskPlan,
        ocr_text: str | None = None,
        recovery_context: str | None = None,
    ) -> AgentOutput:
        method = agent.run
        signature = inspect.signature(method)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        optional = {
            "plan": plan,
            "ocr_text": ocr_text,
            "recovery_context": recovery_context,
        }
        kwargs = {
            key: value
            for key, value in optional.items()
            if accepts_kwargs or key in signature.parameters
        }
        return method(question, image_content, **kwargs)

    def _selected_agent(self, plan: TaskPlan) -> Any:
        if self.router is not None:
            return self.router.route(plan)
        return self.agent

    def _run_requested_agent_tool(
        self,
        *,
        result: RunResult,
        question: QuestionRecord,
        plan: TaskPlan,
        agent: Any,
        agent_output: AgentOutput,
        document_path: Path,
        full_content: list[dict[str, object]],
    ) -> AgentOutput | None:
        """执行专业 Agent 主动请求的共享能力；当前只开放表格区域检查。"""

        if not agent_output.tool_calls:
            return agent_output
        if self.recovery.max_agent_tool_calls < 1:
            result.warnings.append("Agent 请求了工具，但配置已禁用主动工具调用")
            return None
        if len(agent_output.tool_calls) > self.recovery.max_agent_tool_calls:
            result.warnings.append("Agent 工具调用超过单题预算，已只执行第一个请求")
        call = agent_output.tool_calls[0]
        arguments = validate_tool_call(call)
        result.warnings.append(
            f"Agent 主动调用 {call.name}: {arguments.reason}"
        )
        return self._visual_recovery(
            result=result,
            question=question,
            plan=plan,
            agent=agent,
            document_path=document_path,
            full_content=full_content,
            failure_reason=(
                f"Agent 主动请求读取表格；目标：{arguments.query}；原因：{arguments.reason}"
            ),
        )

    def _record_soft_evidence_risks(
        self,
        *,
        result: RunResult,
        plan: TaskPlan,
        agent_output: AgentOutput,
    ) -> None:
        """只记录 Evidence 风险，不再生成第二候选或覆盖原答案。"""

        if self.validate_evidence or not self.recovery.enabled:
            return
        risks = assess_evidence(result.evidence or agent_output.evidence, plan)
        if not risks:
            return
        risk_summary = "；".join(f"{risk.code}: {risk.message}" for risk in risks)
        result.warnings.append(f"Evidence 软检查：{risk_summary}")
        result.warnings.append("Evidence 风险仅记录；保留原答案")

    def _visual_recovery(
        self,
        *,
        result: RunResult,
        question: QuestionRecord,
        plan: TaskPlan,
        agent: Any,
        document_path: Path,
        full_content: list[dict[str, object]],
        failure_reason: str,
    ) -> AgentOutput | None:
        attempt = RecoveryAttempt(
            action="locate_crop_ocr",
            reason=failure_reason,
        )
        result.recovery_attempts.append(attempt)
        if (
            not self.recovery.enabled
            or self.recovery.max_visual_retries < 1
            or not self.retrieval.enabled
            or self.locator is None
        ):
            result.warnings.append("未配置区域定位器，跳过视觉恢复")
            return None
        try:
            self._wait_for_request_slot()
            regions, locator_usage = self.locator.locate(
                question,
                plan,
                full_content,
                failure_reason=failure_reason,
            )
            self._add_usage(result, locator_usage)
            region_paths = self.processor.prepare_regions(
                document_path,
                regions,
                roi_dpi=self.retrieval.roi_dpi,
                padding_ratio=self.retrieval.padding_ratio,
            )
            region_content = self.processor.as_region_content(region_paths, regions)
            result.regions = regions
            ocr_text = None
            if self.ocr.enabled and self.ocr_backend is not None:
                try:
                    self._wait_for_request_slot()
                    ocr_result, ocr_usage = self.ocr_backend.recognize(
                        question, region_content
                    )
                    self._add_usage(result, ocr_usage)
                    ocr_text = ocr_result.as_prompt_text(self.ocr.max_prompt_chars)
                    result.ocr_text = ocr_text or None
                except Exception as exc:
                    result.warnings.append(f"OCR 候选生成失败，继续使用高清区域: {exc}")
            self._wait_for_request_slot()
            specialist_content = (
                structure_recovery_content(full_content, region_content)
                if plan.specialist == "structure" else region_content
            )
            recovered = self._call_agent(
                agent,
                question,
                specialist_content,
                plan=plan,
                ocr_text=ocr_text,
                recovery_context=failure_reason,
            )
            attempt.succeeded = recovered.evidence.status == "success"
            return recovered
        except Exception as exc:
            result.warnings.append(f"视觉恢复失败: {exc}")
            return None

    def _execute_agent_output(
        self,
        *,
        result: RunResult,
        question: QuestionRecord,
        plan: TaskPlan,
        agent: Any,
        agent_output: AgentOutput,
        structure_images: list[dict[str, object]] | None = None,
    ) -> None:
        result.warnings.extend(agent_output.warnings)
        if agent_output.evidence.direct_answer is not None:
            if plan.specialist not in {"extract", "visual_attribute"}:
                raise AnswerNormalizationError(
                    f"{plan.specialist}/{plan.mode} 工作流不能绕过确定性 Operation"
                )
            result.tool_result = agent_output.evidence.direct_answer
            result.final_answer = normalize_answer(
                agent_output.evidence.direct_answer,
                question.answer_format,
                {},
            )
            contract_errors = validate_answer_text(
                result.final_answer,
                question,
                allow_blank=False,
            )
            if contract_errors:
                raise AnswerNormalizationError("；".join(contract_errors))
            return
        operation = agent_output.evidence.operation
        if operation is None:
            raise EvidenceValidationError("operation 不能为空")
        evidence_items = agent_output.evidence.evidence
        try:
            if self.validate_evidence:
                validate_operation_grounding(operation, evidence_items)
            tool_result = execute_operation(operation, evidence_items)
        except (ArgumentGroundingError, OperationExecutionError) as exc:
            repair_method = getattr(agent, "repair_operation", None)
            if not self.recovery.enabled or self.recovery.max_operation_repairs < 1 or not callable(
                repair_method
            ):
                raise
            attempt = RecoveryAttempt(action="repair_operation", reason=str(exc))
            result.recovery_attempts.append(attempt)
            self._wait_for_request_slot()
            repaired = repair_method(question, plan, agent_output.evidence, exc)
            self._add_usage(result, repaired.usage)
            result.evidence = repaired.evidence
            operation = repaired.evidence.operation
            if operation is None:
                raise EvidenceValidationError("修复后的 operation 为空") from exc
            if self.validate_evidence:
                validate_operation_grounding(operation, evidence_items)
            tool_result = execute_operation(operation, evidence_items)
            agent_output = repaired
            attempt.succeeded = True

        if question.answer_format == "json" and self.recovery.repair_structure_dimensions:
            try:
                TableStructureAnswer.model_validate(tool_result)
            except Exception:
                attempt = RecoveryAttempt(
                    action="repair_structure",
                    reason="结构几何协议校验失败",
                )
                result.recovery_attempts.append(attempt)
                try:
                    tool_result = repair_structure(tool_result)
                    attempt.succeeded = True
                except StructureRepairError as exc:
                    attempt.reason = str(exc)
                    propose = getattr(agent, "propose_structure_patch", None)
                    if (
                        not self.recovery.enabled or not self.recovery.max_structure_patches
                        or not structure_images or not callable(propose)
                        or any(a.action == "patch_structure" for a in result.recovery_attempts)
                    ):
                        raise
                    patch_attempt = RecoveryAttempt(action="patch_structure", reason=str(exc))
                    result.recovery_attempts.append(patch_attempt)
                    self._wait_for_request_slot()
                    try:
                        patch_images = structure_images
                        if result.regions and result.document_path:
                            paths = self.processor.prepare_regions(
                                result.document_path, result.regions,
                                roi_dpi=self.retrieval.roi_dpi,
                                padding_ratio=self.retrieval.padding_ratio,
                            )
                            patch_images = structure_recovery_content(
                                structure_images,
                                self.processor.as_region_content(paths, result.regions),
                            )
                        completion = propose(question, tool_result, patch_images, exc)
                        self._add_usage(result, completion.usage)
                        patch_attempt.raw_output = completion.text
                        tool_result = apply_structure_patch(
                            tool_result, json.loads(completion.text),
                        )
                        patch_attempt.succeeded = True
                    except Exception as patch_error:
                        patch_attempt.reason += f"；受限修复失败：{patch_error}"
                        raise StructureRepairError(patch_attempt.reason) from patch_error

        result.tool_result = tool_result
        projection = _effective_projection(
            agent_output.evidence.answer_projection,
            plan.answer_projection,
        )
        if operation_selects_answer(operation):
            projection = None
        # list 已按 arguments.values 顺序组装完毕，不能再次按对象键投影。
        if operation.name == "list" and projection is not None and projection.mode == "fields":
            projection = None
            result.warnings.append("list 已生成有序数组，跳过重复 fields 投影")
        try:
            projected_result = project_answer(tool_result, projection)
        except AnswerProjectionError:
            if projection == plan.answer_projection:
                raise
            projected_result = project_answer(tool_result, plan.answer_projection)
            result.warnings.append("模型投影与工具结果不兼容，已回退 TaskPlan 投影")
        if question.answer_format == "json_array" and not isinstance(
            projected_result, (list, tuple)
        ):
            result.warnings.append("题目要求 json_array，但工具返回标量；已按单元素数组包装")
        result.final_answer = normalize_answer(
            projected_result,
            question.answer_format,
            agent_output.evidence.output,
        )
        contract_errors = validate_answer_text(
            result.final_answer,
            question,
            allow_blank=False,
        )
        if contract_errors:
            raise AnswerNormalizationError("；".join(contract_errors))

    def _wait_for_request_slot(self) -> None:
        interval = self.runtime.request_interval_seconds
        if interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request_at - now)
            if delay:
                time.sleep(delay)
            self._next_request_at = time.monotonic() + interval

    def run_one(self, question: QuestionRecord) -> RunResult:
        started = time.perf_counter()
        result = RunResult(
            question_id=question.id,
            question=question.question,
            document_id=question.file_name,
        )
        try:
            if isinstance(self.planner, FunctionCallingIntentPlanner):
                self._wait_for_request_slot()
            planning = self.planner.plan(question)
            if isinstance(planning, PlanningResult):
                plan = planning.plan
                result.planning_raw_output = planning.raw_output
                result.warnings.extend(planning.warnings)
                self._add_usage(result, planning.usage)
            else:
                plan = planning
            result.plan = plan
            result.specialist = plan.specialist
            agent = self._selected_agent(plan)
            document_path = self.resolver.resolve(question.file_name)
            pages = self.processor.prepare(document_path)
            result.document_path = str(document_path)
            result.pages = [str(page) for page in pages]

            self._wait_for_request_slot()
            full_content = self.processor.as_openai_content(pages)
            try:
                agent_output = self._call_agent(
                    agent,
                    question,
                    full_content,
                    plan=plan,
                )
            except EvidenceValidationError as exc:
                result.raw_model_output = exc.raw_text
                self._add_usage(result, exc.usage)
                recovered = self._visual_recovery(
                    result=result,
                    question=question,
                    plan=plan,
                    agent=agent,
                    document_path=document_path,
                    full_content=full_content,
                    failure_reason=str(exc),
                )
                if recovered is None:
                    raise
                agent_output = recovered
            if result.raw_model_output is None:
                result.raw_model_output = agent_output.raw_text
            self._add_usage(result, agent_output.usage)
            result.evidence = agent_output.evidence

            if agent_output.tool_calls:
                recovered = self._run_requested_agent_tool(
                    result=result,
                    question=question,
                    plan=plan,
                    agent=agent,
                    agent_output=agent_output,
                    document_path=document_path,
                    full_content=full_content,
                )
                if recovered is None:
                    self._wait_for_request_slot()
                    recovered = self._call_agent(
                        agent,
                        question,
                        full_content,
                        plan=plan,
                        recovery_context=(
                            "共享工具未能返回结果。本轮不能再次调用工具，请基于当前页面给出"
                            "最佳非空答案。"
                        ),
                    )
                if recovered.tool_calls:
                    raise EvidenceValidationError("达到工具预算后仍重复请求工具")
                agent_output = recovered
                self._add_usage(result, recovered.usage)
                result.evidence = recovered.evidence

            if agent_output.evidence.status != "success":
                recovered = None
                if not any(
                    item.action == "locate_crop_ocr" for item in result.recovery_attempts
                ):
                    recovered = self._visual_recovery(
                        result=result,
                        question=question,
                        plan=plan,
                        agent=agent,
                        document_path=document_path,
                        full_content=full_content,
                        failure_reason=(
                            agent_output.evidence.reason or agent_output.evidence.status
                        ),
                    )
                if recovered is None or recovered.evidence.status != "success":
                    result.status = agent_output.evidence.status
                    result.error_type = "E12_AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE"
                    result.error_message = agent_output.evidence.reason
                    return result
                agent_output = recovered
                self._add_usage(result, recovered.usage)
                result.evidence = recovered.evidence

            try:
                self._execute_agent_output(
                    result=result,
                    question=question,
                    plan=plan,
                    agent=agent,
                    agent_output=agent_output,
                    structure_images=full_content,
                )
            except (
                AnswerNormalizationError,
                AnswerProjectionError,
                ArgumentGroundingError,
                OperationExecutionError,
                StructureRepairError,
            ) as exc:
                if self.recovery.max_visual_retries < 1 or any(
                    item.action in {"locate_crop_ocr", "patch_structure"}
                    for item in result.recovery_attempts
                ):
                    raise
                recovered = self._visual_recovery(
                    result=result,
                    question=question,
                    plan=plan,
                    agent=agent,
                    document_path=document_path,
                    full_content=full_content,
                    failure_reason=str(exc),
                )
                if recovered is None or recovered.evidence.status != "success":
                    raise
                self._add_usage(result, recovered.usage)
                result.evidence = recovered.evidence
                self._execute_agent_output(
                    result=result,
                    question=question,
                    plan=plan,
                    agent=agent,
                    agent_output=recovered,
                    structure_images=full_content,
                )
            self._record_soft_evidence_risks(
                result=result,
                plan=plan,
                agent_output=agent_output,
            )
            result.status = "success"
            return result
        except EvidenceValidationError as exc:
            result.raw_model_output = exc.raw_text
            if exc.usage is not None:
                result.token_usage = exc.usage
            result.status = "error"
            result.error_type = "E11_OUTPUT_PROTOCOL_ERROR"
            result.error_message = str(exc)
            logger.error("题目 {} Evidence 协议失败: {}", question.id, exc)
            logger.opt(exception=exc).debug("题目 {} 详细堆栈", question.id)
            if not self.runtime.continue_on_error:
                raise
            return result
        except Exception as exc:
            if (
                not self.validate_evidence
                and result.specialist in {"extract", "visual_attribute"}
                and isinstance(
                    exc,
                    (
                        AnswerNormalizationError,
                        AnswerProjectionError,
                        ArgumentGroundingError,
                        OperationExecutionError,
                        StructureRepairError,
                    ),
                )
            ):
                result.final_answer = _recover_format_only_answer(
                    question,
                    result.evidence.evidence if result.evidence is not None else [],
                )
                result.status = "success"
                result.error_type = None
                result.error_message = None
                result.warnings.append(
                    f"format-only 本地恢复：{type(exc).__name__}: {exc}"
                )
                logger.warning(
                    "题目 {} 执行异常已由 format-only 本地恢复: {}",
                    question.id,
                    exc,
                )
                return result
            result.status = "error"
            result.error_type = _classify_error(exc)
            result.error_message = str(exc)
            logger.error("题目 {} 执行失败 [{}]: {}", question.id, result.error_type, exc)
            logger.opt(exception=exc).debug("题目 {} 详细堆栈", question.id)
            if not self.runtime.continue_on_error:
                raise
            return result
        finally:
            result.latency_seconds = round(time.perf_counter() - started, 4)

    def run_batch(
        self,
        questions: Iterable[QuestionRecord],
        *,
        store: JsonlRunStore,
        resume: bool = True,
    ) -> list[RunResult]:
        selected = list(questions)
        history = store.load_all() if resume else []
        existing: dict[int, RunResult] = {}
        for historical in history:
            existing[historical.question_id] = historical
        historical_success: dict[int, RunResult] = {}
        question_by_id = {question.id: question for question in selected}
        for historical in history:
            question = question_by_id.get(historical.question_id)
            if question is not None and _is_usable_success(historical, question):
                historical_success[historical.question_id] = historical
        completed: dict[int, RunResult] = {}
        for question in selected:
            result = historical_success.get(question.id)
            if result is not None:
                completed[question.id] = result
                continue
            latest = existing.get(question.id)
            if latest is not None and latest.status == "success":
                logger.info("题目 {} 的旧答案不符合当前提交协议，将重新执行", question.id)

        results_by_id: dict[int, RunResult] = {}
        pending: list[QuestionRecord] = []
        for question in selected:
            if question.id in completed:
                results_by_id[question.id] = completed[question.id]
            else:
                pending.append(question)

        with tqdm(total=len(selected), desc="Table QA", unit="题") as progress:
            progress.update(len(results_by_id))
            if self.runtime.max_workers == 1:
                for question in pending:
                    result = self.run_one(question)
                    store.append(result)
                    results_by_id[question.id] = result
                    progress.update(1)
            else:
                with ThreadPoolExecutor(
                    max_workers=self.runtime.max_workers,
                    thread_name_prefix="table-qa",
                ) as executor:
                    future_to_question = {
                        executor.submit(self.run_one, question): question for question in pending
                    }
                    for future in as_completed(future_to_question):
                        question = future_to_question[future]
                        result = future.result()
                        store.append(result)
                        results_by_id[question.id] = result
                        progress.update(1)

        return [results_by_id[question.id] for question in selected]


def _classify_error(exc: Exception) -> str:
    """把程序异常映射到 Agent.md 的错误分类。"""

    if isinstance(exc, DocumentNotFoundError):
        return "E1_DOCUMENT_RETRIEVAL_ERROR"
    if isinstance(exc, DocumentProcessingError):
        return "E2_PAGE_RETRIEVAL_ERROR"
    if isinstance(exc, ArgumentGroundingError):
        return "E9_ARGUMENT_BINDING_ERROR"
    if isinstance(exc, OperationExecutionError):
        return "E10_TOOL_ERROR"
    if isinstance(exc, AnswerNormalizationError):
        return "E11_OUTPUT_NORMALIZATION_ERROR"
    if isinstance(exc, (AnswerProjectionError, StructureRepairError)):
        return "E11_OUTPUT_NORMALIZATION_ERROR"
    return f"SYSTEM_ERROR:{type(exc).__name__}"


def _recover_format_only_answer(
    question: QuestionRecord,
    evidence: Iterable[Any],
) -> str:
    """从已有 Evidence 恢复格式合法答案；无可用候选时使用类型安全非空兜底。"""

    evidence_items = list(evidence)
    if question.answer_format == "json":
        for item in evidence_items:
            try:
                return normalize_answer(item.value, "json")
            except AnswerNormalizationError:
                continue
    elif question.answer_format == "json_array":
        values = [
            item.value
            for item in evidence_items
            if item.value is not None
            and not (isinstance(item.value, str) and not item.value.strip())
        ]
        if values:
            candidate = (
                values[0]
                if len(values) == 1 and isinstance(values[0], list)
                else values
            )
            return normalize_answer(candidate, "json_array")
    elif question.answer_format == "number":
        for item in evidence_items:
            for candidate in (item.value, item.value_raw):
                try:
                    return normalize_answer(candidate, "number")
                except AnswerNormalizationError:
                    continue
    else:
        for item in evidence_items:
            for candidate in (
                item.entity,
                item.column_header,
                item.row_header,
                item.value,
                item.value_raw,
                item.source_text,
            ):
                if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
                    continue
                return normalize_answer(candidate, "string")
    return FORCED_ANSWER_BY_FORMAT[question.answer_format]


def _effective_projection(
    evidence_projection: Any,
    plan_projection: Any,
) -> Any:
    """计划中的明确投影优先于模型无意写出的默认 identity。"""

    if evidence_projection is None:
        return plan_projection
    if evidence_projection.mode == "identity" and plan_projection.mode != "identity":
        return plan_projection
    return evidence_projection


def _is_usable_success(result: RunResult, question: QuestionRecord) -> bool:
    return result.status == "success" and not validate_answer_text(
        result.final_answer,
        question,
        allow_blank=False,
    )


def replay_trace_result(
    result: RunResult,
    question: QuestionRecord,
    *,
    replay_success_operation: bool = False,
) -> RunResult:
    """用当前协议重放 Evidence；默认只恢复失败，可显式重算成功 Operation。"""

    if _is_usable_success(result, question) and not replay_success_operation:
        return result
    evidence = result.evidence
    if evidence is None and result.raw_model_output:
        try:
            evidence = parse_evidence_response(result.raw_model_output)
        except Exception:
            return result
    if evidence is None or evidence.status != "success" or evidence.operation is None:
        return result
    try:
        validate_operation_grounding(evidence.operation, evidence.evidence)
        tool_result = execute_operation(evidence.operation, evidence.evidence)
        if question.answer_format == "json":
            try:
                TableStructureAnswer.model_validate(tool_result)
            except Exception:
                tool_result = repair_structure(tool_result)
        # 主动重算成功答案时沿用当时计划，避免把路由/投影变化混入确定性执行器实验。
        # 失败日志可能缺计划，才使用当前规则 Planner 恢复。
        plan = (
            result.plan
            if replay_success_operation and result.plan is not None
            else TaskPlanner().plan(question)
        )
        projection = _effective_projection(evidence.answer_projection, plan.answer_projection)
        if operation_selects_answer(evidence.operation):
            projection = None
        try:
            projected_result = project_answer(tool_result, projection)
        except AnswerProjectionError:
            projected_result = project_answer(tool_result, plan.answer_projection)
        final_answer = normalize_answer(projected_result, question.answer_format, evidence.output)
        errors = validate_answer_text(final_answer, question, allow_blank=False)
        if errors:
            return result
    except Exception:
        visible_values = [item.value for item in evidence.evidence]
        if visible_values and all(
            value is None or (isinstance(value, str) and value.strip() in {"", "—", "-"})
            for value in visible_values
        ):
            return result.model_copy(
                update={
                    "status": "insufficient",
                    "error_type": "E12_AMBIGUOUS_OR_INSUFFICIENT_EVIDENCE",
                    "error_message": "目标单元格没有可提交的数值",
                    "warnings": [*result.warnings, "历史 Evidence 显示目标值为空"],
                }
            )
        return result
    replay_message = (
        "使用当前执行协议重算历史成功 Operation"
        if replay_success_operation and result.status == "success"
        else "使用当前执行协议从历史 Evidence 重放恢复"
    )
    warnings = [*result.warnings, replay_message]
    return result.model_copy(
        update={
            "evidence": evidence,
            "tool_result": tool_result,
            "final_answer": final_answer,
            "status": "success",
            "error_type": None,
            "error_message": None,
            "warnings": warnings,
        }
    )


def select_submission_results(
    questions: Iterable[QuestionRecord],
    result_sources: Iterable[Iterable[RunResult]],
    *,
    replay_success_counts: bool = False,
) -> list[RunResult]:
    """按来源优先级选择最近的合法成功结果，避免新一轮失败覆盖旧答案。"""

    question_list = list(questions)
    question_by_id = {question.id: question for question in question_list}
    selected: dict[int, RunResult] = {}
    primary_latest: dict[int, RunResult] = {}
    for source_index, source in enumerate(result_sources):
        usable_in_source: dict[int, RunResult] = {}
        for original in source:
            question = question_by_id.get(original.question_id)
            if question is None:
                continue
            replay_count = bool(
                replay_success_counts
                and original.evidence is not None
                and original.evidence.operation is not None
                and can_safely_replay_count(
                    original.evidence.operation,
                    original.evidence.evidence,
                )
            )
            candidate = replay_trace_result(
                original,
                question,
                replay_success_operation=replay_count,
            )
            if source_index == 0:
                primary_latest[original.question_id] = candidate
            if _is_usable_success(candidate, question):
                usable_in_source[original.question_id] = candidate
        for question_id, candidate in usable_in_source.items():
            if question_id not in selected:
                selected[question_id] = candidate

    merged: list[RunResult] = []
    for question in question_list:
        result = selected.get(question.id) or primary_latest.get(question.id)
        if result is None:
            result = RunResult(
                question_id=question.id,
                question=question.question,
                document_id=question.file_name,
                status="error",
                error_type="E13_MISSING_RUN_RESULT",
                error_message="日志中没有该题运行记录",
            )
        merged.append(result)
    return merged


def find_error_derived_empty_ids(
    results: Iterable[RunResult],
    questions: Iterable[QuestionRecord],
) -> list[int]:
    """只标记执行/格式错误造成的空答案，不误伤证据确实不足的题。"""

    question_by_id = {question.id: question for question in questions}
    ids: list[int] = []
    for result in results:
        question = question_by_id.get(result.question_id)
        if question is None or result.status != "error":
            continue
        if validate_answer_text(result.final_answer, question, allow_blank=False):
            ids.append(result.question_id)
    return sorted(ids)


def force_complete_results(
    results: Iterable[RunResult],
    questions: Iterable[QuestionRecord],
) -> tuple[list[RunResult], list[int]]:
    """format-only 策略下把缺失/非法答案替换为类型安全的非空兜底。"""

    question_by_id = {question.id: question for question in questions}
    completed: list[RunResult] = []
    forced_ids: list[int] = []
    for result in results:
        question = question_by_id.get(result.question_id)
        if question is None or not validate_answer_text(
            result.final_answer,
            question,
            allow_blank=False,
        ):
            completed.append(result)
            continue
        forced_ids.append(result.question_id)
        fallback = FORCED_ANSWER_BY_FORMAT[question.answer_format]
        completed.append(
            result.model_copy(
                update={
                    "final_answer": fallback,
                    "warnings": [
                        *result.warnings,
                        "format-only 策略：缺失或非法答案已替换为格式安全非空兜底",
                    ],
                }
            )
        )
    return completed, sorted(forced_ids)


def _submission_answer(
    result: RunResult,
    question_by_id: dict[int, QuestionRecord],
) -> str:
    if result.final_answer is not None and result.final_answer.strip():
        return result.final_answer
    question = question_by_id.get(result.question_id)
    answer_format: AnswerFormat = question.answer_format if question else "string"
    return EMPTY_ANSWER_BY_FORMAT[answer_format]


def write_submission(
    results: Iterable[RunResult],
    path: Path | str,
    questions: Iterable[QuestionRecord] | None = None,
) -> Path:
    """输出与 submit-template.xlsx 一致的 id/answer 两列工作簿。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    question_by_id = {question.id: question for question in questions or []}
    rows = [
        {"id": result.question_id, "answer": _submission_answer(result, question_by_id)}
        for result in results
    ]
    frame = pd.DataFrame(rows, columns=["id", "answer"]).sort_values("id")
    frame.to_excel(output_path, index=False)
    return output_path


def publish_submission(
    results: Iterable[RunResult],
    path: Path | str,
    questions: Iterable[QuestionRecord],
) -> dict[str, Any]:
    """在临时文件中生成并校验，通过后才原子替换正式 submission。"""

    result_list = list(results)
    question_list = list(questions)
    output_path = Path(path)
    if output_path.suffix.lower() != ".xlsx":
        raise SubmissionBuildError(
            "提交文件必须使用 .xlsx 扩展名，拒绝发布",
            {
                "is_valid": False,
                "validation_error": "输出路径扩展名不是 .xlsx",
                "error_derived_empty_count": 0,
                "error_derived_empty_ids": [],
            },
        )

    error_empty_ids = find_error_derived_empty_ids(result_list, question_list)
    if error_empty_ids:
        report = {
            "is_valid": False,
            "error_derived_empty_count": len(error_empty_ids),
            "error_derived_empty_ids": error_empty_ids,
        }
        raise SubmissionBuildError("仍有运行错误导致的空答案，拒绝发布", report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.candidate{output_path.suffix}"
    )
    try:
        write_submission(result_list, candidate_path, question_list)
        try:
            report = validate_submission(candidate_path, question_list)
        except (OSError, ValueError) as exc:
            raise SubmissionBuildError(
                "候选提交文件未通过结构校验，拒绝发布",
                {
                    "is_valid": False,
                    "validation_error": str(exc),
                    "error_derived_empty_count": 0,
                    "error_derived_empty_ids": [],
                },
            ) from exc
        report["error_derived_empty_count"] = 0
        report["error_derived_empty_ids"] = []
        if not report["is_valid"]:
            raise SubmissionBuildError("候选提交文件未通过格式校验，拒绝发布", report)
        candidate_path.replace(output_path)
        return report
    finally:
        with suppress(FileNotFoundError):
            candidate_path.unlink()


def validate_submission(
    path: Path | str,
    expected_questions: Iterable[QuestionRecord] | None = None,
) -> dict[str, Any]:
    """逐题校验提交列、ID、答案类型及比赛输出协议。"""

    frame = pd.read_excel(path, dtype=object, keep_default_na=False)
    required_columns = {"id", "answer"}
    if not required_columns.issubset(frame.columns):
        missing_columns = sorted(required_columns - set(frame.columns))
        raise ValueError(f"提交文件缺少必填列: {missing_columns}")
    if frame["id"].duplicated().any():
        raise ValueError("提交文件包含重复 id")

    question_by_id: dict[int, QuestionRecord] = {}
    if expected_questions is not None:
        question_by_id = {question.id: question for question in expected_questions}
        expected_ids = set(question_by_id)
        try:
            actual_ids = {int(value) for value in frame["id"]}
        except (TypeError, ValueError) as exc:
            raise ValueError("提交 id 必须是整数") from exc
        if expected_ids != actual_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(f"提交 ID 不匹配，缺少={missing}，多出={extra}")

    normalized_answers = frame["answer"].astype(str).str.strip()
    explicit_placeholders = set(EMPTY_ANSWER_BY_FORMAT.values())
    empty_mask = normalized_answers.isin({"", *explicit_placeholders})
    empty_ids = [int(value) for value in frame.loc[empty_mask, "id"]]
    format_errors: list[dict[str, Any]] = []
    if question_by_id:
        for row in frame[["id", "answer"]].itertuples(index=False):
            question_id = int(row.id)
            answer_text = str(row.answer).strip()
            expected_placeholder = EMPTY_ANSWER_BY_FORMAT[question_by_id[question_id].answer_format]
            answer = "" if answer_text in {"", expected_placeholder} else row.answer
            errors = validate_answer_text(
                answer,
                question_by_id[question_id],
                allow_blank=True,
            )
            format_errors.extend({"id": question_id, "message": message} for message in errors)

    return {
        "is_valid": not format_errors,
        "row_count": len(frame),
        "empty_answer_count": len(empty_ids),
        "empty_answer_ids": empty_ids,
        "format_error_count": len(format_errors),
        "format_errors": format_errors,
    }
