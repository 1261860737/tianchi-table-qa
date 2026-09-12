"""项目命令行入口。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from table_qa_agent.agents import build_specialist_router
from table_qa_agent.client import ModelConfigurationError, OpenAICompatibleVLClient
from table_qa_agent.config import load_config
from table_qa_agent.dataset import (
    DocumentResolver,
    read_questions,
    select_questions,
    summarize_dataset,
)
from table_qa_agent.documents import DocumentProcessor
from table_qa_agent.normalizer import validate_answer_text
from table_qa_agent.ocr import VisionOCRBackend
from table_qa_agent.orchestration import FunctionCallingIntentPlanner, TaskPlanner
from table_qa_agent.pipeline import (
    BaselinePipeline,
    JsonlRunStore,
    SubmissionBuildError,
    force_complete_results,
    publish_submission,
    select_submission_results,
    validate_submission,
)
from table_qa_agent.retrieval import PageRegionLocator
from table_qa_agent.schemas import QuestionRecord, RunResult

app = typer.Typer(
    name="table-qa",
    help="复杂表格识别与内容理解分层专家基线",
    no_args_is_help=True,
)
console = Console()


class AnswerPolicy(StrEnum):
    """运行答案策略；默认只硬校验最终提交格式。"""

    FORMAT_ONLY = "format-only"
    STRICT_EVIDENCE = "strict-evidence"


def _uses_format_only_policy(
    answer_policy: AnswerPolicy,
    experimental_all_answers: bool,
) -> bool:
    """旧实验参数等价于 format-only，确保已有命令向后兼容。"""

    return experimental_all_answers or answer_policy == AnswerPolicy.FORMAT_ONLY


def _parse_ids(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise typer.BadParameter("--ids 应为逗号分隔的整数，如 1,2,63") from exc


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    logger.add(
        log_dir / "table_qa_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="20 MB",
        retention="14 days",
        encoding="utf-8",
    )


def _apply_answer_overrides(
    results: list[RunResult],
    questions: list[QuestionRecord],
    path: Path | None,
) -> list[RunResult]:
    """应用人工复核答案；配置与题目格式不一致时立即失败。"""

    if path is None:
        return results
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_answers = payload.get("answers", payload)
    if not isinstance(raw_answers, dict):
        raise ValueError("答案覆盖文件必须是 id 到 answer 的映射，或包含 answers 映射")
    question_by_id = {question.id: question for question in questions}
    result_by_id = {result.question_id: result for result in results}
    for raw_id, raw_answer in raw_answers.items():
        question_id = int(raw_id)
        if question_id not in question_by_id:
            raise ValueError(f"答案覆盖文件包含未知题目 ID: {question_id}")
        answer = str(raw_answer).strip()
        errors = validate_answer_text(
            answer,
            question_by_id[question_id],
            allow_blank=False,
        )
        if errors:
            raise ValueError(f"题目 {question_id} 的人工答案格式错误: {'；'.join(errors)}")
        current = result_by_id.get(question_id)
        if current is None:
            current = RunResult(
                question_id=question_id,
                question=question_by_id[question_id].question,
                document_id=question_by_id[question_id].file_name,
            )
        result_by_id[question_id] = current.model_copy(
            update={
                "final_answer": answer,
                "status": "success",
                "error_type": None,
                "error_message": None,
                "warnings": [*current.warnings, f"人工复核覆盖: {path}"],
            }
        )
    return [result_by_id[question.id] for question in questions]


@app.command("inspect-data")
def inspect_data(
    input_path: Annotated[Path, typer.Option("--input", exists=True)] = Path("tests.xlsx"),
    files_dir: Annotated[Path, typer.Option("--files-dir", exists=True)] = Path("files"),
) -> None:
    """检查题型分布、输出格式以及文件名纠错情况。"""

    questions = read_questions(input_path)
    summary = summarize_dataset(questions, DocumentResolver(files_dir))
    console.print_json(json.dumps(summary, ensure_ascii=False))


@app.command("preprocess")
def preprocess(
    input_path: Annotated[Path, typer.Option("--input", exists=True)] = Path("tests.xlsx"),
    files_dir: Annotated[Path, typer.Option("--files-dir", exists=True)] = Path("files"),
    config_path: Annotated[Path, typer.Option("--config", exists=True)] = Path(
        "configs/baseline.yaml"
    ),
    ids: Annotated[str | None, typer.Option(help="逗号分隔的题目 ID")] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """只做文件解析与 PDF 转图，不调用模型。"""

    config = load_config(config_path)
    resolver = DocumentResolver(files_dir)
    processor = DocumentProcessor(config.documents)
    questions = select_questions(read_questions(input_path), ids=_parse_ids(ids), limit=limit)
    unique_files = sorted({question.file_name for question in questions})

    table = Table("题目文件", "实际文件", "页数")
    for file_name in unique_files:
        path = resolver.resolve(file_name)
        pages = processor.prepare(path)
        table.add_row(file_name, path.name, str(len(pages)))
    console.print(table)


@app.command("inspect-plan")
def inspect_plan(
    input_path: Annotated[Path, typer.Option("--input", exists=True)] = Path("tests.xlsx"),
    ids: Annotated[str | None, typer.Option(help="逗号分隔的题目 ID")] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """查看题目路由和初始 TaskPlan，不调用模型。"""

    questions = select_questions(read_questions(input_path), ids=_parse_ids(ids), limit=limit)
    planner = TaskPlanner()
    payload = [
        {
            "id": question.id,
            "question": question.question,
            "plan": planner.plan(question).model_dump(mode="json"),
        }
        for question in questions
    ]
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("run")
def run(
    input_path: Annotated[Path, typer.Option("--input", exists=True)] = Path("tests.xlsx"),
    files_dir: Annotated[Path, typer.Option("--files-dir", exists=True)] = Path("files"),
    output_path: Annotated[Path, typer.Option("--output")] = Path(
        "outputs/submissions/submission_v0.xlsx"
    ),
    config_path: Annotated[Path, typer.Option("--config", exists=True)] = Path(
        "configs/baseline.yaml"
    ),
    log_path: Annotated[Path | None, typer.Option("--log-path")] = None,
    ids: Annotated[str | None, typer.Option(help="逗号分隔的题目 ID")] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, max=32, help="并发请求数")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    fallback_logs: Annotated[
        list[Path] | None,
        typer.Option("--fallback-log", exists=True, help="可重复指定历史日志，防止正确答案回退"),
    ] = None,
    override_file: Annotated[
        Path | None,
        typer.Option("--override-file", exists=True, help="人工复核答案 YAML"),
    ] = None,
    answer_policy: Annotated[
        AnswerPolicy,
        typer.Option(
            "--answer-policy",
            help="答案策略：format-only 只硬校验最终格式；strict-evidence 启用严格证据门禁",
        ),
    ] = AnswerPolicy.FORMAT_ONLY,
    experimental_all_answers: Annotated[
        bool,
        typer.Option(
            "--experimental-all-answers",
            help="兼容旧命令，等价于 --answer-policy format-only",
        ),
    ] = False,
) -> None:
    """运行分层 Structured Evidence Agent 并生成提交工作簿。"""

    config = load_config(config_path)
    if workers is not None:
        config.runtime = config.runtime.model_copy(update={"max_workers": workers})
    format_only = _uses_format_only_policy(answer_policy, experimental_all_answers)
    active_policy = AnswerPolicy.FORMAT_ONLY if format_only else AnswerPolicy.STRICT_EVIDENCE
    _configure_logging(config.runtime.log_dir)
    questions = select_questions(read_questions(input_path), ids=_parse_ids(ids), limit=limit)
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = config.runtime.log_dir / f"run_{timestamp}.jsonl"

    try:
        client = OpenAICompatibleVLClient(config.provider)
    except ModelConfigurationError as exc:
        console.print(f"[red]模型配置错误：{exc}[/red]")
        raise typer.Exit(code=2) from exc

    ocr_client = client
    if config.ocr.model and config.ocr.model != config.provider.model:
        ocr_client = OpenAICompatibleVLClient(
            config.provider.model_copy(update={"model": config.ocr.model})
        )
    rule_planner = TaskPlanner()
    planner = (
        FunctionCallingIntentPlanner(
            client,
            fallback=rule_planner,
            fallback_to_rules=config.planning.fallback_to_rules,
        )
        if config.planning.model_intent_enabled
        else rule_planner
    )
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(files_dir),
        processor=DocumentProcessor(config.documents),
        planner=planner,
        router=build_specialist_router(
            client,
            validate_evidence=not format_only,
            force_answer=format_only,
            enable_tool_calls=(
                config.recovery.enabled and config.recovery.max_agent_tool_calls > 0
            ),
        ),
        locator=PageRegionLocator(client, max_regions=config.retrieval.max_regions),
        ocr_backend=VisionOCRBackend(ocr_client),
        runtime=config.runtime,
        retrieval=config.retrieval,
        ocr=config.ocr,
        recovery=config.recovery,
        validate_evidence=not format_only,
    )
    run_results = pipeline.run_batch(
        questions,
        store=JsonlRunStore(log_path),
        resume=resume,
    )
    sources = [run_results]
    sources.extend(JsonlRunStore(path).load_all() for path in fallback_logs or [])
    results = select_submission_results(questions, sources)
    results = _apply_answer_overrides(results, questions, override_file)
    forced_ids: list[int] = []
    if format_only:
        results, forced_ids = force_complete_results(results, questions)
    try:
        report = publish_submission(results, output_path, questions)
    except SubmissionBuildError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print_json(json.dumps(exc.report, ensure_ascii=False))
        raise typer.Exit(code=1) from exc
    error_empty_ids = report["error_derived_empty_ids"]
    success_count = sum(result.status == "success" for result in results)
    local_recovery_ids = [
        result.question_id
        for result in results
        if any("format-only 本地恢复" in warning for warning in result.warnings)
    ]
    console.print(
        f"完成 {len(results)} 题，并发 {config.runtime.max_workers}，"
        f"成功 {success_count} 题，"
        f"空答案 {report['empty_answer_count']}，"
        f"错误导致的空答案 {len(error_empty_ids)}，"
        f"格式错误 {report['format_error_count']}，输出：{output_path}"
    )
    console.print(f"答案策略：{active_policy.value}")
    run_success_count = sum(result.status == "success" for result in run_results)
    replay_ids = [
        result.question_id for result in results
        if "使用当前执行协议从历史 Evidence 重放恢复" in result.warnings
    ]
    console.print(
        f"运行记录：成功 {run_success_count} 题，"
        f"非成功 {len(run_results) - run_success_count} 题；"
        f"提交结果中 Evidence 重放恢复 {len(replay_ids)} 题"
    )
    if replay_ids:
        console.print(f"[yellow]Evidence 重放恢复题目：{replay_ids}[/yellow]")
    if config.planning.model_intent_enabled:
        model_plan_ids = [
            result.question_id
            for result in run_results
            if result.planning_raw_output is not None
            and not any("模型意图规划失败" in warning for warning in result.warnings)
        ]
        plan_fallback_ids = [
            result.question_id
            for result in run_results
            if any("模型意图规划失败" in warning for warning in result.warnings)
        ]
        console.print(
            f"Intent Planner：Function Call 成功 {len(model_plan_ids)} 题，"
            f"规则回退 {len(plan_fallback_ids)} 题"
        )
        if plan_fallback_ids:
            console.print(f"[yellow]Intent Planner 回退题目：{plan_fallback_ids}[/yellow]")
    if local_recovery_ids:
        console.print(
            f"[yellow]format-only 本地恢复 {len(local_recovery_ids)} 题："
            f"{local_recovery_ids}[/yellow]"
        )
    if forced_ids:
        console.print(f"[yellow]格式安全非空兜底 {len(forced_ids)} 题：{forced_ids}[/yellow]")
    console.print(f"逐题日志：{log_path}")


@app.command("build-submission")
def build_submission(
    log_path: Annotated[Path, typer.Option("--log-path", exists=True)],
    output_path: Annotated[Path, typer.Option("--output")],
    input_path: Annotated[Path, typer.Option("--input", exists=True)] = Path("tests.xlsx"),
    fallback_logs: Annotated[
        list[Path] | None,
        typer.Option("--fallback-log", exists=True, help="可重复指定，按顺序回退合法成功答案"),
    ] = None,
    override_file: Annotated[
        Path | None,
        typer.Option("--override-file", exists=True, help="人工复核答案 YAML"),
    ] = None,
    answer_policy: Annotated[
        AnswerPolicy,
        typer.Option(
            "--answer-policy",
            help="答案策略：format-only 补齐非空答案；strict-evidence 保留严格证据门禁",
        ),
    ] = AnswerPolicy.FORMAT_ONLY,
    experimental_all_answers: Annotated[
        bool,
        typer.Option(
            "--experimental-all-answers",
            help="兼容旧命令，等价于 --answer-policy format-only",
        ),
    ] = False,
    replay_success_counts: Annotated[
        bool,
        typer.Option(
            "--replay-success-counts",
            help="安全重算成功日志中的 count；不请求模型，并跳过缺失过滤语义的历史集合",
        ),
    ] = False,
) -> None:
    """从运行日志重放 Evidence、合并历史成功结果并生成提交文件。"""

    questions = read_questions(input_path)
    format_only = _uses_format_only_policy(answer_policy, experimental_all_answers)
    active_policy = AnswerPolicy.FORMAT_ONLY if format_only else AnswerPolicy.STRICT_EVIDENCE
    sources = [JsonlRunStore(log_path).load_all()]
    sources.extend(JsonlRunStore(path).load_all() for path in fallback_logs or [])
    results = select_submission_results(
        questions,
        sources,
        replay_success_counts=replay_success_counts,
    )
    results = _apply_answer_overrides(results, questions, override_file)
    forced_ids: list[int] = []
    if format_only:
        results, forced_ids = force_complete_results(results, questions)
    try:
        report = publish_submission(results, output_path, questions)
    except SubmissionBuildError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print_json(json.dumps(exc.report, ensure_ascii=False))
        raise typer.Exit(code=1) from exc
    error_empty_ids = report["error_derived_empty_ids"]
    console.print(
        f"生成 {len(results)} 题，空答案 {report['empty_answer_count']}，"
        f"错误导致的空答案 {len(error_empty_ids)}，格式错误 {report['format_error_count']}，"
        f"输出：{output_path}"
    )
    console.print(f"答案策略：{active_policy.value}")
    replayed_ids = [
        result.question_id for result in results
        if "使用当前执行协议重算历史成功 Operation" in result.warnings
    ]
    if replayed_ids:
        console.print(
            f"[yellow]确定性重算成功 count {len(replayed_ids)} 题：{replayed_ids}[/yellow]"
        )
    if forced_ids:
        console.print(f"[yellow]格式安全非空兜底 {len(forced_ids)} 题：{forced_ids}[/yellow]")


@app.command("validate-submission")
def validate_submission_command(
    submission: Annotated[Path, typer.Argument(exists=True)],
    input_path: Annotated[Path | None, typer.Option("--input", exists=True)] = None,
) -> None:
    """校验已有提交文件。"""

    expected = read_questions(input_path) if input_path is not None else None
    report = validate_submission(submission, expected)
    console.print_json(json.dumps(report, ensure_ascii=False))
    if not report["is_valid"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
