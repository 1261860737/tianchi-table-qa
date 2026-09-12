"""题目、证据、操作与运行日志的数据协议。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionType = Literal["structure", "extract", "thinking"]
AnswerFormat = Literal["string", "number", "json_array", "json"]
EvidenceStatus = Literal["success", "ambiguous", "insufficient"]
SpecialistName = Literal["extract", "compute", "structure", "visual_attribute"]
TaskKind = Literal[
    "extract_scalar",
    "extract_multi",
    "extract_region",
    "compute_arithmetic",
    "compute_duration",
    "compute_extreme",
    "compute_arg_extreme",
    "compute_boolean",
    "structure_recover",
    "structure_measure",
    "visual_attribute",
]
FieldRole = Literal[
    "label",
    "measure",
    "start_time",
    "end_time",
    "category",
    "attribute",
    "structure",
]
EvidenceValueType = Literal["string", "number", "time", "date", "boolean", "array", "object"]


class RequiredField(BaseModel):
    """Planner 声明的输入字段，不绑定具体题目 ID。"""

    name: str = Field(min_length=1)
    role: FieldRole
    value_type: EvidenceValueType
    aliases: list[str] = Field(default_factory=list)
    required: bool = True


ProjectionPathItem = Annotated[str | int, Field(union_mode="left_to_right")]


class AnswerProjection(BaseModel):
    """从完整工具结果中选择最终提交所需的字段。"""

    mode: Literal["identity", "path", "fields"] = "identity"
    path: list[ProjectionPathItem] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_projection(self) -> AnswerProjection:
        if self.mode == "path" and not self.path:
            raise ValueError("path 投影必须提供 path")
        if self.mode == "fields" and not self.fields:
            raise ValueError("fields 投影必须提供 fields")
        return self


class TaskPlan(BaseModel):
    """确定性路由与专家 Agent 共享的任务计划。"""

    task_kind: TaskKind
    specialist: SpecialistName
    mode: str = "default"
    required_fields: list[RequiredField] = Field(default_factory=list)
    preferred_operation: str | None = None
    answer_projection: AnswerProjection = Field(default_factory=AnswerProjection)
    needs_localization: bool = False
    needs_ocr: bool = False

    @model_validator(mode="after")
    def validate_specialist_matches_task(self) -> TaskPlan:
        expected: dict[str, SpecialistName] = {
            "extract_scalar": "extract",
            "extract_multi": "extract",
            "extract_region": "extract",
            "compute_arithmetic": "compute",
            "compute_duration": "compute",
            "compute_extreme": "compute",
            "compute_arg_extreme": "compute",
            "compute_boolean": "compute",
            "structure_recover": "structure",
            "structure_measure": "structure",
            "visual_attribute": "visual_attribute",
        }
        required_specialist = expected[self.task_kind]
        if self.specialist != required_specialist:
            raise ValueError(
                f"task_kind={self.task_kind} 必须路由到 {required_specialist}，"
                f"不能路由到 {self.specialist}"
            )
        return self


class RegionRef(BaseModel):
    """页内归一化候选区域，bbox 顺序为 x1,y1,x2,y2。"""

    page: int = Field(ge=1)
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    bbox: tuple[float, float, float, float]
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bbox(self) -> RegionRef:
        x1, y1, x2, y2 = self.bbox
        if not all(0.0 <= value <= 1.0 for value in self.bbox):
            raise ValueError("bbox 必须使用 0 到 1 的归一化坐标")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox 必须满足 x1 < x2 且 y1 < y2")
        return self


class OCRTextBlock(BaseModel):
    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class OCRResult(BaseModel):
    blocks: list[OCRTextBlock] = Field(default_factory=list)

    def as_prompt_text(self, max_chars: int = 6000) -> str:
        text = "\n".join(block.text for block in self.blocks if block.text.strip())
        return text[:max_chars]


class RecoveryAttempt(BaseModel):
    action: Literal["repair_operation", "locate_crop_ocr", "repair_structure", "patch_structure"]
    reason: str
    succeeded: bool = False
    raw_output: str | None = None


class QuestionRecord(BaseModel):
    """tests.xlsx 中的一道题。"""

    id: int = Field(gt=0)
    file_name: str = Field(min_length=1)
    question_type: QuestionType
    question: str = Field(min_length=1)
    table_hint: str | None = None
    answer_format: AnswerFormat


class EvidenceItem(BaseModel):
    """模型从文档中定位出的最小证据。"""

    # 视觉模型偶尔会在稳定 ID 中保留原字段的 Unicode 字符。ID 只用于本次响应内
    # 的引用，无需强制 ASCII；禁止空白即可避免歧义。
    id: str | None = Field(default=None, pattern=r"^\S+$")
    role: FieldRole | None = None
    entity: str | None = None
    metric: str | None = None
    row_header: str | list[str] | None = None
    column_header: str | list[str] | None = None
    # 结构恢复和区域提取时原始值可以是数组或对象。
    value_raw: Any = None
    value: Any
    value_type: EvidenceValueType | None = None
    unit: str | None = None
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    source_text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("bbox", mode="before")
    @classmethod
    def clip_bbox_to_page(cls, value: Any) -> Any:
        """裁剪模型轻微越界的 bbox；裁剪后无面积仍交给几何校验拒绝。"""

        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return value
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return value
        return tuple(min(1.0, max(0.0, float(item))) for item in value)

    @model_validator(mode="after")
    def validate_bbox(self) -> EvidenceItem:
        if self.bbox is None:
            return self
        x1, y1, x2, y2 = self.bbox
        if not all(0.0 <= value <= 1.0 for value in self.bbox):
            raise ValueError("Evidence bbox 必须使用 0 到 1 的归一化坐标")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Evidence bbox 必须满足 x1 < x2 且 y1 < y2")
        return self


AtomicOperationName = Literal[
    "lookup",
    "list",
    "count",
    "add",
    "subtract",
    "multiply",
    "divide",
    "ratio",
    "percentage_change",
    "percentage_of_total",
    "percentage_point_difference",
    "sum",
    "average",
    "max",
    "min",
    "duration",
    "sum_durations",
    "argmax",
    "argmin",
    "boolean",
    "concat",
]


class OperationStep(BaseModel):
    """多步确定性计算中的一个原子步骤。"""

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    name: AtomicOperationName
    arguments: dict[str, Any] = Field(default_factory=dict)


class OperationSpec(BaseModel):
    """交给 Python 确定性执行的单步操作或有向多步计划。"""

    name: AtomicOperationName | Literal["pipeline"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    steps: list[OperationStep] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_pipeline_name(cls, value: Any) -> Any:
        """有 steps 时按 pipeline 解析，容忍模型把 name 留成首个原子操作。"""

        if not isinstance(value, dict) or not value.get("steps"):
            return value
        if value.get("name") == "pipeline":
            return value
        normalized = dict(value)
        normalized["name"] = "pipeline"
        normalized["arguments"] = {}
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> OperationSpec:
        if self.name == "pipeline":
            if not self.steps:
                raise ValueError("pipeline 必须至少包含一个步骤")
            step_ids = [step.id for step in self.steps]
            if len(step_ids) != len(set(step_ids)):
                raise ValueError("pipeline 步骤 id 不能重复")
        elif self.steps:
            raise ValueError("原子操作不能包含 steps")
        return self


class EvidenceResponse(BaseModel):
    """Evidence Agent 的结构化输出。"""

    model_config = ConfigDict(extra="ignore")

    status: EvidenceStatus = "success"
    question_type: str
    target: dict[str, Any] = Field(default_factory=dict)
    task_plan: TaskPlan | None = None
    required_fields: list[RequiredField] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    operation: OperationSpec | None = None
    # Extract/Visual 工作流可以直接提交抽取结果，不必伪造 lookup Operation。
    direct_answer: Any = None
    answer_projection: AnswerProjection | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> EvidenceResponse:
        identifiers = [item.id for item in self.evidence if item.id is not None]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evidence id 不能重复")
        return self


class TableCell(BaseModel):
    """结构恢复答案中的一个真实单元格。"""

    model_config = ConfigDict(extra="forbid")

    text: str
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(ge=1)
    colspan: int = Field(ge=1)


class TableStructureAnswer(BaseModel):
    """比赛要求的结构恢复答案。"""

    model_config = ConfigDict(extra="forbid")

    row_count: int = Field(ge=0)
    col_count: int = Field(ge=0)
    cells: list[TableCell]

    @model_validator(mode="after")
    def validate_cells(self) -> TableStructureAnswer:
        def describe(index: int) -> str:
            cell = self.cells[index]
            return (
                f"cells[{index}] text={cell.text[:120]!r} "
                f"rows=[{cell.row},{cell.row + cell.rowspan}) "
                f"cols=[{cell.col},{cell.col + cell.colspan})"
            )

        occupied: dict[tuple[int, int], int] = {}
        for index, cell in enumerate(self.cells):
            if cell.row + cell.rowspan > self.row_count:
                raise ValueError(
                    f"{describe(index)} 超出 row_count={self.row_count}；"
                    "检查完整表格尺寸或单元格行范围，不按裁剪图重新编号"
                )
            if cell.col + cell.colspan > self.col_count:
                raise ValueError(
                    f"{describe(index)} 超出 col_count={self.col_count}；"
                    "检查完整表格尺寸或单元格列范围"
                )

            for row in range(cell.row, cell.row + cell.rowspan):
                for col in range(cell.col, cell.col + cell.colspan):
                    coordinate = (row, col)
                    if coordinate in occupied:
                        previous = occupied[coordinate]
                        raise ValueError(
                            f"{describe(index)} 与 {describe(previous)} 在 {coordinate} 重叠；"
                            "坐标从 0 开始，范围为左闭右开；需核对位置、跨度或重复输出，"
                            "不能仅为消除冲突而删除单元格"
                        )
                    occupied[coordinate] = index
        return self


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class RunResult(BaseModel):
    """一道题的完整可观测运行结果。"""

    question_id: int
    question: str
    document_id: str
    document_path: str | None = None
    pages: list[str] = Field(default_factory=list)
    plan: TaskPlan | None = None
    planning_raw_output: str | None = None
    specialist: SpecialistName | None = None
    regions: list[RegionRef] = Field(default_factory=list)
    ocr_text: str | None = None
    recovery_attempts: list[RecoveryAttempt] = Field(default_factory=list)
    evidence: EvidenceResponse | None = None
    tool_result: Any = None
    final_answer: str | None = None
    raw_model_output: str | None = None
    latency_seconds: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    status: Literal["success", "ambiguous", "insufficient", "error"] = "error"
    error_type: str | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
