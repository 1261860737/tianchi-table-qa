"""Evidence 软验证：风险用于触发修复，不能直接清空候选答案。"""

from __future__ import annotations

from dataclasses import dataclass

from table_qa_agent.executor import ArgumentGroundingError, validate_operation_grounding
from table_qa_agent.schemas import EvidenceResponse, TaskPlan


@dataclass(frozen=True, slots=True)
class EvidenceRisk:
    code: str
    message: str


def assess_evidence(evidence: EvidenceResponse, plan: TaskPlan) -> list[EvidenceRisk]:
    """返回高置信度、可操作的风险；调用方决定是否修复并始终保留原答案。"""

    risks: list[EvidenceRisk] = []
    if evidence.status != "success":
        risks.append(
            EvidenceRisk(
                code="non_success_status",
                message=evidence.reason or f"Evidence 状态为 {evidence.status}",
            )
        )
        return risks
    if not evidence.evidence:
        risks.append(
            EvidenceRisk(
                code="missing_evidence",
                message="模型给出了答案，但没有返回可定位的 Evidence",
            )
        )
    if evidence.direct_answer is not None:
        if plan.specialist not in {"extract", "visual_attribute"}:
            risks.append(
                EvidenceRisk(
                    code="unexpected_direct_answer",
                    message=f"{plan.specialist} 工作流应使用确定性 Operation",
                )
            )
        return risks
    if evidence.operation is None:
        risks.append(
            EvidenceRisk(
                code="missing_operation",
                message="需要工具执行的工作流没有返回 Operation",
            )
        )
        return risks
    try:
        validate_operation_grounding(evidence.operation, evidence.evidence)
    except ArgumentGroundingError as exc:
        risks.append(EvidenceRisk(code="ungrounded_operation", message=str(exc)))
    return risks
