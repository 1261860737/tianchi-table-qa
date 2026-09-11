"""专业工作流 Agent；共享底层客户端和能力，但拥有不同输出合同与工具白名单。"""

from __future__ import annotations

from table_qa_agent.agent import EvidenceAgent
from table_qa_agent.client import OpenAICompatibleVLClient
from table_qa_agent.orchestration.router import AgentRouter


class ExtractAgent(EvidenceAgent):
    """单值可直接回答；信息不足时可主动请求表格区域检查。"""

    def __init__(
        self,
        client: OpenAICompatibleVLClient,
        *,
        validate_evidence: bool,
        force_answer: bool,
        enable_tool_calls: bool,
    ) -> None:
        super().__init__(
            client,
            specialist="extract",
            validate_evidence=validate_evidence,
            force_answer=force_answer,
            enable_tool_calls=enable_tool_calls,
        )


class ComputeAgent(EvidenceAgent):
    """提取操作数并交由 Python Operation 执行。"""

    def __init__(
        self,
        client: OpenAICompatibleVLClient,
        *,
        validate_evidence: bool,
        force_answer: bool,
        enable_tool_calls: bool,
    ) -> None:
        super().__init__(
            client,
            specialist="compute",
            validate_evidence=validate_evidence,
            force_answer=force_answer,
            enable_tool_calls=enable_tool_calls,
        )


class StructureAgent(EvidenceAgent):
    """使用结构恢复协议，必要时可请求高清表格区域。"""

    def __init__(
        self,
        client: OpenAICompatibleVLClient,
        *,
        validate_evidence: bool,
        force_answer: bool,
        enable_tool_calls: bool,
    ) -> None:
        super().__init__(
            client,
            specialist="structure",
            validate_evidence=validate_evidence,
            force_answer=force_answer,
            enable_tool_calls=enable_tool_calls,
        )


class VisualAttributeAgent(EvidenceAgent):
    """处理颜色、布局和视觉存在性等直接观察问题。"""

    def __init__(
        self,
        client: OpenAICompatibleVLClient,
        *,
        validate_evidence: bool,
        force_answer: bool,
    ) -> None:
        super().__init__(
            client,
            specialist="visual_attribute",
            validate_evidence=validate_evidence,
            force_answer=force_answer,
        )


def build_specialist_router(
    client: OpenAICompatibleVLClient,
    *,
    validate_evidence: bool = True,
    force_answer: bool = False,
    enable_tool_calls: bool = True,
) -> AgentRouter:
    return AgentRouter(
        {
            "extract": ExtractAgent(
                client,
                validate_evidence=validate_evidence,
                force_answer=force_answer,
                enable_tool_calls=enable_tool_calls,
            ),
            "compute": ComputeAgent(
                client,
                validate_evidence=validate_evidence,
                force_answer=force_answer,
                enable_tool_calls=enable_tool_calls,
            ),
            "structure": StructureAgent(
                client,
                validate_evidence=validate_evidence,
                force_answer=force_answer,
                enable_tool_calls=enable_tool_calls,
            ),
            "visual_attribute": VisualAttributeAgent(
                client,
                validate_evidence=validate_evidence,
                force_answer=force_answer,
            ),
        }
    )
