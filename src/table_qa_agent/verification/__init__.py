"""只生成风险信号、不拒绝答案的 Evidence 软检查。"""

from table_qa_agent.verification.evidence import EvidenceRisk, assess_evidence

__all__ = ["EvidenceRisk", "assess_evidence"]
