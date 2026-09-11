import pytest
from pydantic import ValidationError

from table_qa_agent.schemas import EvidenceItem, TableStructureAnswer


def test_evidence_raw_value_accepts_region_arrays() -> None:
    evidence = EvidenceItem(value_raw=["项目", "金额"], value=["项目", "金额"])

    assert evidence.value_raw == ["项目", "金额"]


def test_structure_cells_may_leave_gaps_for_partial_recovery() -> None:
    answer = TableStructureAnswer.model_validate(
        {
            "row_count": 10,
            "col_count": 5,
            "cells": [
                {"text": "项目", "row": 0, "col": 0, "rowspan": 2, "colspan": 1},
                {"text": "销售额", "row": 2, "col": 0, "rowspan": 1, "colspan": 1},
            ],
        }
    )

    assert answer.row_count == 10
    assert len(answer.cells) == 2


def test_structure_cells_cannot_overlap() -> None:
    with pytest.raises(ValidationError, match="重叠"):
        TableStructureAnswer.model_validate(
            {
                "row_count": 2,
                "col_count": 2,
                "cells": [
                    {"text": "金额", "row": 0, "col": 0, "rowspan": 1, "colspan": 2},
                    {"text": "2025", "row": 0, "col": 1, "rowspan": 1, "colspan": 1},
                ],
            }
        )
