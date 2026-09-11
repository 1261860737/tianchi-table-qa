import pytest

from table_qa_agent.structure import StructureRepairError, repair_structure


def test_structure_repair_infers_minimum_dimensions_and_deduplicates() -> None:
    cell = {"text": "金额", "row": 1, "col": 2, "rowspan": 1, "colspan": 2}
    repaired = repair_structure({"row_count": 1, "col_count": 2, "cells": [cell, cell]})

    assert repaired.row_count == 2
    assert repaired.col_count == 4
    assert len(repaired.cells) == 1


def test_structure_repair_does_not_guess_overlapping_semantics() -> None:
    with pytest.raises(StructureRepairError, match="无法安全"):
        repair_structure(
            {
                "row_count": 2,
                "col_count": 2,
                "cells": [
                    {"text": "金额", "row": 0, "col": 0, "rowspan": 1, "colspan": 2},
                    {"text": "2025", "row": 0, "col": 1, "rowspan": 1, "colspan": 1},
                ],
            }
        )
