import pytest
from pydantic import ValidationError

from table_qa_agent.schemas import TableStructureAnswer
from table_qa_agent.structure import StructureRepairError, repair_structure


def test_structure_repair_infers_minimum_dimensions_and_deduplicates() -> None:
    cell = {"text": "金额", "row": 1, "col": 2, "rowspan": 1, "colspan": 2}
    repaired = repair_structure({"row_count": 1, "col_count": 2, "cells": [cell, cell]})

    assert repaired.row_count == 2
    assert repaired.col_count == 4
    assert len(repaired.cells) == 1


def test_structure_repair_does_not_guess_overlapping_semantics() -> None:
    with pytest.raises(StructureRepairError, match="无法安全") as caught:
        repair_structure(
            {
                "row_count": 2,
                "col_count": 1,
                "cells": [
                    {"text": "金额", "row": 0, "col": 0, "rowspan": 1, "colspan": 2},
                    {"text": "2025", "row": 0, "col": 1, "rowspan": 1, "colspan": 1},
                ],
            }
        )
    assert caught.value.candidate is not None
    assert caught.value.candidate["col_count"] == 2


def test_overlap_reports_both_cells_and_preserves_input() -> None:
    import copy

    value = {"row_count": 3, "col_count": 2, "cells": [
        {"text": "类别", "row": 0, "col": 0, "rowspan": 2, "colspan": 1},
        {"text": "子项", "row": 1, "col": 0, "rowspan": 1, "colspan": 1},
    ]}
    original = copy.deepcopy(value)
    with pytest.raises(StructureRepairError) as caught:
        repair_structure(value)
    message = str(caught.value)
    for fragment in ("cells[0]", "cells[1]", "类别", "子项", "rows=[0,2)", "(1, 0)"):
        assert fragment in message
    assert value == original


def test_out_of_bounds_diagnostic_includes_dimension_and_cell_range() -> None:
    with pytest.raises(ValidationError) as caught:
        TableStructureAnswer.model_validate({"row_count": 2, "col_count": 3, "cells": [
            {"text": "表头", "row": 1, "col": 0, "rowspan": 2, "colspan": 1},
        ]})
    assert "row_count=2" in str(caught.value)
    assert "rows=[1,3)" in str(caught.value)


def test_valid_partial_structure_keeps_global_coordinates() -> None:
    value = {"row_count": 20, "col_count": 7, "cells": [
        {"text": "多级表头", "row": 2, "col": 3, "rowspan": 1, "colspan": 3},
        {"text": "", "row": 3, "col": 3, "rowspan": 2, "colspan": 1},
        {"text": "子列", "row": 3, "col": 4, "rowspan": 1, "colspan": 2},
    ]}
    assert repair_structure(value).model_dump() == value
