"""只允许修复冲突单元格的几何属性，不更改文本和非冲突单元格。"""

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from table_qa_agent.schemas import TableCell, TableStructureAnswer


class CellPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    index: int = Field(ge=0)
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(ge=1)
    colspan: int = Field(ge=1)


class StructurePatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    updates: list[CellPatch] = Field(min_length=1)


def conflict_details(value: dict[str, Any]) -> list[dict[str, Any]]:
    """一次列出全部越界与重叠，避免修复器只看到校验器的首个错误。"""

    cells = [TableCell.model_validate(cell) for cell in value["cells"]]
    details: list[dict[str, Any]] = []
    for i, a in enumerate(cells):
        row_end = a.row + a.rowspan
        col_end = a.col + a.colspan
        if row_end > value["row_count"] or col_end > value["col_count"]:
            details.append({
                "kind": "out_of_bounds",
                "indices": [i],
                "cell": a.model_dump(),
                "row_count": value["row_count"],
                "col_count": value["col_count"],
            })
        for j, b in enumerate(cells[:i]):
            if (
                max(a.row, b.row) < min(a.row + a.rowspan, b.row + b.rowspan)
                and max(a.col, b.col) < min(a.col + a.colspan, b.col + b.colspan)
            ):
                details.append({
                    "kind": "overlap",
                    "indices": [j, i],
                    "cells": [b.model_dump(), a.model_dump()],
                })
    return details


def conflict_indices(value: dict[str, Any]) -> list[int]:
    indices = {
        index
        for detail in conflict_details(value)
        for index in detail["indices"]
    }
    return sorted(indices)


def apply_structure_patch_candidate(
    value: dict[str, Any], proposal: Any,
) -> dict[str, Any]:
    """应用受限几何补丁并返回候选；最终合法性由调用者统一确认。"""

    patch = StructurePatch.model_validate(proposal)
    allowed = set(conflict_indices(value))
    indices = [update.index for update in patch.updates]
    if len(indices) != len(set(indices)) or not set(indices).issubset(allowed):
        raise ValueError("修复只能修改冲突单元格，且索引不得重复")
    candidate = deepcopy(value)
    for update in patch.updates:
        candidate["cells"][update.index].update(update.model_dump(exclude={"index"}))
    return candidate


def apply_structure_patch(value: dict[str, Any], proposal: Any) -> TableStructureAnswer:
    candidate = apply_structure_patch_candidate(value, proposal)
    # 不自动扩容、删除单元格或修改非冲突单元格。
    return TableStructureAnswer.model_validate(candidate)
