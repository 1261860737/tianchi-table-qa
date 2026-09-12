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


def conflict_indices(value: dict[str, Any]) -> list[int]:
    cells = [TableCell.model_validate(cell) for cell in value["cells"]]
    indices: set[int] = set()
    for i, a in enumerate(cells):
        if a.row + a.rowspan > value["row_count"] or a.col + a.colspan > value["col_count"]:
            indices.add(i)
        for j, b in enumerate(cells[:i]):
            if (
                max(a.row, b.row) < min(a.row + a.rowspan, b.row + b.rowspan)
                and max(a.col, b.col) < min(a.col + a.colspan, b.col + b.colspan)
            ):
                indices.update((i, j))
    return sorted(indices)


def apply_structure_patch(value: dict[str, Any], proposal: Any) -> TableStructureAnswer:
    patch = StructurePatch.model_validate(proposal)
    allowed = set(conflict_indices(value))
    indices = [update.index for update in patch.updates]
    if len(indices) != len(set(indices)) or not set(indices).issubset(allowed):
        raise ValueError("修复只能修改冲突单元格，且索引不得重复")
    candidate = deepcopy(value)
    for update in patch.updates:
        candidate["cells"][update.index].update(update.model_dump(exclude={"index"}))
    # 不自动扩容、删除单元格或进行第二次猜测性修复。
    return TableStructureAnswer.model_validate(candidate)
