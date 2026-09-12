"""只修复可由几何约束确定的问题，不猜测表格语义。"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from table_qa_agent.schemas import TableCell, TableStructureAnswer


class StructureRepairError(ValueError):
    """结构无法通过确定性规则安全修复。"""


def repair_structure(value: Any) -> TableStructureAnswer:
    if isinstance(value, TableStructureAnswer):
        return value
    if not isinstance(value, dict) or not isinstance(value.get("cells"), list):
        raise StructureRepairError("结构答案必须是包含 cells 的对象")

    cells: list[TableCell] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for index, raw_cell in enumerate(value["cells"]):
        try:
            cell = TableCell.model_validate(raw_cell)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                for error in exc.errors(include_input=False)
            )
            raise StructureRepairError(f"cells[{index}] 单元格字段不合法: {details}") from exc
        signature = (cell.text, cell.row, cell.col, cell.rowspan, cell.colspan)
        if signature in seen:
            continue
        seen.add(signature)
        cells.append(cell)

    row_count = max(
        int(value.get("row_count") or 0),
        max((cell.row + cell.rowspan for cell in cells), default=0),
    )
    col_count = max(
        int(value.get("col_count") or 0),
        max((cell.col + cell.colspan for cell in cells), default=0),
    )

    try:
        return TableStructureAnswer(row_count=row_count, col_count=col_count, cells=cells)
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors(include_input=False))
        raise StructureRepairError(
            "结构存在无法安全自动处理的重叠或几何冲突；"
            f"以下索引对应去除完全重复项后的 cells：{details}"
        ) from exc
