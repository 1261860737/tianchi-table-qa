"""参数合同只限制接口形状，不改变合法操作的答案。"""

import pytest

from table_qa_agent.executor import OPERATIONS, OperationExecutionError, execute_operation
from table_qa_agent.operation_contracts import ARGUMENT_MODELS
from table_qa_agent.schemas import OperationSpec


def test_every_operation_has_argument_contract() -> None:
    assert ARGUMENT_MODELS.keys() == OPERATIONS.keys()


@pytest.mark.parametrize("name,arguments", [
    ("add", {"a": 1}),
    ("list", {"values": 1.5}),
    ("argmax", {"records": [1.5]}),
    ("duration", {"start": "10:00", "end": "11:00", "output_unit": "day"}),
    ("sum_durations", {"ranges": [{"start": "10:00"}]}),
])
def test_invalid_shapes_raise_domain_error(name: str, arguments: dict) -> None:
    with pytest.raises(OperationExecutionError):
        execute_operation(OperationSpec(name=name, arguments=arguments), [])


def test_list_preserves_values_order_and_types() -> None:
    values = [20, "10", None]
    assert execute_operation(OperationSpec(name="list", arguments={"values": values}), []) == values
