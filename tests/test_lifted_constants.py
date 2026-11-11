import pytest
import torch
import torch.nn as nn

from dl_backtrace.pytorch_backtrace import DLBacktrace
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.trace_utils import (
    get_weight_from_placeholder,
    map_placeholders_to_state_dict,
)


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 2)

    def forward(self, x):
        return self.a(x) * torch.tensor(2.0)


class BufferScale(Scale):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        return self.a(x) * self.scale


class Floor(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 2)

    def forward(self, x):
        return self.a(torch.where(x > 0, x, torch.tensor(0.5)))


class TwoConstants(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 2)

    def forward(self, x):
        return self.a(x) * torch.tensor(2.0) + torch.tensor(3.0)


def trace(model):
    torch.manual_seed(0)
    x = torch.randn(1, 4)
    return model.eval(), x, DLBacktrace(model=model, input_for_graph=(x,), device="cpu", verbose=False)


def test_constant_placeholder_resolves_to_its_value():
    model = Scale().eval()
    program = torch.export.export(model, (torch.randn(1, 4),))
    mapping = map_placeholders_to_state_dict(program, model)
    assert mapping["c_lifted_tensor_0"] == "lifted_tensor_0"
    value = get_weight_from_placeholder("c_lifted_tensor_0", program, model, mapping)
    assert value.item() == 2.0


def test_each_constant_keeps_its_own_value():
    model = TwoConstants().eval()
    program = torch.export.export(model, (torch.randn(1, 4),))
    mapping = map_placeholders_to_state_dict(program, model)
    values = [get_weight_from_placeholder(name, program, model, mapping).item() for name in sorted(mapping) if name.startswith("c_")]
    assert values == [2.0, 3.0]


def test_parameters_still_resolve():
    model = Scale().eval()
    program = torch.export.export(model, (torch.randn(1, 4),))
    mapping = map_placeholders_to_state_dict(program, model)
    assert mapping["p_a_weight"] == "a.weight"
    value = get_weight_from_placeholder("p_a_weight", program, model, mapping)
    assert torch.equal(value, model.a.weight)


@pytest.mark.parametrize("model_class", [Scale, Floor, TwoConstants])
def test_forward_pass_reaches_the_output(model_class):
    model, x, dlb = trace(model_class())
    node_io = dlb.predict(x)
    assert torch.allclose(node_io["output"]["output_values"], model(x))


@pytest.mark.parametrize("model_class", [Scale, Floor, TwoConstants])
def test_relevance_reaches_the_input(model_class):
    model, x, dlb = trace(model_class())
    dlb.predict(x)
    relevance = dlb.evaluation(mode="default", multiplier=100.0, task="multi-class classification")
    assert relevance["output"].sum() == pytest.approx(100.0)
    assert "x" in relevance


def test_constant_explains_like_the_same_value_in_a_buffer():
    relevance = []
    for model_class in (BufferScale, Scale):
        torch.manual_seed(1)
        _, x, dlb = trace(model_class())
        dlb.predict(x)
        relevance.append(dlb.evaluation(mode="default", multiplier=100.0, task="multi-class classification")["x"])
    buffered, constant = relevance
    assert constant == pytest.approx(buffered)
    assert constant.sum() != 0
