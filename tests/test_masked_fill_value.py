import pytest
import torch
import torch.nn as nn

from dl_backtrace.pytorch_backtrace import DLBacktrace

X = torch.tensor([[-1.0, 2.0, -3.0, 4.0]])


class Fill(nn.Module):
    def __init__(self, fill):
        super().__init__()
        self.fill = fill
        self.a = nn.Linear(4, 2)

    def forward(self, x):
        return self.a(x.masked_fill(x < 0, self.fill))


class FillFromBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fill", torch.tensor(-1.5))
        self.a = nn.Linear(4, 2)

    def forward(self, x):
        return self.a(x.masked_fill(x < 0, self.fill))


class FillFromGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 2)

    def forward(self, x):
        return self.a(x.masked_fill(x < 0, x.max()))


class MaskedSoftmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x.masked_fill(x < 0, float("-inf")), dim=-1)


class ReplaceToken(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(8, 2)

    def forward(self, ids):
        return self.e(ids.masked_fill(ids == 0, 5))


def predict(model, x):
    model = model.eval()
    dlb = DLBacktrace(model=model, input_for_graph=(x,), device="cpu", verbose=False)
    return model, dlb, dlb.predict(x)


@pytest.mark.parametrize("fill", [-1.5, 0.0, 7, -1e9])
def test_number_fill_is_applied(fill):
    model, _, node_io = predict(Fill(fill), X)
    assert node_io["masked_fill"]["output_values"].tolist() == X.masked_fill(X < 0, fill).tolist()
    assert torch.allclose(node_io["output"]["output_values"], model(X))


@pytest.mark.parametrize("model_class", [FillFromBuffer, FillFromGraph])
def test_tensor_fill_is_applied(model_class):
    model, _, node_io = predict(model_class(), X)
    assert torch.allclose(node_io["output"]["output_values"], model(X))


def test_infinite_fill_masks_the_softmax():
    model, _, node_io = predict(MaskedSoftmax(), X)
    output = node_io["output"]["output_values"]
    assert torch.equal(output, model(X))
    assert output[0, 0] == 0 and output[0, 2] == 0


def test_integer_fill_on_token_ids():
    model, _, node_io = predict(ReplaceToken(), torch.tensor([[0, 3, 0, 1]]))
    assert node_io["masked_fill"]["output_values"].tolist() == [[5, 3, 5, 1]]
    assert torch.equal(node_io["output"]["output_values"], model(torch.tensor([[0, 3, 0, 1]])))


@pytest.mark.parametrize("model", [Fill(-1.5), FillFromBuffer()], ids=["number", "buffer"])
def test_relevance_runs_after_the_fill(model):
    _, dlb, _ = predict(model, X)
    relevance = dlb.evaluation(mode="default", multiplier=100.0, task="multi-class classification")
    assert relevance["output"].sum() == pytest.approx(100.0)
    assert "x" in relevance


class FillFromBiasNamedBuffer(nn.Module):
    """The fill value sits in a buffer whose name contains "bias", which the engine treats as a layer bias."""

    def __init__(self):
        super().__init__()
        self.register_buffer("masked_bias", torch.tensor(-1.5))
        self.a = nn.Linear(4, 2)

    def forward(self, x):
        return self.a(x.masked_fill(x < 0, self.masked_bias))


def test_fill_from_a_bias_named_buffer_matches_a_neutral_name():
    torch.manual_seed(0)
    model = FillFromBiasNamedBuffer()
    neutral = FillFromBuffer()
    neutral.a.load_state_dict(model.a.state_dict())

    model, dlb, node_io = predict(model, X)
    assert node_io is not None
    assert node_io["masked_fill"]["output_values"].tolist() == X.masked_fill(X < 0, -1.5).tolist()
    assert torch.allclose(node_io["output"]["output_values"], model(X))

    _, neutral_dlb, _ = predict(neutral, X)
    kwargs = dict(mode="default", multiplier=100.0, task="multi-class classification")
    relevance = dlb.evaluation(**kwargs)
    assert relevance["x"] == pytest.approx(neutral_dlb.evaluation(**kwargs)["x"])
