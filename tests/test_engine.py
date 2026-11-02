import torch

from micrograd import MLP, Value


def test_matches_torch():
    a, b = Value(-4.0), Value(2.0)
    y = ((a * b + b**3) / (a + 3.0)).tanh() + (a - b).exp()
    y.backward()

    at, bt = torch.tensor(-4.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
    yt = ((at * bt + bt**3) / (at + 3.0)).tanh() + (at - bt).exp()
    yt.backward()

    assert abs(y.data - yt.item()) < 1e-5
    assert abs(a.grad - at.grad.item()) < 1e-5
    assert abs(b.grad - bt.grad.item()) < 1e-5


def test_mlp_has_41_params():
    assert len(MLP(3, [4, 4, 1]).parameters()) == 41
