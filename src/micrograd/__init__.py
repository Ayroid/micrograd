from .engine import Value
from .nn import MLP, Layer, Neuron

__all__ = ["Value", "Neuron", "Layer", "MLP", "main"]


def main() -> None:
    """Train a small MLP on a 4-example dataset and print the result."""
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    net = MLP(3, [4, 4, 1])
    for step in range(100):
        loss = sum((net(x) - y) ** 2 for x, y in zip(xs, ys))
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.data += -0.05 * p.grad
        if step % 10 == 0:
            print(f"step {step:3d}  loss {loss.data:.6f}")
    print([round(net(x).data, 3) for x in xs], "targets", ys)
