import torch
from adam.optimizer import Adam

def test_linear_regression_loss_decreases():
    torch.manual_seed(42)

    # y = 3x + 2 with noise
    x = torch.randn(200, 1)
    y = 3 * x + 2 + 0.1 * torch.randn_like(x)

    w = torch.randn(1, 1, requires_grad=True)
    b = torch.randn(1, 1, requires_grad=True)

    opt = Adam([w, b], lr=0.05)

    losses = []
    for _ in range(200):
        pred = x @ w + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Loss did not decrease"
    assert abs(w.item() - 3) < 0.2, "Weight not close to true value"
    assert abs(b.item() - 2) < 0.2, "Bias not close to true value"


def test_adam_matches_pytorch():
    torch.manual_seed(0)

    # dummy param
    x = torch.randn(10, requires_grad=True)
    y = x ** 2
    y.sum().backward()

    x1 = x.clone().detach().requires_grad_()
    x2 = x.clone().detach().requires_grad_()

    torch_opt = torch.optim.Adam([x1], lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    my_opt = Adam([x2], lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(10):
        # dummy loss: x^2
        y1 = (x1 ** 2).sum()
        y2 = (x2 ** 2).sum()
        y1.backward()
        y2.backward()
        torch_opt.step()
        my_opt.step()
        torch_opt.zero_grad()
        my_opt.zero_grad()

    diff = torch.norm(x1 - x2).item()
    assert diff < 1e-3, f"Mismatch with PyTorch Adam: {diff}"


def test_handles_zero_grad():
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    opt = Adam([x], lr=1e-2)

    # no backward called = no .grad
    opt.step()  # should not crash

    # call zero_grad safely
    opt.zero_grad()