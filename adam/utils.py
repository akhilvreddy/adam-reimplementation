import torch

def flatten_params(params):
    return torch.cat([p.view(-1) for p in params])

def compute_param_diff(params1, params2):
    return torch.norm(flatten_params(params1) - flatten_params(params2)).item()

def zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

def clone_params(params):
    return [p.clone().detach().requires_grad_() for p in params]

def set_seed(seed):
    torch.manual_seed(seed)