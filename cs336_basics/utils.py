import torch

def truncated_normal(tensor, din, dout):

    """Initialize a tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from a normal distribution with mean 0 and
    standard deviation sqrt(2 / (din + dout)), but values more than two standard
    deviations from the mean are redrawn.

    Args:
        tensor (torch.Tensor): The tensor to initialize.
        din (int): The input dimension size.
        dout (int): The output dimension size.
    """
    stddev = (2.0 / (din + dout)) ** 0.5
    mean = 0.0
    lower, upper = -3 * stddev, 3 * stddev

    count = tensor.numel()
    tmp = torch.empty(count * 2, device=tensor.device, dtype=tensor.dtype).normal_(mean=mean, std=stddev)
    valid = tmp[(tmp > lower) & (tmp < upper)]

    while valid.numel() < count:
        tmp = torch.empty(count, device=tensor.device, dtype=tensor.dtype).normal_(mean=mean, std=stddev)
        valid = torch.cat([valid, tmp[(tmp > lower) & (tmp < upper)]], dim=0)

    tensor.copy_(valid[:count].reshape(tensor.shape))
    return tensor