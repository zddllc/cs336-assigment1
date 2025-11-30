import torch

from einops import rearrange


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


def softmax(x, dim=-1):

    #步骤1: 在指定维度上找到最大值，保持维度以便广播
    x_max = x.max(dim=dim, keepdim=True).values
    
    # 步骤2: 减去最大值（数值稳定性关键）
    x_shifted = x - x_max
    
    # 步骤3: 计算 exp
    exp_x = torch.exp(x_shifted)
    
    # 步骤4: 归一化
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    softmax_x = exp_x / sum_exp
    
    return softmax_x


def softmax2(x, dim=-1):

    x_max = x.max(dim=-1, keepdim=True).values
    x = x - x_max
    log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)

    return torch.exp(log_probs)

def cross_entropy(predictions, targets):

    # Subtract max value for numerical stability
    pred_max = predictions.max(dim=-1, keepdim=True).values
    predictions = predictions - pred_max

    log_probs = predictions - torch.logsumexp(predictions, dim=-1, keepdim=True)
    
    # Gather the log probabilities corresponding to the target indices
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Compute the negative log likelihood loss
    return -target_log_probs.mean()