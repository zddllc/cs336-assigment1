
import numpy
import math
import torch


def get_lr_cosine_schedule(t, alpha_max, alpha_min, warmup_iters, cosine_cycle_iters):
    
    """
    Compute the learning rate at iteration t using a cosine schedule with linear warmup.

    Args:
        t (int): Current iteration.
        alpha_max (float): Maximum learning rate.
        alpha_min (float): Minimum learning rate.
        warmup_iters (int): Number of warmup iterations.
        cosine_cycle_iters (int): Number of iterations in one cosine cycle.

    Returns:
        float: Learning rate at iteration t.
    """
    if t < warmup_iters:
        return alpha_max * t / warmup_iters
    elif warmup_iters <= t <= cosine_cycle_iters:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + numpy.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return alpha_min


def gradient_clipping(parameters, max_norm):
    
    """
    Clip the gradients of the given parameters to have a maximum norm of max_norm.

    Args:
        parameters (Iterable[torch.Tensor]): Iterable of model parameters.
        max_norm (float): Maximum allowed norm of the gradients.
    """

    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self):
        # Perform a single optimization step.
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Retrieve state for this parameter
                state = self.state[p]
                
                # Initialize state if it doesn't exist
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                g = p.grad.data

                # Update biased first moment estimate
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                state['step'] += 1
                step = state['step']

                # Compute bias-corrected learning rate
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                alpha_t = lr * (bias_correction2 ** 0.5) / bias_correction1

                # Update parameters using AdamW update rule
                p.data.addcdiv_(m, (v.sqrt() + eps), value=-alpha_t)

                # Apply weight decay separately
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)