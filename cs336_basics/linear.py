import torch


from cs336_basics.utils import truncated_normal

class LinearModule(torch.nn.Module):

    def __init__(self, input_dim, output_dim, device=None, dtype=None):

        super(LinearModule, self).__init__()

        weight_tensor = torch.empty((input_dim, output_dim), device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(truncated_normal(weight_tensor, input_dim, output_dim))

        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):

        return torch.matmul(x, self.weight) + self.bias