import torch

from einops import rearrange, einsum
from cs336_basics.utils import truncated_normal
from jaxtyping import Float, Int

class LinearModule(torch.nn.Module):

    def __init__(self, input_dim, output_dim, device=None, dtype=None):

        super(LinearModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def assign_weight(self, new_weight):
        
        self.weight = torch.nn.Parameter(new_weight)

    def forward(self, x):

        return einsum(self.weight, x, "od id, b l id -> b l od")
    

class EmbeddingModule(torch.nn.Module):

    def __init__(self, num_embedding, embedding_dim, device=None, dtype=None):

        super(EmbeddingModule, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

    def assign_weight(self, new_weight):
        
        self.weight = torch.nn.Parameter(new_weight)

    def forward(self, x):

        ret_vec = []
        for i in range(x.shape[0]):
            case_vec = []
            for j in range(x.shape[1]):
                id = x[i][j]
                ev = self.weight[id]
                case_vec.append(ev.tolist())
            ret_vec.append(case_vec)

        return torch.Tensor(ret_vec)
    
class RMSNormModule(torch.nn.Module):

    def __init__(self, dim, eps=1e-5, device=None, dtype=None):

        super(RMSNormModule, self).__init__()
        self.dim = dim
        self.eps = eps

    def assign_weight(self, new_weight):
        
        self.weight = torch.nn.Parameter(new_weight)

    def forward(self, x):

        in_dtype = x.dtype
        x.to(torch.float32)

        norm_x = torch.sum(x * x, dim=-1, keepdim=True)
        norm_x = x / torch.sqrt((norm_x / self.dim) + self.eps)
        rmsnormed_x = norm_x * self.weight
        rmsnormed_x.to(in_dtype)

        return rmsnormed_x
    

class FeedforwardModule(torch.nn.Module):

    def __init__(self, d_model, d_ff, device=None, dtype=None):

        super(FeedforwardModule, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff


    def assign_weight(self, new_weight1, new_weight2, new_weight3):
        
        self.weight1 = torch.nn.Parameter(new_weight1)
        self.weight2 = torch.nn.Parameter(new_weight2)
        self.weight3 = torch.nn.Parameter(new_weight3)

    def forward(self, x):

        silu_input = einsum(self.weight1, x, "df dm, b l dm -> b l df")
        silu_output = silu_input * torch.sigmoid(silu_input)

        w3_output = einsum(self.weight3, x, "df dm, b l dm -> b l df")
        
        silu_w3 = silu_output * w3_output
        
        return einsum(self.weight2, silu_w3, "dm df, b l df -> b l dm")


class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) layer.

    Args
    ----
    theta : float
        Base used to generate inverse frequencies (e.g. 10_000).
    d_k : int
        Dimension of the key / query vectors (must be even).
    max_seq_len : int
        Maximum sequence length expected at inference / training time.
    device : torch.device | None
        Where to place the pre-computed sine / cosine tables.
    """
    def __init__(self,
                 theta: float,
                 max_seq_len: int,
                 d_k: int,
                 device,
                 dtype):
        
        super().__init__()
        
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        
        self.d_k = d_k
        
        # ---- pre-compute inverse frequencies ----
        # freq[k] = 1 / theta ** (2k / d_k)          (k = 0,1,…,d_k/2-1)
        freq = 1.0 / (theta ** (torch.arange(0, d_k,2, device=device).float() / d_k))

        # shape: (max_seq_len, d_k // 2)
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)

        # cache cos/sin; no gradients needed → persistent=False
        self.register_buffer('cos_cached', torch.cos(freqs),persistent=False) # persistent=False does not save to state_dict
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
    
    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
        ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to `x`.  Works with any batch shape prefix.
        """
        # Check if the last dimension matches d_k
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")
        
        # Gather the cached tables for the required positions
        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        # Split even / odd channels
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply the 2-D rotation to each pair
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos

        # Re-interleave
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out
    
class SoftmaxModule(torch.nn.Module):

    def __init__(self):

        super(SoftmaxModule, self).__init__()

    def forward(self, x, dim):


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
    
class AttentionModule(torch.nn.Module):

    def __init__(self):

        super(AttentionModule, self).__init__()

    def forward(self, Q, K, V, mask):

        d_k = Q.shape[-1]
        qk = einsum(Q, K, "b m d_k, b n d_k -> b m n")
        scaled_qk = qk / (d_k ** 0.5)
        
        final_mask = torch.zeros_like(mask, dtype=torch.float)
        final_mask = final_mask.masked_fill(mask == False, float('-inf'))
        scaled_qk = scaled_qk + final_mask

        softmax_module = SoftmaxModule()
        softmax_qk = softmax_module.forward(scaled_qk, dim=-1)
        
        attetion = einsum(softmax_qk, V, "b m n, b n d_k -> b m d_k")

        return attetion

       