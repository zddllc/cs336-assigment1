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
        qk = einsum(Q, K, "... m d_k, ... n d_k -> ... m n")
        scaled_qk = qk / (d_k ** 0.5)
        
        final_mask = torch.zeros_like(mask, dtype=torch.float)
        final_mask = final_mask.masked_fill(mask == False, float('-inf'))
        scaled_qk = scaled_qk + final_mask

        softmax_module = SoftmaxModule()
        softmax_qk = softmax_module.forward(scaled_qk, dim=-1)
        
        attetion = einsum(softmax_qk, V, "... m n, ... n d_k -> ... m d_k")

        return attetion


class MultiHeadAttentionModule(torch.nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len, rope_theta, device=None, dtype=None):

        super(MultiHeadAttentionModule, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k  # match d_k for simplicity
        self.max_seq_len = max_seq_len
        # self.q_proj, self.k_proj, self.v_proj, self.o_proj = [LinearModule(d_model, d_model, device=device, dtype=dtype) for _ in range(4)]

        self.attn = AttentionModule()
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", self.mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if rope_theta is None:
            rope_theta = 10000
        
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            max_seq_len=max_seq_len,
            d_k=self.d_k,
            device=device,
            dtype=dtype
        )
       
    def assign_weight(self, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):

        # self.q_proj.assign_weight(q_proj_weight)
        # self.k_proj.assign_weight(k_proj_weight)
        # self.v_proj.assign_weight(v_proj_weight)
        # self.o_proj.assign_weight(o_proj_weight)

        self.q_weight = torch.nn.Parameter(q_proj_weight)
        self.k_weight = torch.nn.Parameter(k_proj_weight)
        self.v_weight = torch.nn.Parameter(v_proj_weight)
        self.o_weight = torch.nn.Parameter(o_proj_weight)

    def forward(self, x, token_positions=None):

        # Project to multi-head Q, K, V
        # q,k,v = [rearrange(proj(x), "b s (h d) -> b h s d", h=self.num_heads) 
        #          for proj in [self.q_proj, self.k_proj, self.v_proj]]

        B, S, _ = x.shape

        q = einsum(self.q_weight, x, "... hd d, ... d ->  ... hd")
        k = einsum(self.k_weight, x, "... hd d, ... d ->  ... hd")
        v = einsum(self.v_weight, x, "... hd d, ... d ->  ... hd")

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE to Q and K if enabled
        if token_positions is not None: 
            q,k = self.rope(q, token_positions), self.rope(k, token_positions)

        #  Compute attention
        out = self.attn(q, k, v, mask=self.causal_mask[..., :S, :S])

        # Merge heads and project
        out = rearrange(out, "b h s d -> b s (h d)")
        o = einsum(self.o_weight, out, "dh m, b s m -> b s dh")
        return o
    
class TransformerBlockModule(torch.nn.Module):

    def __init__(self, d_model, n_heads, d_ff, max_seq_len, rope_theta, device=None, dtype=None):

        super(TransformerBlockModule, self).__init__()

        self.attn_module = MultiHeadAttentionModule(d_model, n_heads, max_seq_len, rope_theta, device=device, dtype=dtype)
        self.rmsnorm1 = RMSNormModule(d_model, device=device, dtype=dtype)
        self.ffn_module = FeedforwardModule(d_model, d_ff, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNormModule(d_model, device=device, dtype=dtype)

    def assign_weight(self, weights):

        self.attn_module.assign_weight(weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"], weights["attn.output_proj.weight"])
        self.ffn_module.assign_weight(weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"])
        self.rmsnorm1.assign_weight(weights["ln1.weight"])
        self.rmsnorm2.assign_weight(weights["ln2.weight"])

    def forward(self, x, token_positions=None):

        attn_out = self.attn_module.forward(self.rmsnorm1.forward(x), token_positions=token_positions)
        x = x + attn_out
        ffn_out = self.ffn_module.forward(self.rmsnorm2.forward(x))
        x = x + ffn_out
        return x
    
class TransformerLMModule(torch.nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=None, device=None, dtype=None):

        super(TransformerLMModule, self).__init__()
    
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta if rope_theta is not None else 10000

        self.embedding_layer = EmbeddingModule(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([TransformerBlockModule(d_model, num_heads, d_ff, context_length, rope_theta, "cpu", torch.float32) for _ in range(num_layers)])
        self.final_norm = RMSNormModule(d_model, device=device, dtype=dtype)
        self.lm_head = LinearModule(d_model, vocab_size, device=device, dtype=dtype)

        pass

    def assign_weight(self, weights):

        self.embedding_layer.assign_weight(weights["token_embeddings.weight"])
        
        for i in range(self.num_layers):
            q_key_str = "layers.%d.attn.q_proj.weight" % i
            k_key_str = "layers.%d.attn.k_proj.weight" % i
            v_key_str = "layers.%d.attn.v_proj.weight" % i
            o_key_str = "layers.%d.attn.output_proj.weight" % i
            ln1_key_str = "layers.%d.ln1.weight" % i
            w1_key_str = "layers.%d.ffn.w1.weight" % i
            w2_key_str = "layers.%d.ffn.w2.weight" % i
            w3_key_str = "layers.%d.ffn.w3.weight" % i
            ln2_key_str = "layers.%d.ln2.weight" % i
            
            layer_weights = {}
            layer_weights[".".join(q_key_str.split('.')[2:])] = weights[q_key_str]
            layer_weights[".".join(k_key_str.split('.')[2:])] = weights[k_key_str]
            layer_weights[".".join(v_key_str.split('.')[2:])] = weights[v_key_str]
            layer_weights[".".join(o_key_str.split('.')[2:])] = weights[o_key_str]
            layer_weights[".".join(ln1_key_str.split('.')[2:])] = weights[ln1_key_str]
            layer_weights[".".join(w1_key_str.split('.')[2:])] = weights[w1_key_str]
            layer_weights[".".join(w2_key_str.split('.')[2:])] = weights[w2_key_str]
            layer_weights[".".join(w3_key_str.split('.')[2:])] = weights[w3_key_str]
            layer_weights[".".join(ln2_key_str.split('.')[2:])] = weights[ln2_key_str]

            self.layers[i].assign_weight(layer_weights)

        self.final_norm.assign_weight(weights["ln_final.weight"])
        self.lm_head.assign_weight(weights["lm_head.weight"])
        
        pass

    def forward(self, token_ids):

        b, s = token_ids.shape
        x = self.embedding_layer.forward(token_ids)

        token_positions = torch.arange(s).unsqueeze(0).expand(b, s)

        for layer in self.layers:
            x = layer.forward(x, token_positions=token_positions)

        x = self.final_norm.forward(x)
        logits = self.lm_head.forward(x)

        return logits