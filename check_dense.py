import torch
import torch.nn.functional as F
from sparse_attention.attention import attention

torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 128, 64) for _ in range(3))

mine = attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v)
print("max diff:", (mine - ref).abs().max().item())