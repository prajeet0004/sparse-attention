import torch

def attention(q, k, v, mask=None):
    """
    q, k, v : (batch, heads, seq_len, head_dim)
    mask    : (seq_len, seq_len) boolean, True = allowed to attend.
              None means dense with no masking.
    returns : (batch, heads, seq_len, head_dim)
    """
    scores = q @ k.transpose(-2, -1)
    scores = scores / (q.shape[-1] ** 0.5)

    if mask is not None:
      scores = scores.masked_fill(~mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v