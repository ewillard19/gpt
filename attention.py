from utils import *

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = Q @ K.transpose(-2, -1) / self.scale
        weights = torch.softmax(scores, dim=-1)
        out = weights @ V
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        B, T, D = x.size()
        Q = self.q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # [B, num_heads, T, T]

        # Masque causal (futur)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Masque padding (batch x 1 x 1 x T)
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(pad_mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout_attn(weights)
        context = weights @ V
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(context)
