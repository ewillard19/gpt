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
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 1 projection linéaire pour Q, K, V (chaque tête y piochera ses morceaux)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        # projection finale après concat des têtes
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.size()  # batch, seq len, dim

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # [B, num_heads, T, T]

        # === Ajoute ce bloc pour le masquage causal ===
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)  # [T, T], au-dessus de la diagonale = 1
        mask = mask.bool()
        scores = scores.masked_fill(mask, float('-inf'))
        # === Fin du bloc de masquage ===

        weights = torch.softmax(scores, dim=-1)
        context = weights @ V
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(context)
