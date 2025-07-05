from utils import *
from attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, 4)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.2)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x
