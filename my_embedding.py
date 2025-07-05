from utils import *

class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # On initialise une matrice de poids (vecteurs aléatoires)
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.01)

    def forward(self, input_ids):
        # input_ids : [batch_size, seq_len]
        # On fait simplement un lookup : sélection des lignes par index
        return self.weight[input_ids]