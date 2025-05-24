import torch.nn as nn
import torch
import math
from einops import rearrange, einsum

class Linear(nn.Module):
    """
    Linear transformation module
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        in_features: int, final dimension of the input
        out_features: int, final dimension of the output
        device: torch.device | None = None, Device to store the parameters on 
        dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype))
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        if not device:
            self.weight = self.weight.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        output = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return output


class Embedding(nn.Module):
    """
    embedding layer that maps integer token IDs into a vector space of dimension
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int, Size of the vocabulary
        embedding_dim: int, Dimension of the embedding vectors i.e. d_model
        device: torch.device | None = None, Device to store the parameters on 
        dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        if not device:
            self.weight = self.weight.to(device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]




if __name__ == "__main__":

    # layer = Linear(in_features=20, out_features=10)
    # print(layer.state_dict())

    embedding_layer = Embedding(num_embeddings=10, embedding_dim=3)
    weight = torch.linspace(1, 30, 30).reshape((10, 3))
    print(weight)
    embedding_layer.load_state_dict({"weight": weight})
    token_ids = torch.tensor([
        [5,2,3,8], 
        [1,9,6,1], 
    ], dtype=torch.long)
    output = embedding_layer(token_ids)
    print(output.shape)
    print(output)