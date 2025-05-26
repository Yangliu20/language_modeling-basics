import torch.nn as nn
import torch
import math
from einops import rearrange, einsum
from typing import Union

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


class rmsnorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int, Hidden dimension of the model
        eps: float = 1e-5 ,Epsilon value for numerical stability
        device: torch.device | None = None, Device to store the parameters on 
        dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model, ), dtype=dtype))
        if not device:
            self.weight = self.weight.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """

        ## upcast the input to torch.float32 before performing the normalization, to prevent overflow
        ## later downcast to the original dtype
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt( torch.pow(x, 2).sum(dim=2, keepdim=True) / self.d_model + self.eps ) # (batch_size, sequence_length, 1)
        result = x / rms * rearrange(self.weight, "d_model -> 1 1 d_model")

        return result.to(in_dtype)


class positionwise_feedforward(nn.Module):
    """
    position-wise feed-forward network with SwiGLU
    """
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            d_ff = int((d_model*8/3) // 64) * 64
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype))
        std = math.sqrt(2/(d_ff+d_model))
        nn.init.trunc_normal_(self.w1, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w2, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w3, mean=0, std=std, a=-3*std, b=3*std)
        if not device:
            self.w1 = self.w1.to(device)
            self.w2 = self.w2.to(device)
            self.w3 = self.w3.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim (batch_size, sequence_length, d_model)
        """
        def SiLU(x):
            return x * torch.sigmoid(x)
        output_1 = SiLU( einsum(x, self.w1, "... d_in, d_out d_in -> ... d_out") )
        output_3 = einsum(x, self.w3, "... d_in, d_out d_in -> ... d_out") 
        output_final = einsum( output_1 * output_3, self.w2, "... d_in, d_out d_in -> ... d_out")
        return output_final


class RotaryPositionalEmbedding(nn.Module):
    """
    A class that applies RoRE to the input tensor. 
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float, \theta value for the RoPE
        d_k: int, dimension of query and key vectors
        max_seq_len: int, Maximum sequence length that will be inputted 
        device: torch.device | None = None, Device to store the buffer on
        """
        super().__init__()
        self.d_k = d_k

        # i: token position; k: embedding index
        angle = einsum(torch.arange(max_seq_len), 1/torch.pow(theta, (torch.arange(d_k/2))*2/d_k), "token_id, embed_id -> token_id embed_id")
        print(angle)
        self.register_buffer("sin", torch.sin(angle), persistent=False)
        self.register_buffer("cos", torch.cos(angle), persistent=False)

        if not device:
            self.sin = self.sin.to(device)
            self.cos = self.cos.to(device)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x dim (..., seq_len, d_k)
        token_positions (..., seq_len)
        """
        seq_len = token_positions.shape[-1]

        ## dim: (..., seq_len, d_k/2)
        sin_slice = self.sin[token_positions, :]
        cos_slice = self.cos[token_positions, :]
        # print(sin_slice.shape)

        ## construct rotation matrix: (..., seq_len, d_k, d_k)
        # fill diagonal with cos
        rotation = torch.diag_embed(torch.repeat_interleave(cos_slice, 2, dim=-1))
        # fill upper diagonal with -sin
        rotation[..., torch.arange(0, self.d_k, 2), torch.arange(1, self.d_k, 2)] = -sin_slice
        # fill lower diagonal with sin
        rotation[..., torch.arange(1, self.d_k, 2), torch.arange(0, self.d_k, 2)] = sin_slice
        # print(rotation.shape)

        return einsum(rotation, x, "... seq_len d_k_out d_k_in, ... seq_len d_k_in -> ... seq_len d_k_out")


def softmax(x: torch.Tensor, dim: int):

    max_v = torch.max(x, dim=dim, keepdim=True).values
    num = torch.exp(x-max_v)
    den = torch.sum(num, dim=dim, keepdim=True)
    return num/den


def Attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Union[torch.Tensor, None] = None):
    """
    Scaled dot-product attention
    Input:
        queries: torch.Tensor (batch_size, ..., seq_len_q, d_k)
        keys: torch.Tensor (batch_size, ..., seq_len_k, d_k)
        values: torch.Tensor (batch_size, ..., seq_len_k, d_v)
        mask: torch.Tensor (seq_len_q, seq_len_k)
    """
    d_k = queries.shape[-1]
    q_dot_k_scaled = einsum(queries, keys, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / math.sqrt(d_k)
    if mask is not None:
        q_dot_k_scaled[..., ~mask] = -float('inf')
    softmax_q_dot_k_scaled = softmax(q_dot_k_scaled, dim=-1)
    output = einsum(softmax_q_dot_k_scaled, values, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention
    """
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        d_model: int Dimensionality of the Transformer block inputs. 
        num_heads: int Number of heads to use in multi-head self-attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.weight_query = nn.Parameter(torch.empty((self.num_heads*self.d_k, self.d_model), dtype=dtype))
        self.weight_key = nn.Parameter(torch.empty((self.num_heads*self.d_k, self.d_model), dtype=dtype))
        self.weight_value = nn.Parameter(torch.empty((self.num_heads*self.d_v, self.d_model), dtype=dtype))
        self.weight_output = nn.Parameter(torch.empty((self.d_model, self.num_heads*self.d_v), dtype=dtype))

        ## initialize
        std_query_key = math.sqrt(2/(self.num_heads*self.d_k + self.d_model))
        std_value_output = math.sqrt(2/(self.num_heads*self.d_v + self.d_model))
        nn.init.trunc_normal_(self.weight_query, mean=0, std=std_query_key, a=-3*std_query_key, b=3*std_query_key)
        nn.init.trunc_normal_(self.weight_key, mean=0, std=std_query_key, a=-3*std_query_key, b=3*std_query_key)
        nn.init.trunc_normal_(self.weight_value, mean=0, std=std_value_output, a=-3*std_value_output, b=3*std_value_output)
        nn.init.trunc_normal_(self.weight_output, mean=0, std=std_value_output, a=-3*std_value_output, b=3*std_value_output)

        if not device:
            self.weight_query = self.weight_query.to(device)
            self.weight_key = self.weight_key.to(device)
            self.weight_value = self.weight_value.to(device)
            self.weight_output = self.weight_output.to(device)
    
    def forward(self, x: torch.Tensor, positional_embedding_layer: Union[nn.Module, None] = None, token_positions: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        x dim: (batch_size, seq_len, d_model)
        positional_embedding_layer: nn.Module that applies Rotary Positional Embedding
        token_positions: torch.Tensor
        """

        ## compute the key, value, and query projections
        Q = einsum(self.weight_query, x, "d_out d_model, ... seq_len d_model -> ... seq_len d_out")
        K = einsum(self.weight_key, x, "d_out d_model, ... seq_len d_model -> ... seq_len d_out")
        V = einsum(self.weight_value, x, "d_out d_model, ... seq_len d_model -> ... seq_len d_out")

        ## rearrange
        Q_heads = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        K_heads = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        V_heads = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)

        ## apply rotary positional embedding to query and key vectors
        if positional_embedding_layer is not None:
            Q_heads = positional_embedding_layer(x=Q_heads, token_positions=token_positions)
            K_heads = positional_embedding_layer(x=K_heads, token_positions=token_positions)

        ## create causal mask
        seq_len_q = Q.shape[-2]
        seq_len_k = K.shape[-2]
        causal_mask = ~torch.triu(torch.ones((seq_len_q, seq_len_k), dtype=torch.bool), diagonal=1)

        ## apply attention function
        attention_heads = Attention(queries=Q_heads, keys=K_heads, values=V_heads, mask=causal_mask) # (... num_heads, seq_len, d_v)
        attention = rearrange(attention_heads, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        ## final output
        output = einsum(self.weight_output, attention, "d_model d_in, ... seq_len d_in -> ... seq_len d_model")

        return output


if __name__ == "__main__":

    # layer = Linear(in_features=20, out_features=10)
    # print(layer.state_dict())

    # embedding_layer = Embedding(num_embeddings=10, embedding_dim=3)
    # weight = torch.linspace(1, 30, 30).reshape((10, 3))
    # print(weight)
    # embedding_layer.load_state_dict({"weight": weight})
    # token_ids = torch.tensor([
    #     [5,2,3,8], 
    #     [1,9,6,1], 
    # ], dtype=torch.long)
    # output = embedding_layer(token_ids)
    # print(output.shape)
    # print(output)


    # pos_emb = RotaryPositionalEmbedding(theta=2., d_k=20, max_seq_len=7)
    # x = torch.randn((1, 7, 20))
    # token_positions = torch.arange(7).reshape(1, -1)
    # pos_emb(x=x, token_positions=token_positions)

    attention_layer = MultiHeadSelfAttention(d_model=20, num_heads=4)
    x = torch.randn((1, 7, 20))
    output = attention_layer(x)
    print(output.shape)