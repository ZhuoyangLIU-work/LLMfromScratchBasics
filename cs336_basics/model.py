import torch
from utils import init_xnt_std, softmax, scaled_dot_product_attention
from einops import rearrange, einsum, repeat
from jaxtyping import Float, Int
from torch.distributed.checkpoint import load_state_dict

'''
Global variables
'''
PARAMS_INIT_XNT_MEAN = 0.0
PARAMS_INIT_XNT_MSTD = 3
FFD_MODEL_DIM_RATIO = 8/3
HARDWARE_DIGITS = 64




class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        initialization of linear layer
        :param in_features: int, final dimension of input features
        :param out_features: int, final dimension of output features
        :param device: torch.device | None, device to store the parameters on
        :param dtype: torch.type | None, data type of the parameters
        init weights W of shape (out_features, in_features) using truncated normal distribution
        '''
        super().__init__()
        assert type(in_features) is int, 'linear: __init__: in_features must be int'
        assert type(out_features) is int, 'linear: __init__: out_features must be int'
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        init_sigma = init_xnt_std(in_features, out_features)
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size=(out_features, in_features), dtype=dtype, device=device),
                mean = PARAMS_INIT_XNT_MEAN,
                std = init_sigma,
                a = - PARAMS_INIT_XNT_MSTD * init_sigma,
                b = PARAMS_INIT_XNT_MSTD * init_sigma
                ),
            requires_grad = True
        )

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> torch.Tensor:
        '''
        Apply the linear transformation to the input
        :param x:
        :return:
        '''
        output = einsum(x, self.weights, '... d_in, d_out d_in -> ... d_out')
        return output

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        '''
        :param num_embeddings: int Size of the vocabulary
        :param embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        :param device: torch.device | None = None Device to store the parameters on
        :param dtype: torch.dtype | None = None Data type of the parameters
        '''
        assert type(num_embeddings) is int, 'Embedding: __init__: num_embeddings must be int'
        assert type(embedding_dim) is int, 'Embedding: __init__: embedding_dim must be int'
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        init_sigma = init_xnt_std(num_embeddings, embedding_dim)
        self.embedding_mtrx = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size=(num_embeddings, embedding_dim), dtype=dtype, device=device),
                mean = PARAMS_INIT_XNT_MEAN,
                std = init_sigma,
                a = - PARAMS_INIT_XNT_MSTD * init_sigma,
                b = PARAMS_INIT_XNT_MSTD * init_sigma
                ),
            requires_grad = True
        )

    def forward(self, token_ids: Float[torch.Tensor, "..."]) -> torch.Tensor:
        '''

        :param token_ids:
        :return:
        '''
        return self.embedding_mtrx[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device = None, dtype = None):
        '''
        :param d_model: int Hidden dimension of the model
        :param eps: float = 1e-5 Epsilon value for numerical stability
        :param device: torch.device | None = None Device to store the parameters on
        :param dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.device = device
        self.scale = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype),
        )

    def forward(self, x: Float[torch.Tensor, "batch seq hidden"]) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        :param x:
        :return:
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rmssqrt = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rmssqrt
        rms_result = einsum(x_norm, self.scale, "batch seq hidden, hidden -> batch seq hidden")
        return rms_result.to(in_dtype)

class SwiGLU(torch.nn.Module):
    '''
    position-wise feed-forward layer
    '''
    def __init__(self, d_model: int, d_ff: int | None = None, device = None, dtype = None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model

        if d_ff is None:
            target_dff = FFD_MODEL_DIM_RATIO * self.d_model
            d_ff = max(1, round(target_dff/HARDWARE_DIGITS)) * HARDWARE_DIGITS
        self.d_ff = d_ff
        
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device = device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device = device, dtype = dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> torch.Tensor:
        a = self.w1(x)
        silu = a * torch.sigmoid(a)
        linr = self.w3(x)
        return self.w2(einsum(silu, linr, "... d_ff, ... d_ff -> ... d_ff"))

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super(RoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_buffer(
            "_freq_cis_cache",
            RoPE._init_cache(max_seq_len, d_k, theta), persistent=False
        )

    @staticmethod
    def _init_cache(max_seq_len, d_k, theta):
        assert d_k % 2 == 0, 'RoPE: query and key vectors dimension must be an even number to apply RoPE'

        d = torch.arange(0, d_k, 2) / d_k
        query_slices = theta** (-d)
        token_positions = torch.arange(max_seq_len)

        freqs = einsum(token_positions, query_slices, "token_positions, theta_slices -> token_positions theta_slices")
        cos, sin = torch.cos(freqs), torch.sin(freqs)

        return torch.stack([cos, sin])

    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Float[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_k"]:
        '''

        :param x: input tensor of shape (..., seq_len, d_k)
        :param token_positions: token positions of shape (..., seq_len)
        :return: rotated input of shape (..., seq_len, d_k)
        '''
        # slicing the inputs tokens so that x1 contains leading items in each rotated pair and x2 contains the following item in each rotated pair
        x1, x2 = rearrange(x, "... (half_d xypair) -> xypair ... half_d", xypair = 2)

        # get cos sin
        cos, sin = self._freq_cis_cache[:, token_positions, :]

        x1r = cos * x1 - sin * x2
        x2r = sin * x1 + cos * x2

        result = rearrange(torch.stack([x1r, x2r]), "xypair ... half_d -> ... (half_d xypair)").contiguous()
        return result

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, position_encoder: RoPE | None = None, device = None, dtype = None):
        assert d_model % num_heads == 0, 'd_model should be divisible by num_heads'
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.wq = Linear(d_model, num_heads * self.d_k, device = device, dtype = dtype)
        self.wk = Linear(d_model, num_heads * self.d_k, device = device, dtype = dtype)
        self.wv = Linear(d_model, num_heads * self.d_v, device = device, dtype = dtype)
        self.wout = Linear(num_heads * self.d_v, d_model, device = device, dtype = dtype)

        self.position_encoder = position_encoder


    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"], token_positions: Int[torch.Tensor, "... seq_len"]|None = None) -> Float[torch.Tensor, "... seq_len h d_k"]:
        *b, seq_len, d_model = x.shape
        # assert d_model == self.d_model, "input not compatible with Module d_model"
        query = self.wq(x) # "... seq_len (num_heads dk)"
        key = self.wk(x)
        value = self.wv(x)

        query, key, value = (rearrange(T, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
                             for T in [query, key, value])

        # seq_len = seq_len * self.d_k

        if self.position_encoder is not None:

            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            query = self.position_encoder(query, token_positions)
            key = self.position_encoder(key, token_positions)

        attn_mask = ~torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        out = scaled_dot_product_attention(query, key, value, attn_mask)

        out = rearrange(out, "... h l d -> ... l (h d)")
        out = self.wout(out)

        return out





