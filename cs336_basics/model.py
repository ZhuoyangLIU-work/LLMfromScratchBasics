import torch
import math
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch.distributed.checkpoint import load_state_dict

'''
Global variables
'''
PARAMS_INIT_XNT_MEAN = 0.0
PARAMS_INIT_XNT_MSTD = 3
FFD_MODEL_DIM_RATIO = 8/3
HARDWARE_DIGITS = 64

def init_xnt_std(d_in, d_out):
    return math.sqrt(2.0 / (d_in + d_out))

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
        assert max_seq_len % 2 == 0, 'RoPE: max_seq_len must be an even number'

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pairs = max_seq_len // 2
        positions = torch.arange(max_seq_len, device=device)
