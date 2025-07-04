import torch
import numpy as np
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor

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
        self._in_features = in_features
        self._out_features = out_features
        self._device = device
        self._dtype = dtype
        init_sigma = np.sqrt(2.0 / (in_features + out_features))
        self._weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size=(out_features, in_features), dtype=dtype, device=device),
                mean = 0.0,
                std = init_sigma,
                a = - 3 * init_sigma,
                b = 3 * init_sigma
                ),
            requires_grad = True
        )

    # def set_weights(self, W: Float[Tensor, "d_out d_in"]):
    #     assert W.shape == (self._out_features, self._in_features), 'Linear: set_weights:: shape of weights W inconsistent with object attributes'
    #     self._weights = W
    #     device = W.device
    #     dtype = W.dtype
    #     if self._device is None: self._device = device
    #     if self._dtype is None: self._dtype = dtype
    #     assert self._device == device, 'Linear: set_weights:: weights not on the same device with object'
    #     assert self._dtype == dtype, 'Linear: set_weights:: weights dtype inconsistent with object attributes'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the linear transformation to the input
        :param x:
        :return:
        '''
        output = einsum(x, self._weights, '... d_in, d_out d_in -> ... d_out')
        return output