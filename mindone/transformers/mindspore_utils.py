# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import List, Optional, Set, Tuple, Union, TypeVar, Any

from transformers.utils import logging

import mindspore as ms
from mindspore import nn, ops, Parameter
from mindspore.nn import Cell
from mindspore.common.initializer import Normal, Zero, initializer

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

logger = logging.get_logger(__name__)


def prune_linear_layer(layer: nn.Dense, index: ms.Tensor, dim: int = 0) -> nn.Dense:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`mindspore.nn.Dense`): The layer to prune.
        index (`mindspore.Tensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `mindspore.nn.Dense`: The pruned layer as a new layer with `requires_grad=True`.
    """
    w = layer.weight.index_select(dim, index).clone()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone()
        else:
            b = layer.bias[index].clone()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Dense(new_size[1], new_size[0], has_bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    ops.assign(new_layer.weight, w)
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        ops.assign(new_layer.bias, b)
        new_layer.bias.requires_grad = True
    return new_layer


class Conv1D(nn.Cell):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = ms.Parameter(initializer(Normal(0.02), [nx, nf], dtype=ms.float32), name="weight")
        self.bias = ms.Parameter(initializer(Zero(), [nf], dtype=ms.float32), name="bias")

    def construct(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x = ops.addmm(self.bias, x.view(-1, x.shape[-1]), self.weight)
        x = x.view(size_out)
        return x


def prune_conv1d_layer(layer: Conv1D, index: ms.Tensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer ([`~pytorch_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    w = layer.weight.index_select(dim, index).clone()
    if dim == 0:
        b = layer.bias.clone()
    else:
        b = layer.bias[index].clone()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0])
    new_layer.weight.requires_grad = False
    ops.assign(new_layer.weight, w)
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    ops.assign(new_layer.bias, b)
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer: Union[nn.Dense, Conv1D], index: ms.Tensor, dim: Optional[int] = None) -> Union[nn.Dense, Conv1D]:
    """
    Prune a Conv1D or linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`Union[torch.nn.Dense, Conv1D]`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Dense` or [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    if isinstance(layer, nn.Dense):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError(f"Can't prune layer of class {layer.__class__}")


def find_pruneable_heads_and_indices(
        heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], ms.Tensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = ops.ones((n_heads, head_size))
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).eq(1)
    index = ops.arange(len(mask))[mask].long()
    return heads, index

T_cell = TypeVar('T_cell', bound=Cell)

def weight_norm(cell: T_cell, name: str = 'weight', dim: int = 0) -> T_cell:
    r"""Apply weight normalization to a parameter in the given cell.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Cell.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparametrization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.
        * The weight is no longer recomputed once at cell forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the cell
          in question.

    Args:
        cell (Cell): containing cell
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original cell with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Dense(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(cell, name, dim)
    return cell


def _norm_except_dim(weight_v, pows, dim):
    r"""
    calculte g/||weight_v|| * weight_v method 
    """
    if dim == -1:
        return ops.norm(weight_v, pows)
    if dim == 0:
        w_shape_v = weight_v.shape[0]  # avoid macOS error
        output_size = (w_shape_v,) + (1,) * (weight_v.ndim - 1)
        return ops.norm(weight_v.view((w_shape_v, -1)), pows, 1).view(output_size)
    if dim == (weight_v.ndim - 1):
        output_size = (1,) * (weight_v.ndim - 1) + (weight_v.shape[weight_v.ndim - 1],)
        return ops.norm(weight_v.view((-1, weight_v.shape[weight_v.ndim - 1])), pows, 0).view(output_size)
    return _norm_except_dim(weight_v.swapaxes(0, dim), pows, dim).swapaxes(0, dim)


def _weight_norm(weight_v, weight_g, dim):
    r"""
    calculte weight_g/||weight_v|| * weight_v method 
    """
    return weight_v * (weight_g / _norm_except_dim(weight_v, 2, dim))


class WeightNorm:
    r"""
    The 'WeightNorm' class implements weight normalization for neural network cells. It provides methods to compute normalized weights, apply weight normalization to a cell, wrap a function, and remove
    weight bias from a cell. The class also contains an initializer for the name and dimension of the weight parameters, as well as a method to compute the weight using the normalized parameters. Additionally, it
    includes a method to remove the weight bias and a wrapper function for transposing cell_id to cell. 
    """
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, cell: Cell) -> Any:
        g = getattr(cell, self.name + '_g')
        v = getattr(cell, self.name + '_v')
        return Parameter(_weight_norm(v, g, self.dim))

    @staticmethod
    def apply(cell: Cell, name: str, dim: int) -> 'WeightNorm':
        for k, hook in cell._forward_pre_hook.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(cell, name)
        # if isinstance(weight, UninitializedParameter):
        #     raise ValueError(
        #         'The cell passed to `WeightNorm` can\'t have uninitialized parameters. '
        #         'Make sure to run the dummy forward before applying weight normalization')
        # remove w from parameter list
        del cell._params[name]

        # add g and v as new parameters and express w as g/||v|| * v
        cell.register_parameter(name + '_g', Parameter(_norm_except_dim(weight, 2, dim)))
        cell.register_parameter(name + '_v', Parameter(weight))
        setattr(cell, name, fn.compute_weight(cell))

        # recompute weight before every forward()
        cell.register_forward_pre_hook(fn)

        return fn

    def wrapper_func(self, cell, func):
        r"""
        wrapper_func where used to transpose cell_id to cell
        """

        def new_func(_, inputs):
            nonlocal cell
            return func(cell, inputs)

        return new_func

    def remove(self, cell: Cell) -> None:
        weight = self.compute_weight(cell)
        delattr(cell, self.name)
        del cell._params[self.name + '_g']
        del cell._params[self.name + '_v']
        setattr(cell, self.name, weight)

    def __call__(self, cell: Cell, inputs: Any) -> None:
        setattr(cell, self.name, self.compute_weight(cell))


def apply_chunking_to_forward(forward_fn, chunk_size, chunk_axis, *input_tensors: ms.Tensor):
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_axis`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., mindspore.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_axis (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[mindspore.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `mindspore.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
    """
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

     # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_axis]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_axis] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_axis]}"
                )

        if input_tensors[0].shape[chunk_axis] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_axis]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_axis] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, axis=chunk_axis) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return ops.cat(output_chunks, axis=chunk_axis)

    return forward_fn(*input_tensors)


def meshgrid(
    *tensors: Union[ms.Tensor, List[ms.Tensor]], indexing: Optional[str] = None
) -> Tuple[ms.Tensor, ...]:
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    return ops.meshgrid(*tensors, indexing=indexing)
