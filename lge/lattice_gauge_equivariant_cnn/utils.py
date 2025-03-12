import torch
import numpy as np
from typing import List, Tuple
from torch import Tensor
from typing import Union

from ..geometic_tools import generate_wilson_loops
from ..group import LieGroupBase, GroupBase, SU2Group


def dagger(w: Tensor) -> Tensor:
    """Compute the conjugate transpose (dagger) of a tensor.

    This function computes the conjugate transpose of the input tensor `w`.
    It assumes that for a tensor representing a complex number via a real-imag
    representation, the last dimension has size 2 (with index 0 being the real
    part and index 1 the imaginary part). Otherwise, if `w` has a complex dtype,
    it uses the built-in conjugation and transposition.

    Args:
        w (Tensor): A tensor representing a matrix or batch of matrices. For a
            real-imag representation, the last dimension must be of size 2 and
            there must be at least two preceding dimensions to interpret as
            matrix shape for transposition.

    Returns:
        Tensor: The conjugate transpose of `w`.

    Raises:
        ValueError: If the input tensor does not match the expected format.

    Example:
        >>> import torch
        >>> # Example using a complex tensor
        >>> w = torch.tensor([[1+2j, 3+4j]], dtype=torch.complex64)
        >>> dagger(w)
        tensor([[1.-2.j],
                [3.-4.j]], dtype=torch.complex64)

        >>> # Example using a real-imag representation tensor of shape (2, 2, 2)
        >>> # which we interpret as a 2x2 matrix, each entry having real/imag parts.
        >>> w_ri = torch.tensor([
        ...     [[1.0,  2.0], [3.0,  4.0]],
        ...     [[5.0,  6.0], [7.0,  8.0]]
        ... ])
        >>> w_ri.shape
        torch.Size([2, 2, 2])
        >>> out = dagger(w_ri)
        >>> out
        tensor([[[ 1., -2.],
                 [ 5., -6.]],

                [[ 3., -4.],
                 [ 7., -8.]]])
    """
    if w.dtype in (torch.complex32, torch.complex64, torch.complex128):
        # Complex tensor => use built-in conj + transpose
        return w.conj().transpose(-2, -1)
    elif w.shape[-1] == 2:
        # Real-imag representation
        return torch.stack(
            (w[..., 0].transpose(-2, -1),
             -w[..., 1].transpose(-2, -1)),
            dim=-1
        )
    else:
        raise ValueError('wrong input tensor for dagger')


def shift(a: Tensor, axis: int, orientation: int, dims: List[int]) -> Tensor:
    """Cyclically shift lattice indices along a specified axis.

    This function shifts the indices of a lattice represented by the tensor `a`
    along the given `axis`. The input tensor is expected to have shape
    [B, prod(dims), ...], where prod(dims) is the total number of lattice sites.
    The shift is performed using precomputed indices (via ``torch.roll`` on an index
    tensor) to avoid reshape and view operations.

    Args:
        a (Tensor): Input tensor with shape [B, prod(dims), ...].
        axis (int): The lattice axis along which to perform the shift.
        orientation (int): The shift direction (positive for forward, negative for backward).
        dims (List[int]): List of integers representing the lattice dimensions.

    Returns:
        Tensor: The tensor with shifted lattice indices.

    Example:
        >>> import torch
        >>> # For a lattice with dimensions [2, 2,] => total 4 sites.
        >>> a = torch.arange(8).reshape(2, 4)
        >>> a
        tensor([[0, 1, 2, 3],
                [4, 5, 6, 7]])
        >>> # We shift along axis=1 by +1 step:
        >>> shifted = shift(a, axis=0, orientation=1, dims=[2, 2])
        >>> shifted.shape
        torch.Size([2, 4])
        >>> shifted
        tensor([[2, 3, 0, 1],
                [6, 7, 4, 5]])
    """
    prod_dim = int(np.prod(dims))
    idx = torch.arange(prod_dim, device=a.device).reshape(dims)
    idx_shifted = torch.roll(idx, shifts=orientation, dims=axis)
    idx_flat = idx_shifted.flatten()
    return a[:, idx_flat, ...]


def complex_einsum(
    pattern: str,
    a: Tensor,
    b: Tensor = None,
    conj_a: int = 1,
    conj_b: int = 1
) -> Tensor:
    """Perform a complex Einstein summation on tensors with separated real and imaginary parts.

    This function applies the Einstein summation (einsum) operation on the real and imaginary
    components of the input tensor(s) separately. The parameters `conj_a` and `conj_b`
    determine whether to conjugate the corresponding tensor by multiplying its imaginary part
    by -1.

    Args:
        pattern (str): The einsum string rule (e.g., 'ij,jk->ik').
        a (Tensor): Input tensor with shape [..., 2], where the last dimension represents [real, imag].
        b (Tensor, optional): Second input tensor with shape [..., 2]. Defaults to None.
        conj_a (int, optional): Factor for conjugating `a` (1 for no conjugation, -1 for conjugation). Defaults to 1.
        conj_b (int, optional): Factor for conjugating `b` (1 for no conjugation, -1 for conjugation). Defaults to 1.

    Returns:
        Tensor: A tensor with shape [..., 2] representing the complex result.

    Example:
        >>> import torch
        >>> # a and b each has shape (1,1,2) => interpret as (batch=1, i=1, real/imag=2).
        >>> a = torch.tensor([[[1.0, 2.0]]])  # (1,1,2)
        >>> b = torch.tensor([[[3.0, 4.0]]])  # (1,1,2)
        >>> # We'll do a very simple sum over the 'i' index => pattern 'bi,bi->b'
        >>> # but to keep dimension naming consistent, let's do 'ijk,ijk->ij' if shapes matched.
        >>> # For demonstration, let's do 'bij,bij->b' ignoring that we only have i=1 dimension:
        >>> # Real part = (1*3 - 2*4) = -5
        >>> # Imag part = (1*4 + 2*3) = 10
        >>> out = complex_einsum('bi,bi->b', a, b)
        >>> out
        tensor([[-5., 10.]])
    """
    a_real, a_imag = a[..., 0], conj_a * a[..., 1]

    if b is not None:
        b_real, b_imag = b[..., 0], conj_b * b[..., 1]
        out_real = torch.einsum(pattern, a_real, b_real) - torch.einsum(pattern, a_imag, b_imag)
        out_imag = torch.einsum(pattern, a_real, b_imag) + torch.einsum(pattern, a_imag, b_real)
    else:
        out_real = torch.einsum(pattern, a_real)
        out_imag = torch.einsum(pattern, a_imag)

    return torch.stack((out_real, out_imag), dim=-1)


def repack_x(u: Tensor, w: Tensor) -> Tensor:
    """Repack the link tensor and Wilson loop tensor into a single tensor.

    The output tensor is formed by concatenating the link tensor `u` and the Wilson
    loop tensor `w` along the channel dimension. It is assumed that the channels (dim=2)
    consist of the link channels (n_dims) and the Wilson loop channels.

    Args:
        u (Tensor): Link tensor.
        w (Tensor): Wilson loop tensor.

    Returns:
        Tensor: The repacked tensor combining `u` and `w`.

    Example:
        >>> import torch
        >>> u = torch.ones((1, 10, 3, 3, 3, 2))
        >>> w = torch.zeros((1, 10, 5, 3, 3, 2))
        >>> out = repack_x(u, w)
        >>> out.shape
        torch.Size([1, 10, 8, 3, 3, 2])
    """
    return torch.cat((u, w), dim=2)


def unpack_x(x: Tensor, n_dims: int) -> Tuple[Tensor, Tensor]:
    """Unpack a tensor into link and Wilson loop tensors.

    Splits the input tensor `x` along the channel dimension (dim=2) into two parts:
    the first `n_dims` channels (link tensor `u`) and the remaining channels (Wilson loop tensor `w`).

    Args:
        x (Tensor): Input tensor of shape [batch_size, num_sites, n_dims + (# Wilson loops), rep_dim, rep_dim, 2].
        n_dims (int): The number of link channels.

    Returns:
        Tuple[Tensor, Tensor]: A tuple (u, w) where `u` is the link tensor and `w` is the Wilson loop tensor.

    Example:
        >>> import torch
        >>> x = torch.randn(1, 10, 7, 3, 3, 2)
        >>> u, w = unpack_x(x, n_dims=3)
        >>> u.shape, w.shape
        (torch.Size([1, 10, 3, 3, 3, 2]), torch.Size([1, 10, 4, 3, 3, 2]))
    """
    u = x[:, :, :n_dims]
    w = x[:, :, n_dims:]
    return u, w


def transport(u: Tensor, w: Tensor, axis: int, orientation: int, dims: List[int]) -> Tensor:
    """Transport Wilson loop elements along a specified lattice axis.

    This function shifts the Wilson loop tensor `w` along the given lattice axis
    using the corresponding link tensor `u`. The direction of transport is determined
    by the `orientation` parameter (positive for forward, negative for backward).

    Args:
        u (Tensor): Link tensor of shape [batch_size, num_sites, D, rep_dim, rep_dim, 2].
        w (Tensor): Wilson loop tensor of shape [batch_size, num_sites, (# Wilson loops), rep_dim, rep_dim, 2].
        axis (int): The lattice axis along which to perform transport.
        orientation (int): Transport direction; positive for forward and negative for backward.
        dims (List[int]): List representing the dimensions of the lattice.

    Returns:
        Tensor: The transported Wilson loop tensor.

    Example:
        >>> import torch
        >>> # Dummy data for a small lattice with dims=[2,2,], total 4 sites:
        >>> u = torch.randn(1, 4, 2, 3, 3, 2)
        >>> w = torch.randn(1, 4, 1, 3, 3, 2)
        >>> # Transport along axis=0 by +1 step:
        >>> transported = transport(u, w, axis=0, orientation=1, dims=[2,2])
        >>> transported.shape
        torch.Size([1, 4, 1, 3, 3, 2])
    """
    # Shift w along the negative orientation first
    w_shifted = shift(w, axis, -orientation, dims)
    ua = u.select(dim=2, index=axis)  # link for the chosen axis

    if orientation > 0:
        # forward transport: ua * w_shifted * ua^\dagger
        wt = complex_einsum('bxij,bxwjk->bxwik', ua, w_shifted)
        wt = complex_einsum('bxwij,bxkj->bxwik', wt, ua, conj_b=-1)
    else:
        # backward transport: ua^\dagger * w_shifted * ua
        ua_shifted = shift(ua, axis, +1, dims)
        wt = complex_einsum('bxji,bxwjk->bxwik', ua_shifted, w_shifted, conj_a=-1)
        wt = complex_einsum('bxwij,bxjk->bxwik', wt, ua_shifted)

    return wt


def plaquette(
    dims: List[int],
    u: Tensor,
    group: Union["LieGroupBase", "GroupBase"],
    size: int = 1,
    output_type: str = 'real'
) -> Tensor:
    """Generate plaquette Wilson loops for a given lattice.

    This function generates Wilson loops (plaquettes) on the lattice defined by `dims`
    using the link tensor `u` and the provided gauge group object. The plaquette size and
    output type (real or complex) can be specified.

    Note:
        The function `generate_wilson_loops(...)` must be defined elsewhere.

    Args:
        dims (List[int]): Lattice dimensions, e.g., [4, 8, 8, 8].
        u (Tensor): Link tensor with shape [batch_size, num_sites, D, rep_dim, rep_dim, 2].
        group (Union[LieGroupBase, GroupBase]): Gauge group object.
        size (int, optional): Size of the plaquette. Defaults to 1.
        output_type (str, optional): Output type, either 'real' or 'complex'. Defaults to 'real'.

    Returns:
        Tensor: Repacked tensor combining the link tensor and the generated Wilson loops.

    Example:
        >>> import torch
        >>> # Assume `group` is defined with attribute rep_dim
        >>> group = SU2Group(spin=1.0)
        >>> dims = [4, 8]
        >>> # Then total sites = 32
        >>> u = torch.randn(2, 32, 2, 3, 3, 2)
        >>> x = plaquette(dims, u, group, size=1, output_type='real')
        >>> x.shape
        # shape will be [batch_size, num_sites, D + (#plaquettes), rep_dim, rep_dim, 2]
        torch.Size([2, 32, 3, 3, 3, 2])
    """
    # Suppose generate_wilson_loops is implemented elsewhere
    w = generate_wilson_loops(dims, u, group, size=size, output_type=output_type)
    return repack_x(u, w)
