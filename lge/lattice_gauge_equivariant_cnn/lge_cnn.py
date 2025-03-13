import torch
from typing import Union, List
from torch import nn, einsum, Tensor

from .utils import complex_einsum, dagger, transport, repack_x, unpack_x, plaquette
from ..group import LieGroupBase, GroupBase, SU2Group


class LConvBilin(nn.Module):
    """Bilinear Lattice Convolution Module.

    This module performs a bilinear lattice convolution operation on gauge field tensors,
    combining link and Wilson loop elements. It supports adjustable kernel size, dilation,
    and can optionally include unit elements for bias/residual connections as well as symmetric
    kernel ranges.

    Attributes:
        dims (list): Lattice size array.
        rep_dim (int): Dimension of the gauge group representation.
        in_channels (int): Number of input channels (e.g., number of 1x1 Wilson loops per lattice site).
        out_channels (int): Number of output channels (Wilson loops and links) per lattice site.
        kernel_size (int or list): Kernel size or range for convolution, which means the maximum size of Wilson loop.
        dilation (int): Dilation factor for the convolution.
        use_unit_elements (bool): Flag indicating whether to include unit elements (for bias/residual).
        kernel_range (list): Computed range for kernel indices.
        weight (Parameter): Learnable convolution weights.
        unit_matrix (Tensor): Unit matrix used for bias or residual connection.

    Example:
        >>> import torch
        >>> # Suppose we have a 2D lattice (dims=[8,8]) => total 64 sites
        >>> # For demonstration let's treat D=2 => in_channels=3 might be (2 link channels + 1 loop) etc.
        >>> dims = [8, 8]
        >>> module = LConvBilin(dims=dims, rep_dim=3, in_channels=1, out_channels=6, kernel_size=2)
        >>> # Suppose x has shape: [batch=2, num_sites=64, (1 channels, which means one Wilson loop per site), 3, 3, 2]
        >>> x = torch.randn(2, 64, 1, 3, 3, 2)
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 64, 6, 3, 3, 2])
    """

    def __init__(self, dims: List[int], rep_dim: int,
                 in_channels: int, out_channels: int, kernel_size: Union[int, List[int]],
                 dilation: int = 0,
                 use_unit_elements: bool = True):
        """
        Args:
            dims (List[int]): Lattice dimensions (e.g., [time, spatial dimensions...]).
            rep_dim (int): Dimension of the gauge group representation (e.g., SU(3) → 3).
            in_channels (int): Number of input channels (1x1 Wilson loops per lattice site).
            out_channels (int): Number of output channels.
            kernel_size (int or List[int]): Kernel size. If int, a symmetric
                range is computed . If type is list, must be a two-integer list [a, b]
                with a <= 0 and b >= 0.
            dilation (int, optional): Dilation factor. Defaults to 0.
            use_unit_elements (bool, optional): Whether to include unit elements (for bias/residual). Defaults to False.

        .. note::
            - The kernel size can be a single integer (e.g., 2) or a two-integer list [a, b].
            - If the kernel size is a single integer, the kernel range is [-1, 0, +1] * D.
            - If the kernel size is a two-integer list [a, b], the kernel range is [a, b] * D.
              The two-integer a and b should satisfy a <= 0 and b >= 0.
        """
        super(LConvBilin, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_dim = rep_dim
        self.use_unit_elements = use_unit_elements

        D = len(dims)

        self.kernel_range = None
        if isinstance(kernel_size, int):
            if kernel_size > 0:
                self.kernel_range = [[-(kernel_size - 1), kernel_size - 1]] * D
            else:
                raise ValueError(f"kernel_size should be a positive integer. Got {kernel_size}.")
        elif isinstance(kernel_size, list):
            if len(kernel_size) == 2:
                a, b = kernel_size
                if isinstance(a, int) and isinstance(b, int) and a <= 0 <= b:
                    self.kernel_range = [[a, b]] * D
                else:
                    raise ValueError(
                        f"kernel_size should be a list containing two"
                        f" integers a, b with a <= 0 and b >= 0. Got {kernel_size}."
                    )
            else:
                raise ValueError(
                    f"kernel_size should be a list containing two"
                    f" integers a, b with a <= 0 and b >= 0. Got {kernel_size}."
                )
        else:
            raise ValueError(f"Invalid kernel_size. Got {kernel_size}.")

        # Example code to compute a dimension for the weights
        w_in_size = self.in_channels
        if w_in_size != (D * (D - 1) // 2):
            w_in_size = w_in_size - D  # Just some logic from the original snippet

        t_w_size = w_in_size * (1 + sum([(abs(x[0]) + abs(x[1])) for x in self.kernel_range]))
        w_out_size = self.out_channels - D
        w_in_size = 2 * w_in_size
        t_w_size = 2 * t_w_size

        if self.use_unit_elements:
            w_in_size += 1
            t_w_size += 1

        self.weight = torch.nn.Parameter(torch.empty(w_out_size, w_in_size, t_w_size))
        torch.nn.init.xavier_uniform_(self.weight)

        # Construct the unit matrix
        self.unit_matrix_re = torch.eye(self.rep_dim)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the forward pass of the bilinear lattice convolution.

        Args:
            x (Tensor): Input tensor with shape
                [batch_size, num_sites, (n_dims + number of Wilson loops), rep_dim, rep_dim, 2].

        Returns:
            Tensor: The output tensor after convolution. The result is repacked with link and
                Wilson loop components.

        Example:
            >>> import torch
            >>> # Suppose we have a 2D lattice (dims=[8,8]) => total 64 sites
            >>> # For demonstration let's treat D=2 => in_channels=3 might be (2 link channels + 1 loop) etc.
            >>> dims = [8, 8]
            >>> module = LConvBilin(dims=dims, rep_dim=3, in_channels=1, out_channels=6, kernel_size=2)
            >>> # Suppose x has shape: [batch=2, num_sites=64, (1 channels, one Wilson loop per site), 3, 3, 2]
            >>> x = torch.randn(2, 64, 1, 3, 3, 2)
            >>> y = module(x)
            >>> y.shape
            torch.Size([2, 64, 6, 3, 3, 2])
        """
        # Unpack input
        u, w = unpack_x(x, len(self.dims))

        # Collect transported terms
        transported_terms = [w.clone()]
        for axis in range(len(self.dims)):
            for i, o in zip([0, 1], [-1, +1]):
                w_transport = w.clone()
                kernel_size = abs(self.kernel_range[axis][i])
                for _ in range(kernel_size):
                    for _ in range(self.dilation):
                        w_transport = transport(u, w_transport, axis=axis, orientation=o, dims=self.dims)
                transported_terms.append(w_transport)

        t_w = torch.cat(transported_terms, dim=2)

        # Conjugate expansion
        w_c, t_w_c = dagger(w), dagger(t_w)
        w = repack_x(w, w_c)
        t_w = repack_x(t_w, t_w_c)

        # Add unit elements if needed
        if self.use_unit_elements:
            unit_shape = list(w.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w.device).expand(unit_shape)
            w = repack_x(w, unit_matrix)
            t_w = repack_x(t_w, unit_matrix)

        # Perform complex multiplication
        w = complex_einsum('bxvij,bxwjk->bxvwik', w, t_w)
        w = einsum('uvw,bxvwijc->bxuijc', self.weight, w)

        return repack_x(u, w)


class LTrace(nn.Module):
    """Gauge-invariant trace layer.

    This layer extracts the trace of the Wilson loop components from the input tensor,
    yielding an output that is invariant under gauge transformations by discarding the link tensor.

    Example:
        >>> import torch
        >>> layer = LTrace(dims=[8, 8])
        >>> x = torch.randn(2, 64, 5, 3, 3, 2)  # e.g., 5 channels = 2 (links) + 3 (loops)
        >>> trace = layer(x)
        >>> trace.shape
        torch.Size([2, 64, 3, 2])
    """

    def __init__(self, dims: List[int]):
        """
        Args:
            dims (List[int]): Lattice dimensions.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        """Extract the trace of the Wilson loop tensor.

        Args:
            x (Tensor): Input tensor with shape
                [batch_size, num_sites, (n_dims + number of Wilson loops), rep_dim, rep_dim, 2].

        Returns:
            Tensor: Tensor containing the trace of the Wilson loops with shape [batch_size, num_sites, channels].
        """
        _, w = unpack_x(x, n_dims=len(self.dims))
        # einsum pattern 'bxwiic->bxwc' => trace over i dimension => rep_dim
        tr = einsum('bxwiic->bxwc', w)
        return tr

    def update_dims(self, dims: List[int]):
        """Update the lattice dimensions."""
        self.dims = dims


class Plaquette(nn.Module):
    """Plaquette layer module.

    This module maps the link tensor `u` to plaquette (Wilson loops) using the provided
    gauge group. It internally calls the `plaquette` function.

    Attributes:
        dims (List[int]): Lattice dimensions.
        group (Union[LieGroupBase, GroupBase]): Gauge group object.
        size (int): Size of the plaquette.
        output_type (str): Output type, either 'real' or 'complex'.

    Example:
        >>> import torch
        >>> # Assume `group` is defined with attribute rep_dim.
        >>> group = SU2Group(spin=1.0)
        >>> dims = [4, 8]
        >>> layer = Plaquette(dims, group, size=1, output_type='real')
        >>> u = torch.randn(2, 32, 2, 3, 3, 2)
        >>> x = layer(u)
        >>> x.shape
        # shape => [batch_size, 32, 2 + (#plaquettes), 3, 3, 2]
        torch.Size([2, 32, 3, 3, 3, 2])
    """

    def __init__(
        self,
        dims: List[int],
        group: Union[LieGroupBase, GroupBase],
        size: int = 1,
        output_type: str = 'real'
    ):
        """
        Args:
            dims (List[int]): Lattice dimensions.
            group (Union[LieGroupBase, GroupBase]): Gauge group object.
            size (int, optional): Size of the plaquette. Defaults to 1.
            output_type (str, optional): Output type ('real' or 'complex'). Defaults to 'real'.
        """
        super().__init__()
        self.dims = dims
        self.group = group
        self.size = size
        self.output_type = output_type

    def forward(self, u: Tensor) -> Tensor:
        """Map the link tensor to plaquette (Wilson loops).

        Args:
            u (Tensor): Link tensor with shape [batch_size, num_sites, D, rep_dim, rep_dim, 2].

        Returns:
            Tensor: Repacked tensor combining link and Wilson loop tensors.
        """
        return plaquette(
            dims=self.dims,
            u=u,
            group=self.group,
            size=self.size,
            output_type=self.output_type
        )


class LgeReLU(nn.Module):
    r"""ReLU-based gauge-equivariant nonlinearity.

    This module computes :math:`\mathrm{ReLU}(\mathrm{Re}[\mathrm{Tr}(W)]), W`,
    where :math:`W` denotes the Wilson loop part of the input tensor. The link components
    (often denoted :math:`u`) remain unchanged. By using the real part of the trace and
    scaling the Wilson loops via a ReLU function, the output remains gauge-equivariant,
    since any local gauge transformations cancel in the multiplication with its trace.

    Example:
        >>> import torch
        >>> dims = [8, 8]
        >>> module = LgeReLU(dims=dims)
        >>> # Suppose x has shape: [batch=2, num_sites=64, (2 + # loops), rep_dim, rep_dim, 2]
        >>> x = torch.randn(2, 64, 5, 3, 3, 2)
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 64, 5, 3, 3, 2])
    """

    def __init__(self, dims: List[int]):
        """Initialize the LgeReLU module.

        Args:
            dims (List[int]): Lattice dimensions, e.g. [Lx, Ly, ...].
        """
        super(LgeReLU, self).__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply ReLU-based scaling to the Wilson loop component.

        This computes :math:`\mathrm{ReLU}(\mathrm{Re}[\mathrm{Tr}(W)]), W` for
        the Wilson loop part of the input. The link component :math:`u` is passed through unchanged.

        Args:
            x (Tensor): Input tensor of shape
                ``[batch_size, num_sites, D + (#loops), rep_dim, rep_dim, 2]``.

        Returns:
            Tensor: The output tensor with the same shape, where the Wilson loops are scaled
            by the ReLU of their real trace.
        """
        # Unpack link (u) and Wilson loops (w)
        u, w = unpack_x(x, len(self.dims))

        # Compute ReLU(Re(Tr(W)))
        relu_tr_w = torch.relu(
            torch.einsum("bxwiic -> bxwc", w).select(-1, 0)
        )

        # Scale Wilson loops
        w_res = torch.einsum("bxw, bxwijc -> bxwijc", relu_tr_w, w)

        # Repack link + scaled Wilson loops
        return repack_x(u, w_res)


class TrNorm(nn.Module):
    """Trace-based normalization layer.

    This module normalizes the Wilson loop components in the input tensor based on
    their trace, making the output gauge-equivariant. Depending on the ``trnorm_on_abs``
    flag, it either uses the norm of the trace across channels or the mean of the trace
    across channels. A threshold is applied to avoid division by very small values.

    Attributes:
        dims (List[int]): Lattice dimensions, e.g. [Lx, Ly, ...].
        threshold (float): Minimum clip value for the normalization factor.
        trnorm_on_abs (bool): Whether to normalize by the norm of the trace values
            (True) or directly by the mean trace (False).

    Example:
        >>> import torch
        >>> dims = [8, 8]
        >>> layer = TrNorm(dims=dims, threshold=1e-5, trnorm_on_abs=True)
        >>> x = torch.randn(2, 64, 5, 3, 3, 2)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 64, 5, 3, 3, 2])
    """

    def __init__(
        self,
        dims: List[int],
        threshold: float = 1e-6,
        trnorm_on_abs: bool = True
    ):
        """
        Args:
            dims (List[int]): Lattice dimensions, e.g. [Lx, Ly, ...].
            threshold (float, optional): Minimum clip value for the normalization factor.
                Defaults to 1e-3.
            trnorm_on_abs (bool, optional): If True, normalizes by the norm of the trace
                values across channels; otherwise, normalizes by the mean trace. Defaults
                to True.
        """
        super().__init__()
        self.dims = dims
        self.threshold = threshold
        self.trnorm_on_abs = trnorm_on_abs

    def forward(self, x: Tensor) -> Tensor:
        """Apply trace-based normalization to the Wilson loop part of the input.

        This operation extracts the trace of the Wilson loops, computes a normalization
        factor (either by absolute norm or mean across channels), and divides by it to
        keep values within a stable range. The link component :math:`u` remains untouched.

        Args:
            x (Tensor): Input tensor of shape
                ``[batch_size, num_sites, D + (#loops), rep_dim, rep_dim, 2]``.

        Returns:
            Tensor: The output tensor with normalized Wilson loop components, having
            the same shape as the input.
        """
        # Unpack link (u) and Wilson loops (w)
        u, w = unpack_x(x, len(self.dims))

        # Compute the trace along rep_dim indices
        tr = torch.einsum("bxviic->bxvc", w)

        if self.trnorm_on_abs:
            # Normalize by the absolute norm of the trace across channels
            tr_abs = torch.linalg.norm(tr, dim=-1)
            tr_abs_mean = torch.mean(tr_abs, dim=2)
            norm_factor = torch.clamp(tr_abs_mean, min=self.threshold)
        else:
            # Normalize by the mean of the trace across channels
            tr_mean = torch.mean(tr, dim=2)
            tr_mean_abs = torch.linalg.norm(tr_mean, dim=-1)
            norm_factor = torch.clamp(tr_mean_abs, min=self.threshold)

        w_norm = torch.einsum(
            "bx, bxvijc->bxvijc",
            torch.reciprocal(norm_factor), w
        )

        # Repack link + normalized Wilson loops
        return repack_x(u, w_norm)


class SimpleLgeConvNet(nn.Module):
    """Lattice Gauge Equivariant Convolutional Network.

    This example network first generates 1x1 Wilson loops (via Plaquette),
    then applies multiple layers of bilinear lattice convolution (LConvBilin).
    Optionally applies LTrace for gauge-invariant output.

    Attributes:
        dims (List[int]): Lattice dimensions.
        D (int): Number of lattice dimensions (length of dims).
        group (Union[LieGroupBase, GroupBase]): Gauge group object.
        rep_dim (int): Dimension of the gauge group representation.
        plaquette_layer (Plaquette): Layer to generate plaquettes.
        input_conv (LConvBilin): Initial convolution layer.
        hidden_block (nn.Sequential): Sequential block of hidden layers + activations.
        gauge_invariant (bool): Whether the output is gauge invariant (via LTrace).

    Example:
        >>> import torch
        >>> # Suppose we have a 2D lattice dims=[8,8] => 64 sites
        >>> # and a gauge group with rep_dim=3
        >>> group = ...  # some group object
        >>> hidden_sizes = [10, 20]
        >>> kernel_size = 2
        >>> net = LgeConvNet(dims=[8,8],
        ...                  hidden_sizes=hidden_sizes,
        ...                  kernel_size=kernel_size,
        ...                  out_channels=30,
        ...                  group=group,
        ...                  gauge_invariant=True)
        >>> rep_dim = 3
        >>> # Link tensor shape => [batch=2, 64 sites, D=2, 3x3 complex]
        >>> u = torch.randn(2, 64, 2, rep_dim, rep_dim, 2)
        >>> output = net(u)
        >>> # If gauge_invariant=True, output shape => [batch=2, 64, out_channels, 2]
        >>> output.shape
        torch.Size([2, 64, 30, 2])
        >>> # If gauge_invariant=False, output shape => [batch=2, 64, out_channels, rep_dim, rep_dim, 2]
        torch.Size([2, 64, 30, 2, 2, 2])
    """

    def __init__(
        self,
        dims: List[int],
        hidden_sizes: List[int],
        kernel_size: Union[int, List[int]],
        out_channels: int,
        group: Union["LieGroupBase", "GroupBase"],
        gauge_invariant: bool = False,
    ):
        """
        Args:
            dims (List[int]): Lattice dimensions.
            hidden_sizes (List[int]): List of hidden layer sizes.
            kernel_size (Union[int, List[int]]): kernel sizes for each layer.
            out_channels (int): Number of output channels.
            group (Union[LieGroupBase, GroupBase]): Gauge group object.
            gauge_invariant (bool, optional): Whether to produce gauge-invariant output (via trace). Defaults to False.
        """
        super().__init__()
        self.dims = dims
        self.D = len(dims)
        self.group = group
        self.rep_dim = group.rep_dim

        # 1) Plaquette layer
        self.plaquette_layer = Plaquette(
            dims=self.dims,
            group=self.group,
            size=1,
            output_type='real'
        )

        # 2) First layer: from (D + #1x1 loops) to hidden_sizes[0]
        #   For a simple example, assume in_channels = # of link dims choose 2, etc.
        self.input_conv = LConvBilin(
            dims=self.dims,
            rep_dim=self.rep_dim,
            in_channels=self.D * (self.D - 1) // 2,
            out_channels=hidden_sizes[0],
            kernel_size=kernel_size
        )

        # 3) Build hidden layers
        layers = []
        all_sizes = hidden_sizes + [out_channels + self.D]
        for i in range(len(all_sizes) - 1):
            layers.append(
                LConvBilin(
                    dims=self.dims,
                    rep_dim=self.rep_dim,
                    in_channels=all_sizes[i],
                    out_channels=all_sizes[i+1],
                    kernel_size=kernel_size
                )
            )
            if i != len(all_sizes) - 2:
                pass
            else:
                # final layer => optional LTrace
                if gauge_invariant:
                    layers.append(LTrace(dims=self.dims))

        self.hidden_block = nn.Sequential(*layers)
        self.gauge_invariant = gauge_invariant

    def forward(self, u: Tensor) -> Tensor:
        """Perform a forward pass.

        Args:
            u (Tensor): Link tensor of shape [batch_size, num_sites, D, rep_dim, rep_dim, 2].

        Returns:
            Tensor: If `gauge_invariant` is True, shape => [batch_size, num_sites, out_channels, 2].
                    Otherwise => [batch_size, num_sites, out_channels, rep_dim, rep_dim, 2].
        """
        x = self.plaquette_layer(u)      # shape => [B, num_sites, D+(loops), rep_dim, rep_dim, 2]
        x = self.input_conv(x)
        x = self.hidden_block(x)

        if self.gauge_invariant:
            # Already traced inside LTrace if gauge_invariant is True
            return x
        else:
            _, w = unpack_x(x, n_dims=self.D)
            return w


class LgeConvNet(nn.Module):
    """Lattice Gauge Equivariant Convolutional Network.

    This example network first generates 1x1 Wilson loops (via :class:`Plaquette`),
    then applies multiple layers of bilinear lattice convolution (:class:`LConvBilin`).
    After each bilinear layer, optional activation (:class:`LgeReLU`) and optional
    normalization (:class:`TrNorm`) can be applied, depending on the flags
    ``use_act_fn`` and ``use_norm``.

    If ``gauge_invariant=True``, a final :class:`LTrace` layer is appended so that
    the network output is gauge invariant (i.e., by taking the trace of the Wilson loops).

    Args:
        dims (List[int]): Lattice dimensions.
        hidden_sizes (List[int]): List of hidden layer sizes.
        kernel_size (Union[int, List[int]]): Kernel size (or range) for each bilinear layer.
        out_channels (int): Number of output channels.
        group (Union[LieGroupBase, GroupBase]): Gauge group object.
        gauge_invariant (bool, optional): Whether to produce gauge-invariant output via trace.
            Defaults to False.
        use_act_fn (bool, optional): Whether to insert a :class:`LgeReLU` activation module
            after every :class:`LConvBilin`. Defaults to True.
        use_norm (bool, optional): Whether to insert a :class:`TrNorm` normalization module
            after every :class:`LConvBilin` (and after the optional activation if ``use_act_fn=True``).
            Defaults to True.
        threshold (float, optional): Threshold value for the normalization. Defaults to 1e-6.

    .. note::
        - When both ``use_act_fn`` and ``use_norm`` are True, the order after each
          :class:`LConvBilin` is :class:`LgeReLU` followed by :class:`TrNorm`.
        - The final :class:`LTrace` is appended only if ``gauge_invariant=True``.
        - The user can freely choose to omit activation or normalization by setting
          the respective flags to False.

    Example:
        >>> import torch
        >>> # Suppose we have a 2D lattice dims=[8,8] => 64 sites
        >>> # and a gauge group with rep_dim=3
        >>> group = ...
        >>> hidden_sizes = [10, 20]
        >>> kernel_size = 2
        >>> net = LgeConvNet(
        ...     dims=[8,8],
        ...     hidden_sizes=hidden_sizes,
        ...     kernel_size=kernel_size,
        ...     out_channels=30,
        ...     group=group,
        ...     gauge_invariant=True,
        ...     use_act_fn=True,
        ...     use_norm=True
        ... )
        >>> rep_dim = 3
        >>> # Link tensor shape => [batch=2, 64 sites, D=2, 3x3 complex]
        >>> u = torch.randn(2, 64, 2, rep_dim, rep_dim, 2)
        >>> output = net(u)
        >>> # If gauge_invariant=True, output shape => [batch=2, 64, out_channels, 2]
        >>> # Otherwise, output shape => [batch=2, 64, out_channels, rep_dim, rep_dim, 2]
        >>> output.shape
        torch.Size([2, 64, 30, 2])
    """

    def __init__(
        self,
        dims: List[int],
        hidden_sizes: List[int],
        kernel_size: Union[int, List[int]],
        out_channels: int,
        group: Union[LieGroupBase, GroupBase],
        gauge_invariant: bool = False,
        use_act_fn: bool = True,
        use_norm: bool = True,
        threshold: float = 1e-7,
    ):
        super().__init__()
        self.dims = dims
        self.D = len(dims)
        self.group = group
        self.rep_dim = group.rep_dim

        self.gauge_invariant = gauge_invariant
        self.use_act_fn = use_act_fn
        self.use_norm = use_norm
        self.threshold = threshold

        # 1) Plaquette layer
        self.plaquette_layer = Plaquette(
            dims=self.dims,
            group=self.group,
            size=1,
            output_type='real'
        )

        # 2) Input convolution (from a small number of channels to hidden_sizes[0])
        #    For demonstration purposes, in_channels is set to (D*(D-1))//2 or by similar logic.
        self.input_conv = LConvBilin(
            dims=self.dims,
            rep_dim=self.rep_dim,
            in_channels=self.D * (self.D - 1) // 2,
            out_channels=hidden_sizes[0],
            kernel_size=kernel_size
        )

        # 2a) Create a sequential container for the input convolution to include (optional) ReLU and Norm modules
        input_layers = []
        if self.use_act_fn:
            input_layers.append(LgeReLU(dims=self.dims))
        if self.use_norm:
            input_layers.append(TrNorm(dims=self.dims, threshold=self.threshold, trnorm_on_abs=True))
        self.after_input = nn.Sequential(*input_layers)

        # 3) Construct multiple LConvBilin layers based on hidden_sizes and out_channels.
        #    After each layer, if use_act_fn is True, append LgeReLU; if use_norm is True, append TrNorm.
        layers = []
        # For the final output, add self.D to keep the link channels (for later tracing if needed)
        all_sizes = hidden_sizes + [out_channels + self.D]
        num_hidden = len(all_sizes) - 1  # number of LConvBilin layers to construct

        for i in range(num_hidden):
            in_ch = all_sizes[i]
            out_ch = all_sizes[i + 1]

            # Create the LConvBilin layer for this block
            conv_layer = LConvBilin(
                dims=self.dims,
                rep_dim=self.rep_dim,
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size
            )
            layers.append(conv_layer)

            # Append activation if required
            if self.use_act_fn:
                layers.append(LgeReLU(dims=self.dims))

            # Append normalization if required
            if self.use_norm:
                layers.append(TrNorm(dims=self.dims))

        # 4) If gauge_invariant is desired, append the LTrace layer at the end.
        #    (In this case, the output shape is [B, num_sites, out_channels, 2])
        if self.gauge_invariant:
            layers.append(LTrace(dims=self.dims))

        # Encapsulate the entire hidden block into a nn.Sequential container.
        self.hidden_block = nn.Sequential(*layers)

    def forward(self, u: Tensor) -> Tensor:
        """Perform a forward pass through the network.

        1. Generate plaquettes via :class:`Plaquette`.
        2. Pass through the initial :class:`LConvBilin` (``self.input_conv``).
        3. If ``use_act_fn=True``, apply :class:`LgeReLU`; if ``use_norm=True``, apply :class:`TrNorm`.
        4. Pass through the subsequent block of layers (``self.hidden_block``), which may include
           additional :class:`LConvBilin`, :class:`LgeReLU`, :class:`TrNorm`, and an optional
           final :class:`LTrace` if ``gauge_invariant=True``.

        Args:
            u (Tensor): Link tensor with shape [batch_size, num_sites, D, rep_dim, rep_dim, 2].

        Returns:
            Tensor:
                - If ``gauge_invariant=True``, the output shape is [batch_size, num_sites, out_channels, 2].
                - Otherwise, the output shape is [batch_size, num_sites, out_channels, rep_dim, rep_dim, 2].
        """
        # 1) Generate 1x1 Wilson loops via the Plaquette layer.
        x = self.plaquette_layer(u)  # shape => [B, num_sites, (D + loops), rep_dim, rep_dim, 2]

        # 2) Pass through the input convolution.
        x = self.input_conv(x)

        # 2a) If the use_act_fn/use_norm are set, process through the corresponding modules.
        x = self.after_input(x)

        # 3) Pass through the hidden block (which may include multiple LConvBilin layers along with
        #    optional LgeReLU, TrNorm, and a final LTrace if gauge_invariant=True).
        x = self.hidden_block(x)

        # 4) If gauge_invariant is False (i.e., no LTrace was appended), manually extract and return the Wilson loops.
        if not self.gauge_invariant:
            _, w = unpack_x(x, n_dims=self.D)
            return w
        else:
            # If LTrace is applied, return x directly.
            return x
