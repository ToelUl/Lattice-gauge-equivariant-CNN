import torch
from typing import Union

from ..group import GroupBase, LieGroupBase


def compute_strides(dims, device, dtype=torch.long):
    """Compute the strides for linear indexing based on lattice dimension sizes.

    For a lattice with dimension sizes [d0, d1, ..., d_{D-1}], the stride for
    index i is:
        stride[i] = d_{i+1} * d_{i+2} * ... * d_{D-1}

    Args:
        dims (list[int]): A list of lattice dimension sizes.
        device (torch.device): The device on which the tensor is allocated.
        dtype (torch.dtype, optional): Data type of the strides. Defaults to torch.long.

    Returns:
        torch.Tensor:
            A tensor of shape (D,) containing the stride for each dimension.
    """
    D = len(dims)
    strides = torch.empty(D, device=device, dtype=dtype)
    prod = 1
    # Traverse dimensions in reverse to compute cumulative products
    for i in reversed(range(D)):
        strides[i] = prod
        prod *= dims[i]
    return strides


def generate_coordinates(dims, device):
    """Generate the coordinates of all points on the lattice using torch.meshgrid.

    Args:
        dims (list[int]): A list of lattice dimension sizes.
        device (torch.device): The device on which the coordinates are allocated.

    Returns:
        torch.Tensor:
            A tensor of shape (num_points, D), where num_points = d0 * d1 * ... * d_{D-1},
            and each row contains the integer coordinates of a point in the lattice.
    """
    # Create a range for each dimension
    ranges = [torch.arange(d, device=device) for d in dims]
    # Use meshgrid to get all coordinate combinations
    grid = torch.meshgrid(*ranges, indexing='ij')
    coords = torch.stack(grid, dim=-1)  # Shape: (*dims, D)
    coords = coords.reshape(-1, len(dims))  # Flatten to (num_points, D)
    return coords


def coords_to_linear_idx(coords, strides):
    """Convert multidimensional lattice coordinates to linear indices.

    Args:
        coords (torch.Tensor):
            A tensor of shape (..., D) where each row represents a lattice coordinate.
        strides (torch.Tensor):
            A tensor of shape (D,) containing the stride for each dimension,
            typically computed by `compute_strides`.

    Returns:
        torch.Tensor:
            A tensor of shape coords.shape[:-1] containing the linear indices corresponding
            to each row in coords.
    """
    # Multiply each dimension coordinate by the corresponding stride and sum
    return (coords * strides.view(1, -1)).sum(dim=-1)


def traverse_path(u, cumulative, coords, direction, steps, dims, strides, conjugate=False):
    """Traverse in a specified direction and update the cumulative matrix product and coordinates.

    This function is useful in Wilson loop calculations. Each forward or backward
    step either uses the raw link (forward) or the conjugate transpose of the link (backward).

    Args:
        u (torch.Tensor):
            Lattice link tensor of shape (B, N, D, rep_dim, rep_dim). Here,
            B is batch size, N is the total number of lattice points, D is the lattice dimension,
            and rep_dim is the dimension of the group representation matrix.
        cumulative (torch.Tensor):
            A tensor of shape (B, N, rep_dim, rep_dim) representing the accumulated
            product of link matrices up to the current step.
        coords (torch.Tensor):
            The current lattice coordinates of shape (B, N, D). Must be integer type.
        direction (int):
            The integer index representing the lattice direction along which to move.
        steps (int):
            The number of steps to move in the given direction.
        dims (list[int]):
            The size of each dimension of the lattice.
        strides (torch.Tensor):
            The strides tensor for converting coordinates to linear indices.
        conjugate (bool, optional):
            Whether to use the conjugate transpose of the link matrix for backward paths.
            Defaults to False.

    Returns:
        (torch.Tensor, torch.Tensor):
            - Updated cumulative matrix of shape (B, N, rep_dim, rep_dim).
            - Updated coordinates of shape (B, N, D), after wrapping with periodic boundaries.
    """
    B, N, _ = coords.shape

    for _ in range(steps):
        # Convert multi-dimensional coordinates to linear indices
        linear_idx = coords_to_linear_idx(coords.view(-1, coords.shape[-1]), strides).view(B, N)
        # Build batch indices to select the correct link from 'u'
        batch_indices = torch.arange(B, device=coords.device).view(B, 1).expand(B, N)
        # Extract the link matrix in the specified direction
        link = u[batch_indices, linear_idx, direction]  # Shape: (B, N, rep_dim, rep_dim)

        # If this is a backward path, take the conjugate transpose
        if conjugate:
            link = link.conj().transpose(-2, -1)

        # Update the cumulative matrix by matrix multiplication
        cumulative = torch.matmul(cumulative, link)

        # Update coordinates to move in the specified direction.
        # Use modular arithmetic to maintain periodic boundary conditions.
        delta = -1 if conjugate else 1
        coords[..., direction] = (coords[..., direction] + delta) % dims[direction]

    return cumulative, coords


def generate_wilson_loops(dims, u, group: Union["LieGroupBase", "GroupBase"], size=1, output_type='complex'):
    """Compute Wilson loops on a lattice using the provided link variables.

    The path for each Wilson loop is:
        1. Move forward along mu for `size` steps.
        2. Move forward along nu for `size` steps.
        3. Move backward (using conjugate transpose) along mu for `size` steps.
        4. Move backward (using conjugate transpose) along nu for `size` steps.

    This function supports vectorized batch calculations rather than iterating over
    each lattice point individually.

    Args:
        dims (list[int]):
            A list specifying the lattice dimension sizes, e.g., [Lx, Ly, Lz, ...].
        u (torch.Tensor):
            The lattice link tensor, whose shape depends on whether it is complex or real,
            and whether it has a batch dimension:

            - For a complex group with output_type='complex':
                Shape (N, D, rep_dim, rep_dim) or (B, N, D, rep_dim, rep_dim).
            - For a complex group with output_type='real':
                Shape (N, D, rep_dim, rep_dim, 2) or (B, N, D, rep_dim, rep_dim, 2)
                (the last dimension size 2 is for real and imaginary parts).
            - For a real group:
                Shape (N, D, rep_dim, rep_dim) or (B, N, D, rep_dim, rep_dim).

            Here, B is the batch size, N is the total number of lattice points (product of dims),
            D is the lattice dimension, and rep_dim is the dimension of the group representation.
        group (object):
            The group object with attributes:
                - rep_dim: The dimension of the group representation.
                - random_element(): A method that can be used to check if elements are real or complex.
        size (int, optional):
            The length of the square (plaquette) for the Wilson loop. Defaults to 1.
        output_type (str, optional):
            Applies only if the group elements are complex:
            - 'complex': Output is kept as a complex tensor.
            - 'real': Output is split into real and imaginary parts along the last dimension.
            Defaults to 'complex'.

    Returns:
        torch.Tensor:
            The shape of the output depends on whether the input had a batch dimension
            and on whether the group is complex or real.

            - For a complex group with output_type='complex' or a real group:
                (B, N, num_loops, rep_dim, rep_dim)
                or (N, num_loops, rep_dim, rep_dim) if the input was not batched.
            - For a complex group with output_type='real':
                (B, N, num_loops, rep_dim, rep_dim, 2)
                or (N, num_loops, rep_dim, rep_dim, 2) if not batched.

    Raises:
        ValueError: If dims is invalid, or u does not match expected dimensions, or
            if the output_type is not one of ['complex', 'real'].
    """
    # Validate 'dims'
    if not isinstance(dims, list) or not all(isinstance(d, int) and d > 0 for d in dims):
        raise ValueError("`dims` must be a list of positive integers.")
    # Validate 'u'
    if not torch.is_tensor(u):
        raise ValueError("`u` must be a torch.Tensor.")
    # if not isinstance(group, (GroupBase, LieGroupBase)):
    #     raise ValueError("`group` must be an instance of GroupBase or LieGroupBase.")
    # Validate output_type
    if output_type not in ["complex", "real"]:
        raise ValueError("`output_type` must be either 'complex' or 'real'.")

    device = u.device
    # Determine if the group elements are complex by sampling from 'group'
    is_complex_group = torch.is_complex(group.random_element())

    # Check expected shapes for batched / non-batched
    if is_complex_group:
        if output_type == "complex":
            expected_non_batched_dim = 4  # (N, D, rep_dim, rep_dim)
            expected_batched_dim = 5     # (B, N, D, rep_dim, rep_dim)
        else:  # output_type == 'real'
            expected_non_batched_dim = 5  # (N, D, rep_dim, rep_dim, 2)
            expected_batched_dim = 6     # (B, N, D, rep_dim, rep_dim, 2)
    else:
        expected_non_batched_dim = 4     # (N, D, rep_dim, rep_dim)
        expected_batched_dim = 5        # (B, N, D, rep_dim, rep_dim)

    # Add a batch dimension if necessary
    batch_added = False
    if u.dim() == expected_non_batched_dim:
        u = u.unsqueeze(0)
        batch_added = True
    elif u.dim() != expected_batched_dim:
        raise ValueError("Shape of `u` does not match the expected shape for this group and output_type.")

    # Unpack shape parameters
    B, N, D = u.shape[:3]
    rep_dim = u.shape[3]
    # num_loops = D * (D - 1) // 2  # Number of independent plaquettes in D dimensions

    # Convert to complex tensors if needed
    if is_complex_group:
        if output_type == "complex":
            # If u is not already complex, convert from real+imag parts
            if not torch.is_complex(u):
                if u.shape[-1] != 2:
                    raise ValueError("For a complex group, the last dimension of `u` should be 2 (real and imag).")
                u_complex = torch.complex(u[..., 0], u[..., 1])
                u = u_complex
        else:  # output_type == 'real'
            # We want internal calculations to use complex, then convert back
            if not torch.is_complex(u):
                if u.shape[-1] != 2:
                    raise ValueError("For a complex group, the last dimension of `u` should be 2 (real and imag).")
                u_complex = torch.complex(u[..., 0], u[..., 1])
                u = u_complex
    else:
        # For a real group, ensure that u is not complex
        if torch.is_complex(u):
            raise ValueError("For a real group, `u` must be real-valued.")

    # Compute strides for linear index conversion
    strides = compute_strides(dims, device=device)

    # Generate the coordinates for all lattice points
    coords0 = generate_coordinates(dims, device=device)
    # Expand coords for batch dimension: (B, N, D)
    coords0 = coords0.unsqueeze(0).expand(B, N, len(dims)).clone()

    # Store Wilson loop results for each pair of directions (mu, nu)
    wilson_loops = []
    for mu in range(D):
        for nu in range(mu + 1, D):
            # Initialize coordinates and cumulative matrix for each loop
            coords = coords0.clone()  # Shape: (B, N, D)
            cumulative = torch.eye(rep_dim, dtype=u.dtype, device=device)
            cumulative = cumulative.unsqueeze(0).unsqueeze(0).expand(B, N, rep_dim, rep_dim).clone()

            # Traverse the plaquette in four legs:
            # 1) Forward along mu
            cumulative, coords = traverse_path(u, cumulative, coords, direction=mu,
                                               steps=size, dims=dims, strides=strides, conjugate=False)
            # 2) Forward along nu
            cumulative, coords = traverse_path(u, cumulative, coords, direction=nu,
                                               steps=size, dims=dims, strides=strides, conjugate=False)
            # 3) Backward (conjugate transpose) along mu
            cumulative, coords = traverse_path(u, cumulative, coords, direction=mu,
                                               steps=size, dims=dims, strides=strides, conjugate=True)
            # 4) Backward (conjugate transpose) along nu
            cumulative, coords = traverse_path(u, cumulative, coords, direction=nu,
                                               steps=size, dims=dims, strides=strides, conjugate=True)

            # Add a loop dimension: (B, N, 1, rep_dim, rep_dim)
            wilson_loops.append(cumulative.unsqueeze(2))

    # Concatenate along loop dimension -> (B, N, num_loops, rep_dim, rep_dim)
    result = torch.cat(wilson_loops, dim=2)

    # If the group is complex and output_type='real', split into real/imag
    if is_complex_group and output_type == "real":
        result = torch.stack((result.real, result.imag), dim=-1)

    # If a batch dimension was artificially added, remove it
    if batch_added:
        result = result.squeeze(0)

    return result
