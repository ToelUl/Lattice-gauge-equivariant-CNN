import torch


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
        # Build batch indices to select the correct link from 'u'
        batch_indices = torch.arange(B, device=coords.device).view(B, 1).expand(B, N)

        if not conjugate:
            # Convert multi-dimensional coordinates to linear indices
            linear_idx = coords_to_linear_idx(coords.view(-1, coords.shape[-1]), strides).view(B, N)
            # Extract the link matrix in the specified direction
            link = u[batch_indices, linear_idx, direction]  # Shape: (B, N, rep_dim, rep_dim)

            # Update the cumulative matrix by matrix multiplication
            cumulative = torch.matmul(cumulative, link)

            # Update coordinates to move in the specified direction.
            # Use modular arithmetic to maintain periodic boundary conditions.
            coords[..., direction] = (coords[..., direction] + 1) % dims[direction]

        # If this is a backward path, take the conjugate transpose
        else:
            # Update coordinates to move in the specified direction.
            # Use modular arithmetic to maintain periodic boundary conditions.
            coords[..., direction] = (coords[..., direction] - 1) % dims[direction]

            # Convert multi-dimensional coordinates to linear indices
            linear_idx = coords_to_linear_idx(coords.view(-1, coords.shape[-1]), strides).view(B, N)
            link = u[batch_indices, linear_idx, direction]
            link = link.conj().transpose(-2, -1)

            # Update the cumulative matrix by matrix multiplication
            cumulative = torch.matmul(cumulative, link)

    return cumulative, coords
