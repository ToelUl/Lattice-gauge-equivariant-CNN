import torch
from typing import Union

from .utils import compute_strides, generate_coordinates, traverse_path
from ..group import GroupBase, LieGroupBase


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
