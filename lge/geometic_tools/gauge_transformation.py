import torch
from typing import Optional

from .utils import compute_strides, generate_coordinates, coords_to_linear_idx


def gauge_trans_to_gauge_link(
    u: torch.Tensor,
    global_group_element: Optional[torch.Tensor] = None,
    local_group_elements: Optional[torch.Tensor] = None,
    dims: Optional[list] = None
) -> torch.Tensor:
    """
    Applies a gauge transformation to lattice links based on lattice gauge theory.

    The gauge transformation is defined as:
        U'_μ(x) = Ω(x) · U_μ(x) · Ω†(x + ê_μ),
    where:
        - If only a global transformation g is provided, then Ω(x) = g and:
              U'_μ(x) = g · U_μ(x) · g†.
        - If only a local transformation ω(x) is provided, then Ω(x) = ω(x).
        - If both global and local transformations are provided, the effective transformation is:
              Ω(x) = g · ω(x),
          so that the transformed link becomes:
              U'_μ(x) = g · ω(x) · U_μ(x) · ω†(x + ê_μ) · g†.

    Args:
        u (torch.Tensor):
            Lattice link tensor. The shape can be either:
                - Non-batched: (N, D, rep_dim, rep_dim)
                - Batched: (B, N, D, rep_dim, rep_dim)
            where N is the total number of lattice points (the product of the lattice dimensions),
            D is the number of spatial dimensions, and rep_dim is the representation dimension of the group.
        global_group_element (Optional[torch.Tensor]):
            Global transformation matrix of shape (1, rep_dim, rep_dim).
        local_group_elements (Optional[torch.Tensor]):
            Local gauge transformation tensor. Its shape can be:
                - Non-batched: (N, 1, rep_dim, rep_dim)
                - Batched: (B, N, 1, rep_dim, rep_dim)
            The ordering of lattice points must be consistent with that of the link tensor.
        dims (Optional[list[int]]):
            List of lattice dimension sizes (e.g., [L_x, L_y, ...]). This must be provided when
            performing local transformations to compute the linear indices of neighboring lattice points
            using periodic boundary conditions.

    Returns:
        torch.Tensor:
            The gauge-transformed lattice link tensor with the same shape as the input u.

    Raises:
        ValueError: If there is a mismatch in batch dimensions between u and local_group_elements,
                    or if dims is not provided when performing local transformations.
    """
    # If neither global nor local transformations are provided, return the original tensor.
    if global_group_element is None and local_group_elements is None:
        return u

    device = u.device
    batch_added = False
    # If u is non-batched (shape: (N, D, rep_dim, rep_dim)), add a temporary batch dimension.
    if u.dim() == 4:
        u = u.unsqueeze(0)
        batch_added = True

    B, N, D, rep_dim, _ = u.shape

    # Construct the effective gauge transformation matrix Ω(x) for each lattice point.
    if global_group_element is not None and local_group_elements is not None:
        local_group_elements = local_group_elements.squeeze(dim=-3)
        # If both are provided and local_group_elements is non-batched, add a batch dimension.
        if local_group_elements.dim() == 3:
            local_group_elements = local_group_elements.unsqueeze(0)
        if local_group_elements.shape[0] != B:
            raise ValueError("Batch dimension mismatch: u and local_group_elements have different batch sizes.")
        # Broadcast the global transformation and combine: Ω(x) = g * ω(x)
        global_expanded = global_group_element.unsqueeze(0).expand(B, N, rep_dim, rep_dim)
        effective_local = torch.matmul(global_expanded, local_group_elements)
    elif global_group_element is not None:
        # Only global transformation provided; set Ω(x) = g (broadcast to all lattice points).
        effective_local = global_group_element.unsqueeze(0).expand(B, N, rep_dim, rep_dim)
    else:
        # Only local transformation provided.
        local_group_elements = local_group_elements.squeeze(dim=-3)
        if local_group_elements.dim() == 3:
            effective_local = local_group_elements.unsqueeze(0)  # Shape becomes (1, N, rep_dim, rep_dim)
            effective_local = effective_local.expand(B, N, rep_dim, rep_dim)
        else:
            if local_group_elements.shape[0] != B:
                raise ValueError("Batch dimension mismatch: u and local_group_elements have different batch sizes.")
            effective_local = local_group_elements

    # If only a global transformation is provided (i.e., local_group_elements is None),
    # apply the transformation using broadcasting for left and right matrix multiplications.
    if local_group_elements is None:
        g = global_group_element
        g_dag = g.conj().transpose(-2, -1) if torch.is_complex(g) else g.transpose(-2, -1)
        # Expand dimensions to match u (B, N, D, rep_dim, rep_dim)
        g_expanded = g.unsqueeze(0).unsqueeze(0).expand(B, N, D, rep_dim, rep_dim)
        g_dag_expanded = g_dag.unsqueeze(0).unsqueeze(0).expand(B, N, D, rep_dim, rep_dim)
        u = torch.matmul(g_expanded, u)
        u = torch.matmul(u, g_dag_expanded)
        if batch_added:
            u = u.squeeze(0)
        return u

    # For local transformations, dims must be provided to compute neighboring lattice indices.
    if dims is None:
        raise ValueError("When performing local gauge transformations, dims must be provided.")

    # Use helper functions to compute lattice coordinates and linear indices.
    strides = compute_strides(dims, device=device)
    coords = generate_coordinates(dims, device=device)  # Shape: (N, D)
    # For the batched case, expand coordinates to shape (B, N, D)
    coords = coords.unsqueeze(0).expand(B, N, len(dims)).clone()

    # Clone the original link tensor for updates.
    u_transformed = u.clone()
    # Apply local gauge transformation for each link direction μ.
    for mu in range(D):
        # Compute neighboring lattice coordinates: x + ê_μ (using periodic boundary conditions).
        neighbor_coords = coords.clone()
        neighbor_coords[..., mu] = (neighbor_coords[..., mu] + 1) % dims[mu]
        # Convert neighboring coordinates to linear indices (shape: (B, N)).
        neighbor_indices = coords_to_linear_idx(neighbor_coords.view(-1, len(dims)), strides).view(B, N)
        # Get the transformation matrix Ω(x) for each lattice point.
        Omega_x = effective_local  # Shape: (B, N, rep_dim, rep_dim)
        # Use advanced indexing to obtain the transformation matrix at the neighboring point, Ω(x+μ).
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        Omega_neighbor = effective_local[batch_indices, neighbor_indices]  # Shape: (B, N, rep_dim, rep_dim)
        Omega_neighbor_dag = (Omega_neighbor.conj().transpose(-2, -1)
                              if torch.is_complex(Omega_neighbor)
                              else Omega_neighbor.transpose(-2, -1))
        # Update the link: U'(x, μ) = Ω(x) * U(x, μ) * Ω†(x+μ)
        # Note: This uses vectorized matrix multiplication.
        temp = torch.matmul(Omega_x, u_transformed[:, :, mu])
        u_transformed[:, :, mu] = torch.matmul(temp, Omega_neighbor_dag)

    # Remove the temporary batch dimension if it was added.
    if batch_added:
        u_transformed = u_transformed.squeeze(0)
    return u_transformed


def gauge_trans_to_wilson_loop(
    w: torch.Tensor,
    global_group_element: Optional[torch.Tensor] = None,
    local_group_elements: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies a gauge transformation to Wilson loops based on lattice gauge theory.

    For a Wilson loop at lattice point x, the transformation is defined as:
        W'(x) = Ω(x) · W(x) · Ω(x)†,
    where:
        - If only a global transformation g is provided, then Ω(x) = g, and:
              W'(x) = g · W(x) · g†.
        - If only a local transformation ω(x) is provided, then:
              W'(x) = ω(x) · W(x) · ω(x)†.
        - If both global and local transformations are provided, the effective transformation is:
              Ω(x) = g · ω(x),
          so that:
              W'(x) = g · ω(x) · W(x) · ω(x)† · g†.

    Args:
        w (torch.Tensor):
            Wilson loop tensor. Its shape can be either:
                - Non-batched: (N, D, rep_dim, rep_dim)
                - Batched: (B, N, D, rep_dim, rep_dim)
            where N is the total number of lattice points, D is the number of loop per lattice points
            and rep_dim is the representation dimension.
        global_group_element (Optional[torch.Tensor]):
            Global transformation matrix of shape (1, rep_dim, rep_dim).
        local_group_elements (Optional[torch.Tensor]):
            Local gauge transformation tensor. Its shape can be:
                - Non-batched: (N, D, rep_dim, rep_dim)
                - Batched: (B, N, D, rep_dim, rep_dim)
            The ordering of lattice points must be consistent with that of the Wilson loop tensor.

    Returns:
        torch.Tensor:
            The gauge-transformed Wilson loop tensor with the same shape as the input w.

    Raises:
        ValueError: If there is a mismatch in batch dimensions between w and local_group_elements.
    """
    # If neither transformation is provided, return the original tensor.
    if global_group_element is None and local_group_elements is None:
        return w

    device = w.device
    batch_added = False
    # If the input tensor is non-batched (shape: (N, D, rep_dim, rep_dim)), add a temporary batch dimension.
    if w.dim() == 4:
        w = w.unsqueeze(0)  # New shape: (1, N, D, rep_dim, rep_dim)
        batch_added = True

    B, N, D, rep_dim, _ = w.shape

    # Construct the effective gauge transformation matrix Ω(x) for each lattice point.
    if global_group_element is not None and local_group_elements is not None:
        # If local_group_elements is non-batched, add a batch dimension.
        if local_group_elements.dim() == 4:
            local_group_elements = local_group_elements.unsqueeze(0)
        if local_group_elements.shape[0] != B:
            raise ValueError("Batch dimension mismatch: w and local_group_elements have different batch sizes.")
        # Broadcast the global transformation and combine: Ω(x) = g · ω(x)
        global_expanded = global_group_element.unsqueeze(0).unsqueeze(0).expand(B, N, D, rep_dim, rep_dim)
        effective = torch.matmul(global_expanded, local_group_elements)
    elif global_group_element is not None:
        # Only a global transformation is provided; set Ω(x) = g.
        effective = global_group_element.unsqueeze(0).unsqueeze(0).expand(B, N, D, rep_dim, rep_dim)
    else:
        # Only a local transformation is provided.
        if local_group_elements.dim() == 4:
            local_group_elements = local_group_elements.unsqueeze(0)
        if local_group_elements.shape[0] != B:
            raise ValueError("Batch dimension mismatch: w and local_group_elements have different batch sizes.")
        effective = local_group_elements

    # Compute the Hermitian conjugate (dagger) of the effective transformation matrix.
    effective_dag = effective.conj().transpose(-2, -1) if torch.is_complex(effective) else effective.transpose(-2, -1)

    # Apply the gauge transformation to the Wilson loop: W'(x) = Ω(x) * W(x) * Ω(x)†.
    w_transformed = torch.matmul(effective, w)
    w_transformed = torch.matmul(w_transformed, effective_dag)

    # If a batch dimension was temporarily added, remove it.
    if batch_added:
        w_transformed = w_transformed.squeeze(0)
    return w_transformed
