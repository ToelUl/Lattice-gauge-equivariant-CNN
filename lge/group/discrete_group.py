import numpy as np
import torch

from .base import GroupBase


class CyclicGroup(GroupBase):
    """Implements a discrete cyclic group using PyTorch."""

    def __init__(self, order: int) -> None:
        """
        Initialize the cyclic group with a specified order.

        Args:
            order (int): The number of elements in the cyclic group (order > 1).

        Raises:
            ValueError: If order is not an integer greater than 1.
        """
        if not isinstance(order, int) or order <= 1:
            raise ValueError("The order of the cyclic group must be an integer greater than 1.")
        super().__init__(rep_dim=2, identity=torch.eye(2))
        self.order = order
        self.register_buffer(
            "group_elements",
            torch.linspace(
                0.0,
                2 * np.pi * (self.order - 1) / self.order,
                steps=self.order,
                device=self.identity.device,
                dtype=torch.float32,
            ),
        )

    def elements(self) -> torch.Tensor:
        """
        Return a tensor containing all elements of the cyclic group.

        Returns:
            torch.Tensor: A tensor of shape (order,) containing angles representing group elements.
        """
        return self.group_elements

    def product(self, h: torch.Tensor, h_prime: torch.Tensor) -> torch.Tensor:
        """
        Compute the group product of two cyclic group elements (addition modulo 2π).

        Args:
            h (torch.Tensor): Group element (angle) 1.
            h_prime (torch.Tensor): Group element (angle) 2.

        Returns:
            torch.Tensor: The sum modulo 2π.

        Raises:
            ValueError: If h and h_prime have incompatible shapes.
        """
        if h.shape != h_prime.shape:
            raise ValueError("Group elements h and h_prime must have the same shape.")
        return torch.remainder(h + h_prime, 2 * np.pi)

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of a cyclic group element (angle).

        Args:
            h (torch.Tensor): A group element (angle).

        Returns:
            torch.Tensor: The inverse angle modulo 2π.
        """
        return torch.remainder(-h, 2 * np.pi)

    def left_action_on_Rn(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the left group action of element h on a vector x in ℝ² using torch.matmul.

        Args:
            h (torch.Tensor): A group element (angle) or its matrix representation.
            x (torch.Tensor): A vector or batch of vectors in ℝ² with last dimension 2.

        Returns:
            torch.Tensor: The rotated vector(s).
        """
        rep = h if h.shape[-2:] == (2, 2) else self.matrix_representation(h)
        return torch.matmul(rep, x.unsqueeze(-1)).squeeze(-1)

    def right_action_on_Rn(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the right group action of the inverse of element h on a vector x in ℝ² using torch.matmul.

        Args:
            h (torch.Tensor): A group element (angle) or its matrix representation.
            x (torch.Tensor): A vector or batch of vectors in ℝ² with last dimension 2.

        Returns:
            torch.Tensor: The rotated vector(s).
        """
        rep = h if h.shape[-2:] == (2, 2) else self.matrix_representation(h)
        inv_rep = self.inverse(rep)
        return torch.matmul(x.unsqueeze(-1).transpose(-2, -1), inv_rep).squeeze(-1)

    def matrix_representation(self, h: torch.Tensor) -> torch.Tensor:
        """
        Obtain the 2D rotation matrix representation of a cyclic group element.

        Args:
            h (torch.Tensor): A group element (angle) or batch of angles.

        Returns:
            torch.Tensor: The rotation matrix/matrices with shape (..., 2, 2).
        """
        cos_h = torch.cos(h)
        sin_h = torch.sin(h)
        row1 = torch.stack([cos_h, -sin_h], dim=-1)
        row2 = torch.stack([sin_h, cos_h], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def determinant(self, h: torch.Tensor) -> torch.Tensor:
        """
        Calculate the determinant of the rotation matrix representation of a group element.

        Args:
            h (torch.Tensor): A group element (angle) or batch of angles.

        Returns:
            torch.Tensor: A tensor of ones (since rotation matrices have determinant 1).
        """
        return torch.ones_like(h)

    def normalize_group_elements(self, h: torch.Tensor) -> torch.Tensor:
        """
        Normalize a group element (angle) to the interval [-1, 1].

        Args:
            h (torch.Tensor): A group element (angle) or batch of angles.

        Returns:
            torch.Tensor: The normalized group element(s).
        """
        max_elem = 2 * np.pi
        return (2 * h / max_elem) - 1.0

    def random_element(
        self,
        sample_size: int = 1,
        generator: torch.Generator = None,
        apply_map: bool = True,
    ) -> torch.Tensor:
        """
        Generate a batch of random elements from the cyclic group.

        Args:
            sample_size (int, optional): Number of random elements to sample. Defaults to 1.
            generator (torch.Generator, optional): Random generator for reproducibility.
            apply_map (bool, optional): If True, return the rotation matrix representation;
                if False, return the raw angle. Defaults to True.

        Returns:
            torch.Tensor:
                - If apply_map is True: shape (sample_size, 2, 2).
                - Otherwise: shape (sample_size,).
        """
        indices = torch.randint(low=0, high=self.order, size=(sample_size,),
                                generator=generator, device=self.identity.device)
        angles = self.group_elements[indices]
        if apply_map:
            return self.matrix_representation(angles)
        else:
            return angles
