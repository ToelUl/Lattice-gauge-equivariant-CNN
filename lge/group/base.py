import torch

class GroupBase(torch.nn.Module):
    """Base class for implementing mathematical groups using PyTorch."""

    def __init__(self, rep_dim: int, identity: torch.Tensor) -> None:
        """
        Initialize the group with a given representation dimension and identity element.

        Args:
            rep_dim (int): Dimensionality of the group matrix representation.
            identity (torch.Tensor): The identity element of the group, expected to have shape (rep_dim, rep_dim).

        Raises:
            ValueError: If rep_dim is not a positive integer or identity tensor does not have shape (rep_dim, rep_dim).
        """
        super().__init__()
        if not isinstance(rep_dim, int) or rep_dim <= 0:
            raise ValueError("Representation dimension must be a positive integer.")
        self.rep_dim = rep_dim

        if identity.shape != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Identity must be a tensor of shape ({self.rep_dim}, {self.rep_dim}).")
        self.register_buffer("identity", identity)

    def elements(self) -> torch.Tensor:
        """
        Return a tensor containing all elements of the group.

        Returns:
            torch.Tensor: A tensor containing all group elements.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def product(self, h: torch.Tensor, h_prime: torch.Tensor) -> torch.Tensor:
        """
        Compute the group product of two group elements.

        Args:
            h (torch.Tensor): Group element 1.
            h_prime (torch.Tensor): Group element 2.

        Returns:
            torch.Tensor: The product h ⋅ h_prime.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of a group element.

        Args:
            h (torch.Tensor): A group element.

        Returns:
            torch.Tensor: The inverse of h.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def left_action_on_Rn(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the left group action of element h on a vector x in ℝⁿ.

        Args:
            h (torch.Tensor): A group element.
            x (torch.Tensor): A vector or batch of vectors in ℝⁿ.

        Returns:
            torch.Tensor: The transformed vector(s).

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def right_action_on_Rn(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the right group action of the inverse of element h on a vector x in ℝⁿ.

        Args:
            h (torch.Tensor): A group element.
            x (torch.Tensor): A vector or batch of vectors in ℝⁿ.

        Returns:
            torch.Tensor: The transformed vector(s).

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def matrix_representation(self, h: torch.Tensor) -> torch.Tensor:
        """
        Obtain the matrix representation of a group element h.

        Args:
            h (torch.Tensor): A group element.

        Returns:
            torch.Tensor: The matrix representation of h.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def determinant(self, h: torch.Tensor) -> torch.Tensor:
        """
        Calculate the determinant of the matrix representation of a group element h.

        Args:
            h (torch.Tensor): A group element.

        Returns:
            torch.Tensor: The determinant of the representation of h.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def normalize_group_elements(self, h: torch.Tensor) -> torch.Tensor:
        """
        Normalize group elements to the interval [-1, 1] for standardized input.

        Args:
            h (torch.Tensor): A group element.

        Returns:
            torch.Tensor: The normalized group element.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def random_element(
        self,
        sample_size: int = 1,
        generator: torch.Generator = None,
        apply_map: bool = True,
    ) -> torch.Tensor:
        """
        Generate a random group element.

        Args:
            sample_size (int, optional): Number of random elements to sample. Defaults to 1.
            generator (torch.Generator, optional): Random generator for reproducibility.
            apply_map (bool, optional): If True, return the group element after applying the
                group representation or exponential map. If False, return the raw group parameter.
                Defaults to True.

        Returns:
            torch.Tensor:
                - If apply_map is True: a tensor of shape (sample_size, rep_dim, rep_dim).
                - If False: the raw group element (e.g. angle or parameter vector).

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @staticmethod
    def convert_to_complex(x: torch.Tensor) -> torch.Tensor:
        """
        Convert a real tensor with separate real and imaginary parts to a complex tensor.

        Args:
            x (torch.Tensor): Tensor with shape (..., 2), where the last dimension stores [real, imag].

        Returns:
            torch.Tensor: A complex tensor.

        Raises:
            ValueError: If the last dimension of x is not 2.
        """
        if x.shape[-1] != 2:
            raise ValueError("Input tensor must have shape (..., 2).")
        real = x[..., 0]
        imag = x[..., 1]
        return torch.complex(real, imag)

    @staticmethod
    def convert_to_real(x: torch.Tensor) -> torch.Tensor:
        """
        Convert a complex tensor to a real tensor with separate real and imaginary parts.

        Args:
            x (torch.Tensor): A complex tensor.

        Returns:
            torch.Tensor: A real tensor with shape (..., 2) containing [real, imag].
        """
        real = x.real
        imag = x.imag
        return torch.stack((real, imag), dim=-1)


class LieAlgebraBase(torch.nn.Module):
    """Base class for Lie algebras."""

    def __init__(self, rep_dim: int, lie_alg_dim: int) -> None:
        """
        Initialize the Lie algebra with a representation dimension and Lie algebra dimension.

        Args:
            rep_dim (int): Dimension of the Lie algebra representation.
            lie_alg_dim (int): Dimension of the Lie algebra (number of generators).

        Raises:
            ValueError: If rep_dim or lie_alg_dim is not a positive integer.
        """
        super().__init__()
        if not isinstance(rep_dim, int) or rep_dim <= 0:
            raise ValueError("Representation dimension must be a positive integer.")
        if not isinstance(lie_alg_dim, int) or lie_alg_dim <= 0:
            raise ValueError("Lie algebra dimension must be a positive integer.")
        self.rep_dim = rep_dim
        self.lie_alg_dim = lie_alg_dim

    def generators(self) -> list:
        """
        Return the generators of the Lie algebra.

        Returns:
            list[torch.Tensor]: List of generator tensors.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @staticmethod
    def commutator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the commutator [a, b] = a * b - b * a.

        Args:
            a (torch.Tensor): Generator a.
            b (torch.Tensor): Generator b.

        Returns:
            torch.Tensor: The commutator [a, b].

        Raises:
            ValueError: If a and b do not have the same shape.
        """
        if a.shape != b.shape:
            raise ValueError("Generators a and b must have the same shape.")
        return torch.matmul(a, b) - torch.matmul(b, a)

    def structure_constants(self) -> torch.Tensor:
        """
        Return the structure constants f_{i,j,k} of the Lie algebra.

        Returns:
            torch.Tensor: A tensor of shape (lie_alg_dim, lie_alg_dim, lie_alg_dim).

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class LieGroupBase(torch.nn.Module):
    """Base class for Lie groups."""

    def __init__(self, rep_dim: int, identity: torch.Tensor, element_shape: tuple = None) -> None:
        """
        Initialize the Lie group with a given representation dimension and identity element.

        Args:
            rep_dim (int): Dimension of the group representation.
            identity (torch.Tensor): Identity element of the group.
            element_shape (tuple, optional): Expected shape of group elements. Defaults to (rep_dim, rep_dim).

        Raises:
            ValueError: If rep_dim is not a positive integer or identity does not match element_shape.
        """
        super().__init__()
        if not isinstance(rep_dim, int) or rep_dim <= 0:
            raise ValueError("Representation dimension must be a positive integer.")
        self.rep_dim = rep_dim

        if element_shape is None:
            element_shape = (self.rep_dim, self.rep_dim)
        self.element_shape = element_shape

        if identity.shape != self.element_shape:
            raise ValueError(f"Identity must be a tensor of shape {self.element_shape}.")
        self.register_buffer("identity", identity)

    def elements(self) -> torch.Tensor:
        """
        Return a tensor containing group elements.

        Returns:
            torch.Tensor: A tensor containing group elements.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Compute the group product of two elements.

        Args:
            g1 (torch.Tensor): Group element 1.
            g2 (torch.Tensor): Group element 2.

        Returns:
            torch.Tensor: The product g1 ⋅ g2.

        Raises:
            ValueError: If g1 and g2 have incompatible shapes.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of a group element.

        Args:
            g (torch.Tensor): Group element.

        Returns:
            torch.Tensor: The inverse of g.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def exponential_map(self, alge_elem: torch.Tensor) -> torch.Tensor:
        """
        Map an element from the Lie algebra to the Lie group via the exponential map.

        Args:
            alge_elem (torch.Tensor): An element in the Lie algebra.

        Returns:
            torch.Tensor: The corresponding group element.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def left_action_on_Cn(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the group left action on vectors in ℂⁿ.

        Args:
            g (torch.Tensor): Group element.
            x (torch.Tensor): Vector in ℂⁿ.

        Returns:
            torch.Tensor: The transformed vector.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def right_action_on_Cn(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the right group action of the inverse of g on a dual vector in ℂⁿ.

        Args:
            g (torch.Tensor): Group element.
            x (torch.Tensor): Dual vector in ℂⁿ.

        Returns:
            torch.Tensor: The transformed dual vector.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def determinant(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the determinant of a group element.

        Args:
            g (torch.Tensor): Group element.

        Returns:
            torch.Tensor: The determinant of g.

        Raises:
            ValueError: If g does not have the correct shape.
        """
        if g.shape[-len(self.element_shape):] != self.element_shape:
            raise ValueError(f"Group element must have shape (..., {self.element_shape}).")
        if self.element_shape == ():
            return torch.abs(g)
        elif self.element_shape == (1, 1):
            return g.squeeze(-1).squeeze(-1)
        else:
            return torch.det(g)

    def is_unitary(self, g: torch.Tensor) -> bool:
        """
        Check whether a group element is unitary.

        Args:
            g (torch.Tensor): Group element.

        Returns:
            bool: True if g is unitary, False otherwise.

        Raises:
            ValueError: If g does not have the correct shape.
        """
        if g.shape[-len(self.element_shape):] != self.element_shape:
            raise ValueError(f"Group element must have shape (..., {self.element_shape}).")
        if self.element_shape == ():
            return torch.allclose(torch.abs(g), torch.tensor(1.0, dtype=g.dtype, device=g.device), atol=1e-6)
        else:
            identity = self.identity.expand_as(g)
            return torch.allclose(g.conj().transpose(-2, -1) @ g, identity, atol=1e-6)

    def is_determinant_one(self, g: torch.Tensor) -> bool:
        """
        Check whether the determinant of a group element is 1.

        Args:
            g (torch.Tensor): Group element.

        Returns:
            bool: True if det(g) == 1, False otherwise.
        """
        det = self.determinant(g)
        det_abs = det.abs() if det.is_complex() else det
        return torch.allclose(det_abs, torch.tensor(1.0, dtype=det_abs.dtype, device=det.device), atol=1e-6)

    def random_element(
        self,
        sample_size: int = 1,
        generator: torch.Generator = None,
        apply_map: bool = True,
    ) -> torch.Tensor:
        """
        Generate a random group element.

        Args:
            sample_size (int, optional): Number of random elements to sample. Defaults to 1.
            generator (torch.Generator, optional): Random generator for reproducibility.
            apply_map (bool, optional): If True, return the group element after applying the
                group representation or exponential map. If False, return the raw group parameter.
                Defaults to True.

        Returns:
            torch.Tensor:
                - If apply_map is True: shape (sample_size, rep_dim, rep_dim).
                - Otherwise: raw group element (shape depends on the group).

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @staticmethod
    def convert_to_complex(x: torch.Tensor) -> torch.Tensor:
        """
        Convert a real tensor with separate real and imaginary parts to a complex tensor.

        Args:
            x (torch.Tensor): Tensor with shape (..., 2), where the last dimension stores [real, imag].

        Returns:
            torch.Tensor: A complex tensor.

        Raises:
            ValueError: If the last dimension of x is not 2.
        """
        if x.shape[-1] != 2:
            raise ValueError("Input tensor must have shape (..., 2).")
        real = x[..., 0]
        imag = x[..., 1]
        return torch.complex(real, imag)

    @staticmethod
    def convert_to_real(x: torch.Tensor) -> torch.Tensor:
        """
        Convert a complex tensor to a real tensor with separate real and imaginary parts.

        Args:
            x (torch.Tensor): A complex tensor.

        Returns:
            torch.Tensor: A real tensor with shape (..., 2) containing [real, imag].
        """
        real = x.real
        imag = x.imag
        return torch.stack((real, imag), dim=-1)
