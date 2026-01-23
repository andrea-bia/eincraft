from typing import Any, Optional, Tuple


class EinTenBaseTensor:
    """
    Represents a fundamental, atomic tensor variable in the eincraft system.

    Attributes:
        name (str): The unique name of the tensor.
        shape (Tuple[int, ...]): The shape of the tensor.
        constant (Optional[Any]): An optional constant value associated with the tensor.
    """

    def __init__(
        self, name: str, shape: Tuple[int, ...], constant: Optional[Any] = None
    ) -> None:
        """
        Initialize a new EinTenBaseTensor.

        Args:
            name: The name of the tensor. Must be a non-empty string.
            shape: The shape of the tensor.
            constant: Optional constant value (e.g., a numpy array) associated with this tensor.

        Raises:
            ValueError: If the name is not a valid string.
        """
        if not isinstance(name, str) or name == "":
            raise ValueError(f"Name '{name}' is not valid")
        self.name = name
        self.constant = constant
        self.shape = tuple(shape)

    def __eq__(self, other: Any) -> bool:
        """
        Check if two base tensors are equal.

        Two base tensors are considered equal if they have the same name and shape.

        Args:
            other: The object to compare with.

        Returns:
            True if other is an EinTenBaseTensor with the same name and shape, False otherwise.
        """
        if not isinstance(other, EinTenBaseTensor):
            return False
        if self.name != other.name:
            return False
        if self.shape != other.shape:
            return False
        return True
