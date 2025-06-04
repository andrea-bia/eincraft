# Tensor objects for the EinCraft library

class EinTenBaseTensor:
    """Basic tensor representation used in contractions."""

    def __init__(self, name: str, shape, constant=None) -> None:
        if not isinstance(name, str) or name == "":
            raise ValueError(f"Name '{name}' is not valid")
        self.name = name
        self.constant = constant
        self.shape = tuple(shape)

    def __eq__(self, other):
        if not isinstance(other, EinTenBaseTensor):
            return False
        if self.name != other.name:
            return False
        if self.shape != other.shape:
            return False
        return True

__all__ = ["EinTenBaseTensor"]
