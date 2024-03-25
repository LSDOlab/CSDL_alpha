from typing import TypedDict

class VariableGroup(TypedDict):
    """
    Represents a group of variables with optionally defined shapes.

    Attributes:
        shape_dict (dict): A dictionary that maps variable names to their shapes.

    Methods:
        __init__(*args): Initializes the VariableGroup object.
        define(*args): Define the shapes of variables.
        declare_shape(name, shape): Declares the shape of a variable (used in define).
        __setitem__(key, value): Sets the value of a variable.

    """

    def __init__(self, *args):
        self.shape_dict = {}

        self.define(*args)

    def define(self, *args):
        """
        Defines the variables in the group.

        Args:
            *args: Variable names.

        """
        pass

    def declare_shape(self, name, shape):
        """
        Declares the shape of a variable.

        Args:
            name (str): The name of the variable.
            shape (tuple): The shape of the variable.

        """
        self.shape_dict[name] = shape

    def __setitem__(self, key, value):
        """
        Sets the value of a variable.

        Args:
            key (str): The name of the variable.
            value: The value to be assigned to the variable.

        Raises:
            ValueError: If the shape of the value does not match the expected shape.
        """
        if key in self.shape_dict:
            if value.shape != self.shape_dict[key]:
                raise ValueError(f"Shape mismatch. Expected {self.shape_dict[key]}, got {value.shape}")
        super().__setitem__(key, value)