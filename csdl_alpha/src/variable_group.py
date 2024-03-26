from typing import TypedDict
from csdl_alpha.src.graph.variable import Variable

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
        self.type_dict = {}

        self.define(*args)

    def define(self, *args):
        """
        User-defined method to enforce shape and type of the variables in the group.

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

    def declare_type(self, name, type):
        """
        Declares the type of a variable.

        Args:
            name (str): The name of the variable.
            type: The type of the variable.

        """
        self.type_dict[name] = type

    def __setitem__(self, key, value):
        """
        Sets the value of a variable.

        Args:
            key (str): The name of the variable.
            value: The value to be assigned to the variable.

        Raises:
            ValueError: If the shape of the value does not match the expected shape.
            ValueError: If the type of the value does not match the expected type.
        """
        if key in self.shape_dict:
            if value.shape != self.shape_dict[key]:
                raise ValueError(f"Shape mismatch. Expected {self.shape_dict[key]}, got {value.shape}")
            if type(value) != self.type_dict[key]:
                raise ValueError(f"Type mismatch. Expected {self.type_dict[key]}, got {type(value)}")
        super().__setitem__(key, value)

    def add_tag(self, tag):
        """
        Adds a tag to the group.

        Args:
            tag: The tag to be added.

        """
        for key in self.keys():
            if type(self[key]) == Variable or type(self[key]) == VariableGroup:
                self[key].add_tag(tag)