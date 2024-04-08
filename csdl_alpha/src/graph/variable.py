from csdl_alpha.src.graph.node import Node
import numpy as np
from typing import Union
from csdl_alpha.utils.inputs import ingest_value, get_shape

class Variable(Node):
    __array_priority__ = 1000
    def __init__(
            self,
            shape: tuple = None, 
            name: str = None, 
            value: Union[np.ndarray, float, int] = None,  
            tags: list[str] = None, 
            hierarchy: int = None,
        ):
        
        self.hierarchy = hierarchy
        super().__init__()
        self.recorder._add_node(self)

        self.is_input = True
        self.is_implicit = False
        self.save = False
        self.names = []
        self.name = None

        value = ingest_value(value)
        shape = get_shape(shape, value)

        self.shape = shape
        if len(shape) == 1:
            self.size = shape[0]
        else:
            self.size = np.prod(shape)
        if name is not None:
            self.add_name(name)
        self.value = value
        if tags is None:
            self.tags = []
        else:
            self.tags = tags

    def add_name(self, name: str):
        if self.name is None:
            self.name = name
        if self.recorder.active_namespace.prepend is not None:
            self.names.append(f'{self.namespace.prepend}.{name}')
        else:
            self.names.append(name)
    
    def add_tag(self, tag: str):
        self.tags.append(tag)

    def set_hierarchy(self, hierarchy: int):
        self.hierarchy = hierarchy

    def set_value(self, value: Union[np.ndarray, float, int]):
        self.value = ingest_value(value)
        self.shape = get_shape(self.shape, self.value)

    def set_as_design_variable(self, upper: float = None, lower: float = None, scalar: float = None):
        if not self.is_input:
            raise Exception("Variable is not an input variable")
        self.recorder._add_design_variable(self, upper, lower, scalar)

    def set_as_constraint(self, upper: float = None, lower: float = None, scalar: float = None):
        if self.is_input:
            raise Exception("Variable is an input variable")
        self.recorder._add_constraint(self, upper, lower, scalar)

    def set_as_objective(self, scalar: float = None):
        if self.is_input:
            raise Exception("Variable is an input variable")
        self.recorder._add_objective(self, scalar)

    def set(self, slice, value:'Variable') -> 'Variable':
        from csdl_alpha.src.operations.set_get.setindex import set_index
        return set_index(self, slice, value)
    
    def __getitem__(self, slices) -> 'Variable':
        from csdl_alpha.utils.slice import _slice as slice
        from csdl_alpha.src.operations.set_get.getindex import get_index
        return get_index(self, slice[slices])

    def __add__(self, other):
        from csdl_alpha.src.operations.add import add
        return add(self,other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        from csdl_alpha.src.operations.mult import mult
        return mult(self,other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        from csdl_alpha.src.operations.neg import neg
        return neg(self)
    
    def __sub__(self, other):
        from csdl_alpha.src.operations.sub import sub
        return sub(self, other)
    
    def __truediv__(self, other):
        from csdl_alpha.src.operations.division import div
        return div(self, other)

    def __rtruediv__(self, other):
        from csdl_alpha.src.operations.division import div
        return div(other, self)

    def reshape(self, shape:tuple[int]):
        """
        doc strings
        """
        from csdl_alpha.src.operations.reshape import reshape
        return reshape(self, shape)

    def flatten(self):
        """
        doc strings
        """
        from csdl_alpha.src.operations.reshape import reshape
        return reshape(self, (self.size,))

class ImplicitVariable(Variable):
    pass