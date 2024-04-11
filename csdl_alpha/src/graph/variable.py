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
        """
        Initialize a Variable object.

        Parameters
        ----------
        shape : tuple, optional
            The shape of the variable. If not provided, it will be inferred from the value.
        name : str, optional
            The name of the variable.
        value : Union[np.ndarray, float, int], optional
            The initial value of the variable.
        tags : list[str], optional
            A list of tags associated with the variable.
        hierarchy : int, optional
            The hierarchy level of the variable.

        Attributes
        ----------
        hierarchy : int
            The hierarchy level of the variable.
        shape : tuple
            The shape of the variable.
        size : int
            The size of the variable.
        names : list[str]
            A list of names associated with the variable.
        value : Union[np.ndarray, float, int]
            The value of the variable.
        tags : list[str]
            A list of tags associated with the variable.
        """
        
        self.hierarchy = hierarchy
        super().__init__()
        self.recorder._add_node(self)

        self._save = False
        self.names = []
        self.name = None

        value = ingest_value(value)
        shape = get_shape(shape, value)

        self.shape = shape
        if len(shape) == 0:
            raise ValueError("Shape must have at least one dimension")
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

    # TODO: add checks for parents
    def set_as_design_variable(self, upper: float = None, lower: float = None, scalar: float = None):
        # if not self.is_input:
        #     raise Exception("Variable is not an input variable")
        self.recorder._add_design_variable(self, upper, lower, scalar)

    def set_as_constraint(self, upper: float = None, lower: float = None, scalar: float = None):
        # if self.is_input:
        #     raise Exception("Variable is an input variable")
        self.recorder._add_constraint(self, upper, lower, scalar)

    def set_as_objective(self, scalar: float = None):
        # if self.is_input:
        #     raise Exception("Variable is an input variable")
        self.recorder._add_objective(self, scalar)

    from csdl_alpha.src.operations.set_get.slice import Slice
    def set(self, slices:Slice, value:'Variable') -> 'Variable':
        # return set_index(self, slice, value)
        from csdl_alpha.src.operations.set_get.setindex import set_index
        from csdl_alpha.src.operations.set_get.slice import Slice
        from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice as loop_slice
        
        if isinstance(slices, Slice):
            return set_index(self, slices,value)
        else:
            return set_index(self, loop_slice[slices], value)


    def save(self):
        """Sets variable to be saved
        """
        self._save = True

    def get(self, slices:Slice):
        from csdl_alpha.src.operations.set_get.getindex import get_index
        return get_index(self, slices)
    
    def __getitem__(self, slices) -> 'Variable':
        from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice as loop_slice
        from csdl_alpha.src.operations.set_get.slice import Slice

        if isinstance(slices, Slice):
            return self.get(slices)
        else:
            return self.get(loop_slice[slices])


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
        from csdl_alpha.src.operations.neg import negate
        return negate(self)
    
    def __sub__(self, other):
        from csdl_alpha.src.operations.sub import sub
        return sub(self, other)
    
    def __truediv__(self, other):
        from csdl_alpha.src.operations.division import div
        return div(self, other)

    def __rtruediv__(self, other):
        from csdl_alpha.src.operations.division import div
        return div(other, self)
    
    def __pow__(self, other):
        from csdl_alpha.src.operations.power import power
        return power(self, other)
    
    def __rpow__(self, other):
        from csdl_alpha.src.operations.power import power
        return power(other, self)

    def __matmul__(self, other):
        from csdl_alpha.src.operations.linalg.matmat import matmat
        return matmat(self, other)

    def __rmatmul__(self, other):
        from csdl_alpha.src.operations.linalg.matmat import matmat
        return matmat(other, self)

    def reshape(self, shape:tuple[int]):
        """Returns a reshaped version of the variable.

        Parameters
        ----------
        self : Variable

        Returns
        -------
        out: Variable

        Examples
        --------
        >>> recorder = csdl.Recorder(inline = True)
        >>> recorder.start()
        >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0, 4.0]))
        >>> csdl.reshape(x, (2,2)).value
        array([[1., 2.],
               [3., 4.]])
        >>> x.reshape((2,2)).value # same thing as above
        array([[1., 2.],
               [3., 4.]])
        """
        from csdl_alpha.src.operations.reshape import reshape
        return reshape(self, shape)

    def flatten(self: 'Variable')->'Variable':
        """Returns a 1D version of the variable.

        Parameters
        ----------
        self : Variable

        Returns
        -------
        out: Variable

        Examples
        --------
        >>> recorder = csdl.Recorder(inline = True)
        >>> recorder.start()
        >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0, 4.0]))
        >>> x.flatten().value # reshapes to 1 dimension
        array([1., 2., 3., 4.])
        """
        from csdl_alpha.src.operations.reshape import reshape
        return reshape(self, (self.size,))

class ImplicitVariable(Variable):
    pass