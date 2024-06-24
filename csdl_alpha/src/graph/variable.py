from csdl_alpha.src.graph.node import Node
import numpy as np
from typing import Union
from csdl_alpha.utils.inputs import ingest_value, get_shape, process_shape_and_value, get_type_string

class Variable(Node):
    __array_priority__ = 1000
    dtype = np.float64
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

        shape, value = process_shape_and_value(shape, value)
        self._value = value
        self.shape = shape
        
        if len(shape) == 0:
            raise ValueError("Shape must have at least one dimension")
        if len(shape) == 1:
            self.size = shape[0]
        else:
            self.size = np.prod(shape)
        if name is not None:
            self.add_name(name)

        if tags is None:
            self.tags = []
        else:
            self.tags = tags

        self.post_init()

    @property
    def value(self):
        """The value of the variable used for inline evaluation"""
        return self._value
    
    @value.setter
    def value(self, value: Union[np.ndarray, float, int]):
        """Sets the value of a variable.

        Parameters
        ----------
        value : Union[np.ndarray, float, int]
            Value for the variable
        """
        self.set_value(value)

    def set_value(self, value: Union[np.ndarray, float, int]):
        """Sets the value of a variable.

        Parameters
        ----------
        value : Union[np.ndarray, float, int]
            Value for the variable
        """
        _, self._value = process_shape_and_value(self.shape, value)
    
    def post_init(self):
        pass

    def add_name(self, name: str):
        if self.name is None:
            self.name = name
        if self.recorder.active_namespace.prepend is not None:
            self.names.append(f'{self.recorder.active_namespace.prepend}.{name}')
        else:
            self.names.append(name)
    
    def add_tag(self, tag: str):
        self.tags.append(tag)

    def set_hierarchy(self, hierarchy: int):
        """
        
        Warnings
        --------
        This function should not need to be called by the user
        """

        self.hierarchy = hierarchy

    # TODO: add checks for parents
    # TODO: allow float and arrays
    # TODO: add  checks for shape of upper, scaler etc
    def set_as_design_variable(self, upper: float = None, lower: float = None, scaler: float = None):
        scaler = ingest_value(scaler)
        upper = ingest_value(upper)
        lower = ingest_value(lower)
        self.recorder._add_design_variable(self, upper, lower, scaler)

    def set_as_constraint(self, upper: float = None, lower: float = None, scaler: float = None):
        scaler = ingest_value(scaler)
        upper = ingest_value(upper)
        lower = ingest_value(lower)
        self.recorder._add_constraint(self, upper, lower, scaler)

    def set_as_objective(self, scaler: float = None):
        scaler = ingest_value(scaler)

        if self.size != 1:
            raise ValueError("Objective must be a scalar")
        self.recorder._add_objective(self, scaler)

    from csdl_alpha.src.operations.set_get.slice import Slice
    def set(self, slices:Slice, value:'VariableLike') -> 'Variable':
        """Sets a sliced selection of the variable to a new value. The slicing must be specified by a csdl Slice object.
        See examples for more information.

        Parameters
        ----------
        indices : Slice
            The indices to slice the variable by. See examples for more information.
        value : VariableLike
            The value to set the sliced selection of the variable to.

        Returns
        -------
        out: Variable
            A new variable that represents the original variable with the sliced selection set to the new value.

        Examples
        --------
        
        The set method creates a new variable with the sliced selection set to the new value. The original variable is not modified.

        >>> recorder = csdl.Recorder(inline = True)
        >>> recorder.start()
        >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
        >>> x1 = x.set(csdl.slice[0], 0.0)
        >>> x1.value
        array([0., 2., 3.])

        Use the csdl.slice slicer object when using slices.

        >>> x1 = x.set(csdl.slice[1:3], csdl.Variable(value = np.array([4.0, 5.0])))
        >>> x1.value
        array([1., 4., 5.])
        >>> x = csdl.Variable(value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        >>> x1 = x.set(csdl.slice[1, 1:3], csdl.Variable(value = np.array([10.0, 11.0])))
        >>> x1.value
        array([[ 1.,  2.,  3.],
               [ 4., 10., 11.]])

        The slicing conventions are identical to those in the __getitem__ method and broadcasting from a scalar is also supported.

        >>> x1 = x.set(csdl.slice[[0,1], [1,2]], csdl.Variable(value = np.array([10.0, 11.0])))
        >>> x1.value
        array([[ 1., 10.,  3.],
               [ 4.,  5., 11.]])
        >>> x1 = x.set(csdl.slice[0, 1:], 10.0)
        >>> x1.value
        array([[ 1., 10., 10.],
               [ 4.,  5.,  6.]])
        >>> x1 = x.set(csdl.slice[:, [0, 2]], 11.0)
        >>> x1.value
        array([[11.,  2., 11.],
               [11.,  5., 11.]])

        Slicing with CSDL variables is also supported in some cases when the slice size is constant.

        >>> start = csdl.Variable(value = 0)
        >>> x1 = x.set(csdl.slice[start:start+2, 1], 10.0)
        >>> x1.value
        array([[ 1., 10.,  3.],
               [ 4., 10.,  6.]])

        Get the same behaviour of in-place modification by returning the same variable (it is recommended to combine slices to one .set call when possible to reduce the number of operations).
        
        >>> x = x.set(csdl.slice[0,0], 10.0)
        >>> x = x.set(csdl.slice[1,0], 11.0)
        >>> x = x.set(csdl.slice[1,2], 12.0)
        >>> x.value
        array([[10.,  2.,  3.],
               [11.,  5., 12.]])
        """
        from csdl_alpha.src.operations.set_get.setindex import set_index
        from csdl_alpha.src.operations.set_get.slice import Slice
        from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice as loop_slice
        
        if isinstance(slices, Slice):
            return set_index(self, slices,value)
        else:
            raise TypeError(f"Use csdl.slice to index a variable. For example: x = x.set(csdl.slice[...], val). Type {get_type_string(slices)} given.")

    def save(self):
        """Sets variable to be saved
        """
        self._save = True

    def get(self, slices:Slice) -> 'Variable':
        """Similar to __getitem__ but only accepts a Slice object.

        Parameters
        ----------
        slices : Slice

        Returns
        -------
        out: Variable
        """
        from csdl_alpha.src.operations.set_get.getindex import get_index
        return get_index(self, slices)
    
    def __iter__(self):
        raise TypeError(f"{type(self).__name__} object is not iterable")

    def __getitem__(self, indices:Union[Slice, tuple[list[int], int, slice]]) -> 'Variable':
        """Returns a sliced selection of the variable as a new variable. The slicing can be specified by a csdl Slice object or a tuple of lists of integers, integers, and slices.
        The slicing rules are similar to Numpy's tensor indexing rules with some restrictions. See examples for more information.

        Parameters
        ----------
        indices : Union[Slice, tuple[list[int], int, slice]]
            The indices to slice the variable by. See examples for more information.

        Returns
        -------
        out: Variable
            a new variable that is a indexed selection of the original variable.

        Examples
        --------

        Integer indexing allows a selection of a single element in a dimension and removes that dimension in the output.
        
        >>> recorder = csdl.Recorder(inline = True)
        >>> recorder.start()
        >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
        >>> x[0].shape
        (1,)
        >>> x[0].value
        array([1.])
        >>> x = csdl.Variable(value = np.arange(6).reshape(2,3))
        >>> x[0].shape # removes the first dimension in the output
        (3,)
        >>> x[0].value
        array([0., 1., 2.])
        >>> x[1,2].shape # returns a single element
        (1,)
        >>> x[1,2].value
        array([5.])

        Slicing allows a selection of a range of elements in a dimension using slice notation and keeps that dimension in the output.

        >>> x[1:2].shape # keeps the first dimension in the output
        (1, 3)
        >>> x[1:2].value
        array([[3., 4., 5.]])
        >>> x[:].shape
        (2, 3)
        >>> np.all(x[:].value == x.value)
        True
        >>> x[1:2,:-1].shape
        (1, 2)
        >>> x[1:2,:-1].value
        array([[3., 4.]])

        Integer lists allows for selecting a coordinate of elements across multiple dimensions and compresses them to one one dimension.
        
        >>> x[[0,1]].shape
        (2, 3)
        >>> x[[0,1]].value
        array([[0., 1., 2.],
               [3., 4., 5.]])
        >>> x[[0,1],[0,1]].shape # outputs x[0,0] and x[1,1] in a 1D array
        (2,)
        >>> x[[0,1],[0,1]].value # outputs x[0,0] and x[1,1] in a 1D array
        array([0., 4.])

        All three types of indexing can be combined.

        >>> x = csdl.Variable(value = np.arange(24).reshape(4,2,3))
        >>> x[[0,1],1:].shape
        (2, 1, 3)
        >>> x[[0,1],1:].value
        array([[[ 3.,  4.,  5.]],
        <BLANKLINE>
               [[ 9., 10., 11.]]])
        >>> x[0,[0,1],[1,0]].shape
        (2,)
        >>> x[0,[0,1],[1,0]].value
        array([1., 3.])
        """
        from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice as loop_slice
        from csdl_alpha.src.operations.set_get.slice import Slice

        if isinstance(indices, Slice):
            return self.get(indices)
        else:
            return self.get(loop_slice[indices])


    def __add__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.add import add
        return add(self,other)
    
    def __radd__(self, other:'VariableLike') -> 'Variable':
        return self.__add__(other)

    def __mul__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.mult import mult
        return mult(self,other)

    def __rmul__(self, other:'VariableLike') -> 'Variable':
        return self.__mul__(other)

    def __neg__(self) -> 'Variable':
        from csdl_alpha.src.operations.neg import negate
        return negate(self)
    
    def __sub__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.sub import sub
        return sub(self, other)
    
    def __rsub__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.sub import sub
        return sub(other, self)

    def __truediv__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.division import div
        return div(self, other)

    def __rtruediv__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.division import div
        return div(other, self)
    
    def __pow__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.power import power
        return power(self, other)
    
    def __rpow__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.power import power
        return power(other, self)

    def __matmul__(self, other:'VariableLike') -> 'Variable':
        from csdl_alpha.src.operations.linalg.matmat import matmat
        return matmat(self, other)

    def __rmatmul__(self, other:'VariableLike') -> 'Variable':
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
        from csdl_alpha.src.operations.tensor.reshape import reshape
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
        from csdl_alpha.src.operations.tensor.reshape import reshape
        return reshape(self, (self.size,))

    def T(self: 'Variable')->'Variable':
        """ Invert the axes of a tensor. The shape of the output is the reverse of the input shape.

        Parameters
        ----------
        x : VariableLike
            
        Returns
        -------
        out: Variable

        Examples
        --------
        >>> recorder = csdl.Recorder(inline = True)
        >>> recorder.start()
        >>> x = csdl.Variable(value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        >>> csdl.transpose(x).value
        array([[1., 4.],
               [2., 5.],
               [3., 6.]])
        >>> x.T().value # equivalent to the above
        array([[1., 4.],
               [2., 5.],
               [3., 6.]])
        """
        from csdl_alpha.src.operations.tensor.transpose import transpose
        return transpose(self)

    def inner(self, other: 'Variable') -> 'Variable':
        """
        Inner product of two tensors x and y. 
        The result is a scalar of shape (1,).
        The input tensors must have the same shape.

        Parameters
        ----------
        self : Variable
            First input tensor.
        other : VariableLike
            Second input tensor.
        
        Returns
        -------
        out: Variable
            Scalar inner product of x and y.

        Examples
        --------
        >>> recorder = csdl.Recorder(inline = True)
        >>> recorder.start()
        >>> x = csdl.Variable(value = np.array([1, 2, 3]))
        >>> y = csdl.Variable(value = np.array([4, 5, 6]))
        >>> x.inner(y).value
        array([32.])
        >>> a = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
        >>> b = csdl.Variable(value = np.array([[5, 6], [7, 8]]))
        >>> a.inner(b).value
        array([70.])
        """
        from csdl_alpha.src.operations.tensor.inner import inner
        return inner(self, other)

    def _check_nlsolver_conflict(self):
        if hasattr(self, 'in_solver'):
            if self.in_solver is True:
                return True
            else:
                self.in_solver = True
        else:
            self.in_solver = True
        return False
    
class ImplicitVariable(Variable):
    pass

class SparseMatrix(Variable):
    def post_init(self):
        if len(self.shape) != 2:
            raise ValueError("SparseMatrix must have 2 dimensions")
        
class Constant(Variable):
    pass