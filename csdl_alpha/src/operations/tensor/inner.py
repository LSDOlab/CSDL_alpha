from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import pytest
from csdl_alpha.utils.typing import VariableLike

class VectorInner(Operation):
    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'vec_inner'
        self.set_dense_outputs(((1,),))

    def compute_inline(self, x, y):
        import numpy as np
        return np.inner(x, y)

def inner(x:VariableLike,y:VariableLike)->Variable:
    """Inner product of two vectors x and y. The result is a scalar of shape (1,).

    Parameters
    ----------
    x : Variable
        1D vector
    y : Variable
        1D vector

    Returns
    -------
    out: Variable
        1D scalar

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1, 2, 3]))
    >>> y = csdl.Variable(value = np.array([4, 5, 6]))
    >>> csdl.inner(x, y).value
    array([32])
    """
    x = variablize(x)
    y = variablize(y)

    # checks:
    # - x must be 1D
    # - y must be 1D
    # - x and y must have the same size
    if len(x.shape) != 1:
        raise ValueError(f"Vector x must be 1D, but has shape {x.shape}")
    if len(y.shape) != 1:
        raise ValueError(f"Vector y must be 1D, but has shape {y.shape}")
    if x.size != y.size:
        raise ValueError(f"Vectors x and y must have the same size. {x.size} != {y.size}")
    
    return VectorInner(x, y).finalize_and_return_outputs()

class TestInner(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(10)
        y_val = np.arange(10)+2.0
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)

        compare_values = []
        compare_values += [csdl_tests.TestingPair(csdl.inner(x,y), np.inner(x_val, y_val).flatten())]
        compare_values += [csdl_tests.TestingPair(csdl.inner(x_val,y), np.inner(x_val, y_val).flatten())]
        compare_values += [csdl_tests.TestingPair(csdl.inner(x,y_val), np.inner(x_val, y_val).flatten())]
        self.run_tests(compare_values = compare_values,)

    def test_errors(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x_val = np.arange(10)
        y_val = np.arange(9)+2.0
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        with pytest.raises(ValueError):
            csdl.inner(x,y)
        with pytest.raises(ValueError):
            csdl.inner(x_val,y)
        with pytest.raises(ValueError):
            csdl.inner(x,y_val)

        with pytest.raises(ValueError):
            csdl.inner(y,x)
        with pytest.raises(ValueError):
            csdl.inner(y_val,x)     
        with pytest.raises(ValueError):
            csdl.inner(y,x_val)

        x_val = (np.arange(10)).reshape(2,5)
        y_val = (np.arange(2)+2.0).reshape(2)
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        with pytest.raises(ValueError):
            csdl.inner(x,y)
        with pytest.raises(ValueError):
            csdl.inner(x_val,y)
        with pytest.raises(ValueError):
            csdl.inner(x,y_val)

        with pytest.raises(ValueError):
            csdl.inner(y,x)
        with pytest.raises(ValueError):
            csdl.inner(y_val,x)
        with pytest.raises(ValueError):
            csdl.inner(x_val,y_val)

    def test_docstring(self):
        self.docstest(inner)

if __name__ == '__main__':
    test = TestInner()
    test.test_functionality()
    test.test_errors()
    test.test_docstring()