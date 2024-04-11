from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import pytest
class VectorOuter(Operation):
    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'vec_outer'
        self.set_dense_outputs(((x.shape[0], y.shape[0]),))

    def compute_inline(self, x, y):
        import numpy as np
        return np.outer(x, y)

def outer(x:Variable,y:Variable)->Variable:
    """Outer product of two vectors x and y. The result is a matrix of shape (x.size, y.size).

    Parameters
    ----------
    x : Variable
        1D vector
    y : Variable
        1D vector

    Returns
    -------
    out: Variable
        2D matrix

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1, 2, 3]))
    >>> y = csdl.Variable(value = np.array([4, 5, 6]))
    >>> csdl.outer(x, y).value
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]])
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
    
    return VectorOuter(x, y).finalize_and_return_outputs()

class TestOuter(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(10)
        y_val = np.arange(10)+2.0
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)

        compare_values = []
        compare_values += [csdl_tests.TestingPair(csdl.outer(x,y), np.outer(x_val, y_val))]
        compare_values += [csdl_tests.TestingPair(csdl.outer(x_val,y), np.outer(x_val, y_val))]
        compare_values += [csdl_tests.TestingPair(csdl.outer(x,y_val), np.outer(x_val, y_val))]
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
            csdl.outer(x,y)
        with pytest.raises(ValueError):
            csdl.outer(x_val,y)
        with pytest.raises(ValueError):
            csdl.outer(x,y_val)

        with pytest.raises(ValueError):
            csdl.outer(y,x)
        with pytest.raises(ValueError):
            csdl.outer(y_val,x)     
        with pytest.raises(ValueError):
            csdl.outer(y,x_val)

        x_val = (np.arange(10)).reshape(2,5)
        y_val = (np.arange(2)+2.0).reshape(2)
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        with pytest.raises(ValueError):
            csdl.outer(x,y)
        with pytest.raises(ValueError):
            csdl.outer(x_val,y)
        with pytest.raises(ValueError):
            csdl.outer(x,y_val)

        with pytest.raises(ValueError):
            csdl.outer(y,x)
        with pytest.raises(ValueError):
            csdl.outer(y_val,x)
        with pytest.raises(ValueError):
            csdl.outer(x_val,y_val)

    def test_docstring(self):
        self.docstest(outer)

if __name__ == '__main__':
    test = TestOuter()
    test.test_functionality()
    test.test_errors()
    test.test_docstring()