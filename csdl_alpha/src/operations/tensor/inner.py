from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.src.operations.sum import sum as csdl_sum

class Inner(ComposedOperation):
    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'inner'

    def evaluate_composed(self, x, y):
        return evaluate_inner(x, y)
    
def evaluate_inner(x, y):
    out = csdl_sum(x*y)
    return out

def inner(x:VariableLike,y:VariableLike)->Variable:
    """
    Inner product of two tensors x and y. 
    The result is a scalar of shape (1,).
    The input tensors must have the same shape.

    Parameters
    ----------
    x : VariableLike
        First input tensor.
    y : VariableLike
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
    >>> csdl.inner(x, y).value
    array([32.])
    >>> a = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
    >>> b = csdl.Variable(value = np.array([[5, 6], [7, 8]]))
    >>> csdl.inner(a, b).value
    array([70.])
    """
    x = variablize(x)
    y = variablize(y)

    # checks:
    # - x and y must have the same shape
    if x.shape != y.shape:
        raise ValueError(f"Tesors x and y must have the same shape. {x.shape} != {y.shape}")
    
    return Inner(x, y).finalize_and_return_outputs()

class TestInner(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(10)
        y_val = np.arange(10)+2.0
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        a_val = np.array([[1, 2], [3, 4]])
        b_val = np.array([[5, 6], [7, 8]])
        a = csdl.Variable(value = a_val)
        b = csdl.Variable(value = b_val)

        compare_values = []
        compare_values += [csdl_tests.TestingPair(csdl.inner(x,y), np.inner(x_val, y_val).flatten())]
        compare_values += [csdl_tests.TestingPair(csdl.inner(x_val,y), np.inner(x_val, y_val).flatten())]
        compare_values += [csdl_tests.TestingPair(csdl.inner(x,y_val), np.inner(x_val, y_val).flatten())]
        compare_values += [csdl_tests.TestingPair(csdl.inner(a,b), np.sum(a_val * b_val).flatten())]

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