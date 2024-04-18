from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.src.operations.tensor.expand import expand as csdl_expand

class Outer(ComposedOperation):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = 'outer'
        self.out_shape = x.shape + y.shape
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        rank1 = len(x.shape)
        rank2 = len(y.shape)
        out_rank = rank1 + rank2
        self.action1 = f'{alphabet[:rank1]}->{alphabet[:out_rank]}'
        self.action2 = f'{alphabet[rank1:out_rank]}->{alphabet[:out_rank]}'

    def evaluate_composed(self, x, y):
        return evaluate_outer(x, y, self.out_shape, self.action1, self.action2)
    
def evaluate_outer(x, y, out_shape, action1, action2):
    expand1 = csdl_expand(x, out_shape, action=action1)
    expand2 = csdl_expand(y, out_shape, action=action2)
    out = expand1 * expand2
    return out

def outer(x:VariableLike, y:VariableLike)->Variable:
    """
    Outer product of two tensors x and y. 
    The result is a tensor with shape (x.shape + y.shape).
    For example, if x has shape (3,2) and y has shape (4,5),
    the output will have shape (3,2,4,5).

    Parameters
    ----------
    x : VariableLike
        First input tensor.
    y : VariableLike
        Second input tensor.

    Returns
    -------
    out: Variable
        Outer product of x and y.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1, 2, 3]))
    >>> y = csdl.Variable(value = np.array([4, 5]))
    >>> csdl.outer(x, y).value
    array([[ 4.,  5.],
           [ 8., 10.],
           [12., 15.]])
    >>> z = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
    >>> csdl.outer(x, z).value
    array([[[ 1.,  2.],
            [ 3.,  4.]],
    <BLANKLINE>
           [[ 2.,  4.],
            [ 6.,  8.]],
    <BLANKLINE>
           [[ 3.,  6.],
            [ 9., 12.]]])
    """
    x = variablize(x)
    y = variablize(y)
    op = Outer(x, y)

    return op.finalize_and_return_outputs()

class TestOuter(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(10)
        y_val = np.arange(10)+2.0
        z_val = np.arange(10).reshape(2,5)
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        z = csdl.Variable(value = z_val)

        compare_values = []
        t1 = np.outer(x_val, y_val)
        compare_values += [csdl_tests.TestingPair(csdl.outer(x,y), t1)]
        compare_values += [csdl_tests.TestingPair(csdl.outer(x_val,y), t1)]

        t2 = np.tensordot(x_val, z_val, axes=0)
        compare_values += [csdl_tests.TestingPair(csdl.outer(x,z), t2)]
        compare_values += [csdl_tests.TestingPair(csdl.outer(x_val,z), t2)]
        self.run_tests(compare_values = compare_values,)

    def test_errors(self,):
        pass

    def test_docstring(self):
        self.docstest(outer)

if __name__ == '__main__':
    test = TestOuter()
    test.test_functionality()
    test.test_errors()
    test.test_docstring()