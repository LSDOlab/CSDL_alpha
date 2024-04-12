from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable 
from csdl_alpha.utils.inputs import variablize
from csdl_alpha.utils.typing import VariableLike

import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha.utils.error_utils as error_utils
import numpy as np

@set_properties(linear=True, diagonal_jacobian = True)
class Transpose(Operation):
    '''
    Elementwise addition of two tensors of the same shape.
    '''

    def __init__(self,x:Variable) -> Variable:
        super().__init__(x)
        self.name = 'transpose'
        self.set_dense_outputs((x.shape[::-1], ))

    def compute_inline(self, x):
        return np.transpose(x)

def transpose(x:VariableLike) -> Variable:
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
    x = variablize(x)
    return Transpose(x).finalize_and_return_outputs()

class TestTranspose(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()
        compare_values = []

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        x = csdl.Variable(value = x_val)

        x_val_large = np.arange(720).reshape((10,9,8))
        x_large = csdl.Variable(value = x_val_large)

        y = csdl.transpose(x)
        y_val = np.ones((1,))*3.0
        compare_values += [csdl_tests.TestingPair(y, y_val)]

        yT = x.T()
        compare_values += [csdl_tests.TestingPair(yT, y_val)]

        y2 = csdl.transpose(x_large)
        y2_val = np.transpose(x_val_large)
        compare_values += [csdl_tests.TestingPair(y2, y2_val)]

        y2 = x_large.T()
        compare_values += [csdl_tests.TestingPair(y2, y2_val)]

        # numpy arguments
        y = csdl.transpose(x_val)
        compare_values += [csdl_tests.TestingPair(y, y_val.flatten())]

        y2 = csdl.transpose(x_val_large)
        compare_values += [csdl_tests.TestingPair(y2, y2_val)]

        self.run_tests(compare_values = compare_values,)

    def test_docstring(self):
        self.docstest(transpose)

if __name__ == '__main__':
    test = TestTranspose()
    test.test_functionality()