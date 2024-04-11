from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha as csdl
from typing import Union

import numpy as np

@set_properties(linear=False)
class Sqrt(ElementwiseOperation):
    """
    The elementwise square roots of the input tensor.
    """
    def __init__(self,x):
        super().__init__(x)
        self.name = 'sqrt'

    def compute_inline(self, x):
        return np.sqrt(x)
    

def sqrt(x:Union[Variable, np.ndarray]) -> Variable:
    """
    The elementwise square roots of the input tensor.

    Parameters
    ----------
    x : Variable or np.ndarray or float or int
        Input tensor to take the square root of.

    Returns
    -------
    Variable
        Elementwise square roots of the input tensor.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.sqrt(x)
    >>> y.value
    array([1.        , 1.41421356, 1.73205081])
    """
    op = Sqrt(variablize(x))
    return op.finalize_and_return_outputs()

class TestSqrt(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        x = csdl.Variable(name = 'x', value = x_val)

        compare_values = []
        # square root of a tensor variable
        s1 = csdl.sqrt(x)
        t1 = np.sqrt(x_val)
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # square root of a tensor constant
        s2 = csdl.sqrt(x_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # square root of scalar variables
        s3 = csdl.sqrt(2.0)
        t3 = np.array([np.sqrt(2.0)])
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.docstest(sqrt)

if __name__ == '__main__':
    test = TestSqrt()
    test.test_functionality()
    test.test_example()