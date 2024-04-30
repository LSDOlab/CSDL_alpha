from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike

import numpy as np

class Exp(ComposedOperation):
    '''
    Elementwise exponential of the input tensor or scalar.
    '''
    def __init__(self, x):
        super().__init__(x)
        self.name  = 'exp'

    def evaluate_composed(self, x):
        return evaluate_exp(x)
    
def evaluate_exp(x):
    import csdl_alpha as csdl
    return csdl.power(np.e, x)
    # return np.e ** x

def exp(x:VariableLike) -> Variable:
    '''
    Elementwise exponential of the input tensor or scalar.

    Parameters
    ----------
    x : VariableLike
        Input tensor to take the exponential of.

    Returns
    -------
    Variable
        Elementwise exponential of the input tensor.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.exp(x)
    >>> y.value
    array([ 2.71828183,  7.3890561 , 20.08553692])
    '''
    op = Exp(validate_and_variablize(x))
    return op.finalize_and_return_outputs()

class TestExp(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        x_val = np.arange(6).reshape(2,3)
        x = csdl.Variable(name = 'x', value = x_val)
        
        compare_values = []
        # exponential of a tensor variable
        s1 = csdl.exp(x)
        t1 = np.exp(x_val)
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # exponential of a scalar constant
        s2 = csdl.exp(3.0)
        t2 = np.array([np.exp(3.0)])
        compare_values += [csdl_tests.TestingPair(s2, t2, tag = 's2')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)


    def test_example(self,):
        self.docstest(exp)

if __name__ == '__main__':
    test = TestExp()
    test.test_functionality()
    test.test_example()