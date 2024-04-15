from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha as csdl

import numpy as np

class Absolute(ComposedOperation):
    '''
    Elementwise absolute values of the input tensor.
    '''
    def __init__(self, x, rho=20.):
        super().__init__(x)
        self.name  = 'absolute'
        self.rho   = rho

    def evaluate_composed(self, x):
        return evaluate_absolute(x, self.rho)
    
def evaluate_absolute(x, rho):
    out = csdl.maximum(x, -x, rho=rho)
    # TODO: update with better smooth abs func (the one below is not very good) 
    # once csdl.exp is implemented
    # out = 2/rho*csdl.log((1+csdl.exp(rho*x))/2) - x
    return out

def absolute(x, rho=20.):
    """
    Elementwise absolute values of the input tensor.

    Parameters
    ----------
    x : Variable, np.ndarray, float, or int
        Input tensor to take the absolute values of.
    rho : float, default=20.
        Smoothing parameter for the absolute function.

    Returns
    -------
    Variable
        Elementwise absolute of the input tensor.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([-1.0, 2.0, -3.0]))
    >>> y = csdl.absolute(x)
    >>> y.value
    array([1., 2., 3.])
    >>> y = csdl.absolute(0.0)
    >>> y.value
    array([0.03465736])

    Note that the value of `y` is not exactly `0.0` due to the smoothing term.
    The value of y can be made closer to `0.0` by increasing the value of 
    the smoothing parameter `rho`.`
    
    >>> y = csdl.absolute(0.0, rho=200)
    >>> y.value
    array([0.00346574])

    """
    op = Absolute(validate_and_variablize(x), rho=rho)  
    return op.finalize_and_return_outputs()

class TestAbsolute(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(-3,3).reshape(2,3)

        x = csdl.Variable(name = 'x', value = x_val)

        compare_values = []
        # absolute of a tensor variable
        s1 = csdl.absolute(x, rho=200)
        t1 = np.absolute(x_val)
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1', decimal=2)]

        # absolute of a tensor constant
        s2 = csdl.absolute(x_val, rho=200)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2', decimal=2)]

        # absolute of scalar variables
        s3 = csdl.absolute(0.0, rho=200)
        s4 = csdl.absolute(1.0)
        s5 = csdl.absolute(-1.0)
        t3 = np.array([0.0])
        t4 = np.array([1.0])
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3', decimal=2)]
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]
        compare_values += [csdl_tests.TestingPair(s5, t4, tag = 's4')]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.docstest(absolute)

if __name__ == '__main__':
    test = TestAbsolute()
    test.test_functionality()
    test.test_example()