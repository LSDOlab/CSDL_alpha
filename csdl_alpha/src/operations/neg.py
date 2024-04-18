from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import set_properties 
from csdl_alpha.utils.inputs import validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests

@set_properties(linear=True)
class Neg(ElementwiseOperation):

    def __init__(self,x):
        super().__init__(x)
        self.name = 'neg'

    def compute_inline(self, x):
        return -x
    
    def evaluate_jacobian(self, x):
        return csdl.Constant(x.size, val = -1)

    def evaluate_jvp(self, x, vx):
        return -vx

    def evaluate_vjp(self, x, vout):
        return -vout

def negate(x):
    """Compute -1*x of a variable x

    Parameters
    ----------
    x : Variable

    Returns
    -------
    out: Variable

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0, 4.0]))
    >>> (csdl.negate(x)).value
    array([-1., -2., -3., -4.])
    >>> (-x).value # equivalent to the above
    array([-1., -2., -3., -4.])
    """
    x = validate_and_variablize(x, raise_on_sparse = False)
    return Neg(x).finalize_and_return_outputs()

class TestPower(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(6).reshape(2,3)
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.negate(x)

        compare_values = []
        compare_values += [csdl_tests.TestingPair(y, -x_val)]
        self.run_tests(compare_values = compare_values,)

    def test_docstring(self):
        self.docstest(negate)