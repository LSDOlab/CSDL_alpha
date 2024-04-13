from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.typing import VariableLike

@set_properties(linear=True)
class CopyVar(ElementwiseOperation):

    def __init__(self,x):
        super().__init__(x)
        self.name = 'copy'

    def compute_inline(self, x):
        return x.copy()

def copyvar(x:VariableLike)->Variable:
    """Return a copy of the input variable x.

    Parameters
    ----------
    x : VariableLike

    Returns
    -------
    out: Variable
        A new variable that represents the same value as x
    """
    x = variablize(x)
    return CopyVar(x).finalize_and_return_outputs()

class TestDiv(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x_val = np.arange(10).reshape((2,5))

        x = csdl.Variable(name = 'x', value = x_val)

        compare_values = []
        
        # Variables:
        z = csdl.copyvar(x)
        compare_values += [csdl_tests.TestingPair(z, x_val)]
        self.run_tests(compare_values = compare_values,)