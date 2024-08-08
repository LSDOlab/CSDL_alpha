from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.typing import VariableLike
import pytest

@set_properties(linear=True)
class CopyVar(ElementwiseOperation):

    def __init__(self,x):
        super().__init__(x)
        self.name = 'copy'

    def compute_inline(self, x):
        return x.copy()
    
    def compute_jax(self, x):
        return x+0.0

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y])

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
    x = validate_and_variablize(x, raise_on_sparse=False)
    return CopyVar(x).finalize_and_return_outputs()

@set_properties(linear=True, elementwise=True)
class CopyVarTo(Operation):

    def __init__(self,x,y):
        super().__init__(x)
        self.name = 'copyto'
        self.set_outputs([y])

    def compute_inline(self, x):
        return x.copy()
    
    def compute_jax(self, x):
        return x+0.0
    
    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y])

def copyto(x:Variable, y:Variable)->Variable:
    """connect existing variauble x to y.
    Y MUST NOT BE COMPUTED FROM SOMEWHERE ELSE

    Parameters
    ----------
    x : Variable
    y : Variable
    """

    # Check if y is computed from somewhere else
    import csdl_alpha as csdl
    recorder = csdl.get_current_recorder()
    current_graph = recorder.active_graph
    if not y in current_graph.node_table:
        raise ValueError(f'y ({y.info()}) must be a variable in the current graph')
    if current_graph.in_degree(y) > 0:
        raise ValueError(f'y ({y.info()}) must not be computed from an operation already. ({current_graph.in_degree(y)} predecessors)')
    if y.shape != x.shape:
        raise ValueError(f'x and y must have the same shape. {x.shape} != {y.shape}')

    return CopyVarTo(x, y).finalize_and_return_outputs()


class TestCopy(csdl_tests.CSDLTest):
    
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
        self.run_tests(compare_values = compare_values, verify_derivatives=True)


class TestCopyVarTo(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x_val = np.arange(10).reshape((2,5))

        x = csdl.Variable(name = 'x', value = x_val)

        y = csdl.Variable(name = 'y', value = x_val*0.0)
        z = y+x

        copyto(x, y)

        current_graph = csdl.get_current_recorder().active_graph
        current_graph.execute_inline()

        compare_values = []
        
        # Variables:
        compare_values += [csdl_tests.TestingPair(z, x_val*2.0)]
        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_error(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x_val = np.arange(10).reshape((2,5))

        x = csdl.Variable(name = 'x', value = x_val)
        x_0 = csdl.Variable(name = 'x', value = x_val[0,0])

        y = csdl.Variable(name = 'y', value = x_val*0.0)
        z = y+x

        with pytest.raises(ValueError):
            copyto(x, z)

        with pytest.raises(ValueError):
            copyto(x, x_0)

        new_recorder = csdl.Recorder(inline = True)
        new_recorder.start()
        new_x = csdl.Variable(value = np.zeros((2,5)))
        new_recorder.stop()

        with pytest.raises(ValueError):
            copyto(x, new_x)


if __name__ == '__main__':
    t = TestCopyVarTo()
    t.test_functionality()
    t.test_error()