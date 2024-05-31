from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.typing import VariableLike

@set_properties()
class Div(ElementwiseOperation):
    '''
    Elementwise division of two tensors of the same shape.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'div'

    def compute_inline(self, x, y):
        return x/y

    def evaluate_vjp(self,cotangents, x, y, z):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[z]/y)
        if cotangents.check(y):
            cotangents.accumulate(y, -cotangents[z]*z/y)
            # cotangents.accumulate(y, -cotangents[z]*x/y**2)

@set_properties()
class BroadcastDiv1(Operation):
    '''
    Broadcasted division of a scalar (x) and a tensor (y).
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'bdiv1'
        self.set_dense_outputs((y.shape,))

    def compute_inline(self, x, y):
        return x/y

    def evaluate_vjp(self,cotangents, x, y, z):
        if cotangents.check(x):
            import csdl_alpha as csdl
            cotangents.accumulate(x, csdl.sum(cotangents[z]/y))
        if cotangents.check(y):
            cotangents.accumulate(y, -cotangents[z]*z/y)

@set_properties()
class BroadcastDiv2(Operation):
    '''
    Broadcasted division of a tensor (x) and a scalar (y).
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'bdiv2'
        self.set_dense_outputs((x.shape,))

    def compute_inline(self, x, y):
        return x/y

    def evaluate_vjp(self,cotangents, x, y, z):
        import csdl_alpha as csdl
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[z]/y)
        if cotangents.check(y):
            cotangents.accumulate(y, -csdl.sum(cotangents[z]*z)/y)

def div(x:VariableLike,y:VariableLike)->Variable:
    """Elementwise addition of two tensors x and y.

    Parameters
    ----------
    x : Variable
    y : Variable

    Returns
    -------
    out: Variable

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.Variable(value = np.array([4.0, 5.0, 6.0]))
    >>> csdl.div(x, y).value
    array([0.25, 0.4 , 0.5 ])
    >>> (x/y).value # equivalent to the above
    array([0.25, 0.4 , 0.5 ])
    >>> (x/2.0).value # broadcasting is also supported
    array([0.5, 1. , 1.5])
    """
    x = validate_and_variablize(x)
    y = validate_and_variablize(y)

    if x.shape == y.shape:
        op = Div(x,y)
    elif x.size == 1:
        op = BroadcastDiv1(x.flatten(),y)
    elif y.size == 1:
        op = BroadcastDiv2(x,y.flatten())
    else:
        raise ValueError('Shapes do not match')
    return op.finalize_and_return_outputs()


class TestDiv(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x_val = np.arange(10).reshape((2,5))
        y_val = np.arange(10).reshape((2,5))*0.5+1.0

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []

        # Variables:
        z = csdl.div(x,y)
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val)]

        z = x/y
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val)]
        
        # Constant scalar:
        z = csdl.div(x, 2.0)
        compare_values += [csdl_tests.TestingPair(z, x_val/2.0)]

        z = x/2.0
        compare_values += [csdl_tests.TestingPair(z, x_val/2.0)]

        z = x/(np.ones((1,1,1))*2.0)
        compare_values += [csdl_tests.TestingPair(z, x_val/2.0)]

        z = csdl.div(2.0, y)
        compare_values += [csdl_tests.TestingPair(z, 2.0/y_val)]

        z = 2.0/y
        compare_values += [csdl_tests.TestingPair(z, 2.0/y_val)]

        z = (np.ones((1,1,1))*2.0)/y
        compare_values += [csdl_tests.TestingPair(z, 2.0/y_val)]

        # Constant np array:
        z = csdl.div(x, y_val)
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val)]

        z = x/y_val
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val)]

        z = csdl.div(x_val, y)
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val)]

        z = x_val/y
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val)]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_errors(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        import pytest

        x_val = np.arange(10).reshape((2,5))
        y_val = np.arange(5).reshape((5,1))*0.5+1.0

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        # wrong shapes
        with pytest.raises(ValueError):
            z = csdl.div(x/y)

        with pytest.raises(ValueError):
            z = csdl.div(y/x)

        with pytest.raises(ValueError):
            z = csdl.div(x,y_val)

        with pytest.raises(ValueError):
            z = csdl.div(x_val,y)

        with pytest.raises(ValueError):
            z = csdl.div(x_val,y_val)
    
    def test_docstring(self):
        self.docstest(div)

if __name__ == '__main__':
    test = TestDiv()
    test.test_functionality()
    test.test_errors()