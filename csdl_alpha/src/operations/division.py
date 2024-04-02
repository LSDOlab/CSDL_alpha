from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests

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

def div(x,y):
    """
    doc strings
    """
    x = variablize(x)
    y = variablize(y)

    if x.shape == y.shape:
        op = Div(x,y)
    elif x.size == 1:
        op = BroadcastDiv1(x.reshape((1,)),y)
    elif y.size == 1:
        op = BroadcastDiv2(x,y.reshape((1,)))
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

        self.run_tests(compare_values = compare_values,)

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

if __name__ == '__main__':
    test = TestDiv()
    test.test_functionality()
    test.test_errors()