from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable 
from csdl_alpha.utils.inputs import variablize

import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha.utils.error_utils as error_utils
import numpy as np

@set_properties(linear=True, diagonal_jacobian = True)
class Reshape(Operation):
    '''
    Elementwise addition of two tensors of the same shape.
    '''

    def __init__(self,x, shape):
        super().__init__(x)
        self.name = 'reshape'
        self.new_shape = shape
        self.set_dense_outputs((self.new_shape, ))

    def compute_inline(self, x):
        return x.reshape(self.new_shape)

def reshape(x, shape: tuple[int]) -> Variable:
    """
    doc strings
    """
    try:
        error_utils.check_if_valid_shape(shape)
    except Exception as e:
        raise TypeError(f'Error with shape argument in reshape: {e}')
    
    if x.size == np.prod(shape):
        op = Reshape(x, shape = shape)
    else:
        raise ValueError(f'Variable size and new shape do not match: ({x.size} != {np.prod(shape)})')
    return op.finalize_and_return_outputs()

class TestReshape(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        x = csdl.Variable(name = 'x', value = x_val)

        x_val_large = np.ones((10,10,10))
        x_large = csdl.Variable(name = 'x_large', value = x_val_large)

        compare_values = []

        y = csdl.reshape(x, (1,1))
        compare_values += [csdl_tests.TestingPair(y, np.array([[x_val]]))]

        y = csdl.reshape(x_large, (100,10))
        compare_values += [csdl_tests.TestingPair(y,x_val_large.reshape((100,10)))]

        y = csdl.reshape(x_large, (1000,1))
        compare_values += [csdl_tests.TestingPair(y,x_val_large.reshape((1000,1)))]

        y = csdl.reshape(y, y.shape)
        compare_values += [csdl_tests.TestingPair(y,x_val_large.reshape((1000,1)))]

        y = x.reshape((1,1))
        compare_values += [csdl_tests.TestingPair(y, np.array([[x_val]]))]

        y = x_large.reshape((100,10))
        compare_values += [csdl_tests.TestingPair(y,x_val_large.reshape((100,10)))]

        y = x_large.reshape((1000,1))
        compare_values += [csdl_tests.TestingPair(y,x_val_large.reshape((1000,1)))]

        y = y.reshape(y.shape)
        compare_values += [csdl_tests.TestingPair(y,x_val_large.reshape((1000,1)))]

        self.run_tests(compare_values = compare_values,)

    def test_errors(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        import pytest

        x_val_large = np.ones((10,10,10))
        x_large = csdl.Variable(name = 'x_large', value = x_val_large)

        with pytest.raises(TypeError):
            y = csdl.reshape(x_large, 1000)

        with pytest.raises(TypeError):
            y = csdl.reshape(x_large, [1000])

        with pytest.raises(TypeError):
            y = csdl.reshape(x_large, (1000.0,))

        with pytest.raises(TypeError):
            y = csdl.reshape(x_large, (10, 100.0,))

        with pytest.raises(ValueError):
            y = csdl.reshape(x_large, (10, 100, 10))

if __name__ == '__main__':
    test = TestReshape()
    test.test_functionality()
    test.test_errors()