from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
import numpy as np
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.src.graph.variable import Variable

class Power(ElementwiseOperation):
    '''
    Elementwise power of a tensor.
    Power of the first input to the second input. 
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'power'

    def compute_inline(self, x, y):
        return x ** y
    
    def compute_jax(self, x, y):
        import jax.numpy as jnp
        return (x ** y)
    
    def evaluate_vjp(self, cotangents, x, y, z):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[z]*y*x**(y-1))
        if cotangents.check(y):
            import csdl_alpha as csdl
            cotangents.accumulate(y, cotangents[z]*z*csdl.log(x))

class LeftBroadcastPower(Operation):
    '''
    First input is broadcasted to the shape of the second input.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'left_broadcast_power'
        out_shapes = (y.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return x ** y
    
    def compute_jax(self, x, y):
        import jax.numpy as jnp
        return (x ** y)
    
    def evaluate_vjp(self, cotangents, x, y, z):
        if cotangents.check(x):
            import csdl_alpha as csdl
            cotangents.accumulate(x, csdl.sum(cotangents[z]*y*x**(y-1)))
        if cotangents.check(y):
            import csdl_alpha as csdl
            cotangents.accumulate(y, cotangents[z]*z*csdl.log(x))

class RightBroadcastPower(Operation):
    '''
    Second input is broadcasted to the shape of the first input.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'right_broadcast_power'
        out_shapes = (x.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return x ** y

    def compute_jax(self, x, y):
        import jax.numpy as jnp
        return (x ** y)

    def evaluate_vjp(self, cotangents, x, y, z):
        if cotangents.check(x):
            import csdl_alpha as csdl
            cotangents.accumulate(x, cotangents[z]*y*x**(y-1))
        if cotangents.check(y):
            import csdl_alpha as csdl
            cotangents.accumulate(y, csdl.sum(cotangents[z]*z*csdl.log(x)))

def power(x:VariableLike, y:VariableLike) -> Variable:
    '''
    Computes the power of the first input with exponent as the second input.
    If one of the inputs is a scalar, it is broadcasted to the shape of the other input.

    Parameters
    ----------
    x : Variable, np.ndarray, float or int
        Input tensor whose power needs to be computed.
    y : Variable, np.ndarray, float or int
        Power to which the first input tensor needs to be raised.

    Returns
    -------
    Variable
        Power of the first input with exponent as the second input.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y1 = csdl.power(x, 2)
    >>> y1.value
    array([1., 4., 9.])
    >>> y2 = x ** 2
    >>> y2.value
    array([1., 4., 9.])

    Power raised to a tensor variable exponent

    >>> z = csdl.Variable(value = 3.0 * np.ones((3,)))
    >>> y2 = x ** z
    >>> y2.value
    array([ 1.,  8., 27.])
    '''
    x = validate_and_variablize(x)
    y = validate_and_variablize(y)

    if x.shape == y.shape:
        op = Power(x, y)
    elif x.shape == (1,):
        op = LeftBroadcastPower(x, y)
    elif y.shape == (1,):
        op = RightBroadcastPower(x, y)
    else:
        raise ValueError('Shapes not compatible for the power operation.')
        
    return op.finalize_and_return_outputs()

class TestPower(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(6).reshape(2,3)+1.1
        y_val = 2.0
        z_val = 2.0*np.ones((2,3))
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        
        compare_values = []

        # power of a tensor variable to a tensor variable
        y_tensor = csdl.Variable(value = x_val+1.0)
        s1 = csdl.power(x, y_tensor)
        compare_values += [csdl_tests.TestingPair(s1, x_val**(x_val+1.0), tag = 's0')]

        # power of a tensor variable to a tensor variable
        y_tensor = csdl.Variable(value = -x_val)
        s1 = csdl.power(x, y_tensor)
        compare_values += [csdl_tests.TestingPair(s1, x_val**(-x_val), tag = 's0')]

        # If x is negative, things get strange, y must be integers
        y_tensor = csdl.Variable(name = 'int_tensor', value = np.arange(6).reshape(2,3)+3.0)
        s1 = csdl.power(-x, y_tensor)
        compare_values += [csdl_tests.TestingPair(s1, (-x_val)**(np.arange(6).reshape(2,3)+3.0), tag = 's0')]

        # power of a scalar variable to a tensor variable
        x_scalar = csdl.Variable(value = 3.0)
        y_tensor = csdl.Variable(value = x_val+1.0)
        s1 = csdl.power(x_scalar, y_tensor)
        compare_values += [csdl_tests.TestingPair(s1, 3.0**(x_val+1.0), tag = 's0')]

        # power of a scalar variable to a tensor variable
        y_tensor = csdl.Variable(value = -x_val)
        s1 = csdl.power(x_scalar, y_tensor)
        compare_values += [csdl_tests.TestingPair(s1, 3.0**(-x_val), tag = 's0')]

        # If x is negative, things get strange, y must be integers
        y_tensor = csdl.Variable(name = 'int_tensor', value = np.arange(6).reshape(2,3)+3.0)
        s1 = csdl.power(-x_scalar, y_tensor)
        compare_values += [csdl_tests.TestingPair(s1, (-3.0)**(np.arange(6).reshape(2,3)+3.0), tag = 's0')]


        # power of a tensor variable to a scalar variable
        s1 = csdl.power(x, y)
        compare_values += [csdl_tests.TestingPair(s1, x_val**(y_val), tag = 's0')]

        # power of a tensor variable to a tensor variable
        s1 = csdl.power(x, -y)
        compare_values += [csdl_tests.TestingPair(s1, x_val**(-y_val), tag = 's0')]

        # If x is negative, things get strange, y must be integers
        s1 = csdl.power(-x, y)
        compare_values += [csdl_tests.TestingPair(s1, (-x_val)**(y_val), tag = 's0')]


        # power of a tensor variable to a scalar variable
        s1 = csdl.power(x, y)
        t1 = x_val ** y_val
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # power of a tensor variable to a tensor constant
        s2 = csdl.power(x, z_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # power of a scalar constant to a tensor variable
        s3 = csdl.power(3.0, x)
        t3 = 3.0 ** x_val
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_example(self,):
        self.docstest(power)

if __name__ == '__main__':
    test = TestPower()
    test.test_functionality()
    # test.test_example()