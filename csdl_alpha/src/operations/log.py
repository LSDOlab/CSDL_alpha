from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
import numpy as np
from csdl_alpha.src.operations import add
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests

class Log(ElementwiseOperation):
    '''
    Elementwise logarithm of a tensor.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'log'

    def compute_inline(self, x, y):
        return np.log(x) / np.log(y)
    
    def compute_jax(self, x, y):
        import jax.numpy as jnp
        return jnp.log(x) / jnp.log(y)
    
    def evaluate_vjp(self, cotangents, x, y, z):
        import csdl_alpha as csdl
        vout = cotangents[z]
        if cotangents.check(x):
            cotangents.accumulate(x, vout / (x * csdl.log(y)))
        if cotangents.check(y):
            cotangents.accumulate(y, -vout * csdl.log(x) / (y * (csdl.log(y))**2))

# We need a broadcast log even when the methods are exactly the same because Broadcast cannot inherit from ElementwiseOperation
# TODO: Avoid code duplication
class LeftBroadcastLog(Operation):
    '''
    Logarithm after the first input is broadcasted to the shape of the second input.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'left_broadcast_log'
        out_shapes = (y.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return np.log(x) / np.log(y)
    
    def compute_jax(self, x, y):
        import jax.numpy as jnp
        return jnp.log(x) / jnp.log(y)
    
    def evaluate_vjp(self, cotangents, x, y, z):
        import csdl_alpha as csdl
        vout = cotangents[z]
        if cotangents.check(x):
            cotangents.accumulate(x, csdl.sum(vout / (x * csdl.log(y))))
        if cotangents.check(y):
            cotangents.accumulate(y, -vout * csdl.log(x) / (y * (csdl.log(y))**2))

class RightBroadcastLog(Operation):
    '''
    Logarithm after the second input is broadcasted to the shape of the first input.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'right_broadcast_log'
        out_shapes = (x.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return np.log(x) / np.log(y)

    def compute_jax(self, x, y):
        import jax.numpy as jnp
        return jnp.log(x) / jnp.log(y)

    def evaluate_vjp(self, cotangents, x, y, z):
        import csdl_alpha as csdl
        vout = cotangents[z]
        if cotangents.check(x):
            cotangents.accumulate(x, vout / (x * csdl.log(y)))
        if cotangents.check(y):
            cotangents.accumulate(y, - csdl.sum(vout *csdl.log(x) / (y * (csdl.log(y))**2)))

def log(x, base=None):
    '''
    Computes the natural logarithm of all entries in the input tensor 
    if `base` argument is not provided.
    Otherwise, computes the logarithm of all entries in the input tensor
    with respect to the specified base.
    If one of the inputs is a scalar, it is broadcasted to the shape of the other input.

    Parameters
    ----------
    x : Variable, np.ndarray, float or int
        Input tensor whose logarithm needs to be computed.
    base : Variable, np.ndarray, float or int, default=np.e
        Base of the logarithm. If not provided, natural logarithm is computed.

    Returns
    -------
    Variable
        Logarithm of the first input with base as the second input.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y1 = csdl.log(x)
    >>> y1.value
    array([0.        , 0.69314718, 1.09861229])

    Logarithm with a specified base

    >>> y2 = csdl.log(x, 2)
    >>> y2.value
    array([0.       , 1.       , 1.5849625])

    Logarithm with a specified tensor variable base

    >>> b = csdl.Variable(value = 2.0 * np.ones((3,)))
    >>> y3 = csdl.log(x, b)
    >>> y3.value
    array([0.       , 1.       , 1.5849625])
    '''

    x = validate_and_variablize(x)
    if base is None:
        y = validate_and_variablize(np.e)
    else:
        y = validate_and_variablize(base)

    if x.shape == y.shape:
        op = Log(x, y)
    elif x.size == 1:
        op = LeftBroadcastLog(x.flatten(), y)
    elif y.size == 1:
        op = RightBroadcastLog(x, y.flatten())
    else:
        raise ValueError('Shapes not compatible for log operation.')
        
    return op.finalize_and_return_outputs()

class TestLog(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = 2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        compare_values = []
        # log of a scalar variable
        s1 = csdl.log(x)
        t1 = np.array([np.log(x_val)])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # log of a scalar constant
        s2 = csdl.log(3.0)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # log of a scalar variable with scalar variable base
        s3 = csdl.log(x, y)
        t3 = np.array([np.log(x_val) / np.log(y_val)])
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # log of a scalar variable with scalar constant base
        s4 = csdl.log(x, 2.0)
        compare_values += [csdl_tests.TestingPair(s4, t3, tag = 's4')]

        # log of a scalar constant with scalar constant base
        s5 = csdl.log(3.0, 2.0)
        compare_values += [csdl_tests.TestingPair(s5, t3, tag = 's5')]

        z_val = 2.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # log of a tensor variable with tensor constant base
        s6 = csdl.log(z, 3.0*np.ones((3,2)))
        t6 = np.log(z_val) / np.log(3.0)
        compare_values += [csdl_tests.TestingPair(s6, t6, tag = 's6')]
        
        # log of a tensor constant with a tensor variable base
        s7 = csdl.log(3.0*np.ones((3,2)), z)
        t6 = np.log(3.0) / np.log(z_val)
        compare_values += [csdl_tests.TestingPair(s7, t6, tag = 's7')]

        # log of a scalar constant with a tensor variable base
        s8 = csdl.log(3.0, z)
        compare_values += [csdl_tests.TestingPair(s8, t6, tag = 's8')]

        # log of a scalar constant with a tensor variable base
        s8 = csdl.log(0.001)
        t6 = np.log(0.001).flatten()
        compare_values += [csdl_tests.TestingPair(s8, t6, tag = 's8', decimal = 3)]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_example(self,):
        self.docstest(log)

if __name__ == '__main__':
    test = TestLog()
    test.test_functionality()
    test.test_example()