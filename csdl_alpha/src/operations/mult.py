from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.utils.typing import VariableLike

class Mult(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'mult'

    def compute_inline(self, x, y):
        return x*y

    def compute_jax(self, x, y):
        return self.compute_inline(x, y)
    
    def evaluate_vjp(self,cotangents, x, y, z):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[z]*y)
        if cotangents.check(y):
            cotangents.accumulate(y, cotangents[z]*x)

class BroadcastMult(Operation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'broadcast_mult'
        out_shapes = (y.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return x*y

    def compute_jax(self, x, y):
        return self.compute_inline(x, y)
    
    def evaluate_vjp(self, cotangents, x, y, z):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[z].inner(y))
        if cotangents.check(y):
            cotangents.accumulate(y, cotangents[z]*x)

def mult(x,y):
    """Elementwise multiplication of two tensors x and y.

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
    >>> csdl.mult(x, y).value
    array([ 4., 10., 18.])
    >>> (x * y).value # equivalent to the above
    array([ 4., 10., 18.])
    >>> (x * 2.0).value # broadcasting is also supported
    array([2., 4., 6.])
    """
    x = validate_and_variablize(x, raise_on_sparse = False)
    y = validate_and_variablize(y, raise_on_sparse = False)
    if x.shape == y.shape:
        op = Mult(x,y)
    elif x.size == 1:
        op = BroadcastMult(x.flatten(),y)
    elif y.size == 1:
        op = BroadcastMult(y.flatten(),x)
    else:
        raise ValueError('Shapes not compatible for add operation.')
    return op.finalize_and_return_outputs()

class TestMult(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = 2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        compare_values = []
        # add scalar variables
        s1 = csdl.mult(x,y)
        t1 = np.array([x_val*y_val])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]
        s1 = x*y
        compare_values += [csdl_tests.TestingPair(s1, t1)]

        # add scalar constants
        s2 = csdl.mult(3.0, 2.0)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # add scalar constant and scalar variable
        s3 = csdl.mult(3.0, y)
        compare_values += [csdl_tests.TestingPair(s3, t1, tag = 's3')]
        s3 = 3.0*y
        compare_values += [csdl_tests.TestingPair(s3, t1, tag = 's3')]
        s3 = y*3.0
        compare_values += [csdl_tests.TestingPair(s3, t1, tag = 's3')]

        # add tensor constants
        s4 = csdl.mult(3.0*np.ones((3,2)), 2.0*np.ones((3,2)))
        t2 = 6.0 * np.ones((3,2))
        compare_values += [csdl_tests.TestingPair(s4, t2, tag = 's4')]

        # add scalar constant and tensor constant
        s5 = csdl.mult(3.0, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s5, t2, tag = 's5')]

        # add scalar variable and tensor constant
        s6 = csdl.mult(x, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s6, t2, tag = 's6')]
        s6 = x*2.0*np.ones((3,2))
        compare_values += [csdl_tests.TestingPair(s6, t2, tag = 's6')]
        s6 = 2.0*np.ones((3,2))*x
        compare_values += [csdl_tests.TestingPair(s6, t2, tag = 's6')]

        z_val = 2.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # add scalar variable and tensor variable
        s7 = csdl.mult(x, z)
        compare_values += [csdl_tests.TestingPair(s7, t2, tag = 's7')]
        s7 = x*z
        compare_values += [csdl_tests.TestingPair(s7, t2, tag = 's7')]
        s7 = z*x
        compare_values += [csdl_tests.TestingPair(s7, t2, tag = 's7')]

        # add scalar constant and tensor variable
        s8 = csdl.mult(3.0, z)
        compare_values += [csdl_tests.TestingPair(s8, t2, tag = 's8')]
        s8 = z*3.0
        compare_values += [csdl_tests.TestingPair(s8, t2, tag = 's8')]
        s8 = 3.0*z
        compare_values += [csdl_tests.TestingPair(s8, t2, tag = 's8')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_docstring(self):
        self.docstest(mult)
if __name__ == '__main__':
    test = TestMult()
    test.test_functionality()
    test.test_docstring()