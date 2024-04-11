from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests

@set_properties(linear=True)
class Add(ElementwiseOperation):
    '''
    Elementwise addition of two tensors of the same shape.
    '''

    def __init__(self,x:Variable,y:Variable):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x + y

    def evaluate_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jacobian2(self, derivs_out):
        for key in derivs_out:
            #key = (output1,x)
            #key = (output1,y)

            derivs_out[key] = csdl.Constant(derivs_out[key].size, val = 1)

        # return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)
    
    def evaluate_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x, y, vx, vy):
        # do we to flatten the inputs vx and vy (since they are already vectors)?
        return add(vx.flatten(), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return vout.flatten(), vout.flatten()

# TODO: Do we need a broadcast add? There's a lot of code duplication b/w both classes
class BroadcastAdd(Operation):
    '''
    Addition after the first input is broadcasted to the shape of the second input.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'broadcast_add'
        out_shapes = (y.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return x + y

    def evaluate_jacobian(self, x, y):
        # first jac is dense with 1s, second jac is identity
        return csdl.Constant(y.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x, y, vx, vy):
        return add(vx.flatten()*csdl.Constant(y.size, val = 1), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return csdl.sum(vout), vout.flatten()

class SparseAdd(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sparse_add'

    def compute_inline(self, x, y):
        pass
    
class SparseBroadcastAdd(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sparse_broadcast_add'

    def compute_inline(self, x, y):
        pass

def add(x:Variable,y:Variable)->Variable:
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
    >>> csdl.add(x, y).value
    array([5., 7., 9.])
    >>> (x + y).value # equivalent to the above
    array([5., 7., 9.])
    >>> (x + 2.0).value # broadcasting is also supported
    array([3., 4., 5.])
    """
    x = variablize(x)
    y = variablize(y)

    if x.shape == y.shape:
        op = Add(x,y)
    elif x.size == 1:
        op = BroadcastAdd(x.flatten(),y)
    elif y.size == 1:
        op = BroadcastAdd(y.flatten(),x)
    else:
        raise ValueError('Shapes not compatible for add operation.')
    return op.finalize_and_return_outputs()


class TestAdd(csdl_tests.CSDLTest):
    
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
        s1 = csdl.add(x,y)
        t1 = np.array([x_val + y_val])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # add scalar constants
        s2 = csdl.add(3.0, 2.0)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # add scalar constant and scalar variable
        s3 = csdl.add(3.0, y)
        compare_values += [csdl_tests.TestingPair(s3, t1, tag = 's3')]

        # add tensor constants
        s4 = csdl.add(3.0*np.ones((3,2)), 2.0*np.ones((3,2)))
        t2 = 5.0 * np.ones((3,2))
        compare_values += [csdl_tests.TestingPair(s4, t2, tag = 's4')]

        # add scalar constant and tensor constant
        s5 = csdl.add(3.0, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s5, t2, tag = 's5')]

        # add scalar variable and tensor constant
        s6 = csdl.add(x, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s6, t2, tag = 's6')]

        z_val = 2.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # add scalar variable and tensor variable
        s7 = csdl.add(x, z)
        compare_values += [csdl_tests.TestingPair(s7, t2, tag = 's7')]

        # add scalar constant and tensor variable
        s8 = csdl.add(3.0, z)
        compare_values += [csdl_tests.TestingPair(s8, t2, tag = 's8')]

        # add tensor variables
        s9 = csdl.add(x, z)
        compare_values += [csdl_tests.TestingPair(s9, t2, tag = 's9')]

        self.run_tests(compare_values = compare_values,)

    def test_example(self,):
        self.prep()

        # docs:entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        # add two scalar constants
        s0 = csdl.add(3,2)
        print(s0.value)

        x = csdl.Variable(name = 'x', value = np.ones((3,2))*3.0)
        y = csdl.Variable(name = 'y', value = 2.0)
        z = csdl.Variable(name = 'z', value = np.ones((3,2))*2.0)
        
        # add a tensor variable and a scalar variable
        s1 = csdl.add(x,y)
        print(s1.value)

        # add 2 tensor variables
        s2 = csdl.add(x,z)
        print(s2.value)
        
        # add a tensor variable and a scalar constant
        s3 = csdl.add(3,z)
        print(s3.value)
        # docs:exit

        compare_values = []
        t0 = np.array([5.0])
        t  = np.ones((3,2)) * t0

        compare_values += [csdl_tests.TestingPair(s0, t0)]
        compare_values += [csdl_tests.TestingPair(s1, t)]
        compare_values += [csdl_tests.TestingPair(s2, t)]
        compare_values += [csdl_tests.TestingPair(s3, t)]

        self.run_tests(compare_values = compare_values,)

    def test_docstring(self):
        self.docstest(add)

if __name__ == '__main__':
    test = TestAdd()
    test.test_functionality()
    test.test_example()