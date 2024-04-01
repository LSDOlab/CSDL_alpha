from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.utils.inputs import variablize
from csdl_alpha.utils.test_utils import CSDLTest

@set_properties(linear=True)
class Add(ElementwiseOperation):
    '''
    Elementwise addition of two tensors of the same shape.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x + y

    def evaluate_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x, y, vx, vy):
        # do we to flatten the inputs vx and vy (since they are already vectors)?
        return add(vx.flatten(), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return vout.flatten(), vout.flatten()

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

def add(x,y):
    """
    doc strings
    """
    x = variablize(x)
    y = variablize(y)

    if x.shape == y.shape:
        op = Add(x,y)
    elif x.shape == (1,):
        op = BroadcastAdd(x,y)
    elif y.shape == (1,):
        op = BroadcastAdd(y,x)
    else:
        raise ValueError('Shapes do not match')
    return op.finalize_and_return_outputs()


class TestAdd(CSDLTest):
    def test_values(self,):

        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.ones((3,2))*3.0
        y_val = np.ones((3,2))*2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        # add scalar variables
        s = csdl.add(x,y)

        real_s = np.ones((3,2))*5.0

        self.run_tests(
            compare_values = {s: real_s},
            compare_derivatives = {(s,x): 2.0, (s,y): 1.0},
        )


    def test_functionality(self,):
        import csdl_alpha as csdl
        import numpy as np
        from numpy.testing import assert_array_equal

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x = csdl.Variable(name = 'x', value = 3.0)
        y = csdl.Variable(name = 'y', value = 2.0)
        
        # add scalar variables
        s = csdl.add(x,y)
        assert s.value == np.array([5.])
        assert s.shape == (1,)

        # add scalar constants
        s = csdl.add(3.0, 2.0)
        assert s.value == np.array([5.])
        assert s.shape == (1,)

        # add tensor constants
        s = csdl.add(3.0*np.ones((3,2)), 2.0*np.ones((3,2)))
        assert_array_equal(s.value, np.ones((3,2))*5.0)
        assert s.shape == (3,2)

        # add scalar constant and tensor constant
        s = csdl.add(3.0, 2.0*np.ones((3,2)))
        assert_array_equal(s.value, np.ones((3,2))*5.0)
        assert s.shape == (3,2)

        # add scalar variable and tensor constant
        s = csdl.add(x, 2.0*np.ones((3,2)))
        assert_array_equal(s.value, np.ones((3,2))*5.0)
        assert s.shape == (3,2)

        # add scalar constant and scalar variable
        s = csdl.add(3.0, y)
        assert s.value == np.array([5.])
        assert s.shape == (1,)

        z = csdl.Variable(name = 'z', value = 2.0*np.ones((3,2)))
        
        # add scalar variable and tensor variable
        s = csdl.add(x, z)
        assert_array_equal(s.value, np.ones((3,2))*5.0)
        assert s.shape == (3,2)

        # add scalar constant and tensor variable
        s = csdl.add(3.0, z)
        assert_array_equal(s.value, np.ones((3,2))*5.0)
        assert s.shape == (3,2)

        # add tensor variables
        s = csdl.add(x, z)
        assert_array_equal(s.value, np.ones((3,2))*5.0)
        assert s.shape == (3,2)

    def test_example(self,):
        from numpy.testing import assert_array_equal

        # docs:entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        s1 = csdl.add(3,2)
        print(s1.value)

        x = csdl.Variable(name = 'x', value = np.ones((3,2))*3.0)
        y = csdl.Variable(name = 'y', value = 2.0)
        z = csdl.Variable(name = 'z', value = np.ones((3,2))*2.0)
        s1 = csdl.add(x,y)
        print(s1.value)

        s2 = csdl.add(x,z)
        print(s2.value)
        
        s3 = csdl.add(3,z)
        print(s3.value)
        # docs:exit

        assert_array_equal(s1.value, np.ones((3,2))*5.0)
        assert_array_equal(s2.value, np.ones((3,2))*5.0)

if __name__ == '__main__':
    test = TestAdd()
    test.test_functionality()
    test.test_example()