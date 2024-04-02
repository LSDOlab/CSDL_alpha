from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests

@set_properties(linear=True,)
class Set(Operation):
    '''
    Elementwise setting of a slice s of a tensor x with another tensor y.
    '''

    def __init__(self, x, y, slice):
        '''
        Slice can be a tuple of slices or a single slice or list of index sets.
        '''
        super().__init__(x, y)
        self.name = 'set'
        out_shapes = (x.shape,) 
        self.set_dense_outputs(out_shapes)
        self.slice = slice

    def compute_inline(self, x, y):
        out = x.copy()
        out[self.slice] = y
        return out

class BroadcastSet(Set):
    '''
    Setting all the elements of a slice s of a tensor x with a scalar y.
    '''

    def __init__(self, x, y, slice):
        super().__init__(x, y, slice)
        self.name = 'broadcast_set'


class SparseSet(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sparse_set'

    def compute_inline(self, x, y):
        pass
    
class SparseBroadcastSet(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sparse_broadcast_set'

    def compute_inline(self, x, y):
        pass

def set(x, s, y):
    """
    doc strings
    """
    x = variablize(x)
    y = variablize(y)

    if y.size != 1:
        import numpy as np
        # TODO: index out of bounds error from csdl instead of numpy
        slice_shape = np.zeros(x.shape)[s].shape

        # from csdl_alpha.utils.slice import get_slice_shape
        # slice_shape_ = get_slice_shape(s, x.shape)
        # print(slice_shape_, slice_shape)

        if slice_shape != y.shape:
            raise ValueError('Shapes of inputs do not match for the set operation.')
        op = Set(x, y, s)
    else:
        # TODO: use y.flatten() later once flatten() is implemented
        # op = BroadcastSet(x, y.flatten(), s)
        op = BroadcastSet(x, y, s)
    
    return op.finalize_and_return_outputs()


class TestSet(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        from csdl_alpha import slice
        x_val = 3.0
        y_val = 2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []
        # set a scalar slice with a scalar variable
        x1 = x.set(slice[0:1], y)
        x2 = x.set((0,), y)
        t1 = np.array([2.])
        compare_values += [csdl_tests.TestingPair(x1, t1)]
        compare_values += [csdl_tests.TestingPair(x2, t1)]

        # set a scalar slice with a scalar constant
        x3 = x.set(slice[0:1], 2.0)
        compare_values += [csdl_tests.TestingPair(x3, t1)]

        z_val = 3.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # set a tensor slice with a tensor constant
        z1 = z.set(slice[0:-1:1], 2.0*np.ones((2,2)))
        t2 = np.array([[2.,2.],[2.,2.],[3.,3.]])
        compare_values += [csdl_tests.TestingPair(z1, t2)]

        # set a tensor slice with a scalar constant
        z2 = z.set(slice[0:-1:1], 2.0)
        compare_values += [csdl_tests.TestingPair(z2, t2)]

        # set a tensor slice with a scalar variable
        z3 = z.set(slice[0:-1:1], y)
        compare_values += [csdl_tests.TestingPair(z3, t2)]

        t_val = 2.0*np.ones((2,2))
        t = csdl.Variable(name = 't', value = t_val)
        # set a tensor slice with a tensor variable
        z4 = z.set(slice[0:-1:1], t)
        compare_values += [csdl_tests.TestingPair(z4, t2)]

        t = csdl.Variable(name = 't', value = 2.0*np.ones((2,1)))
        # set a tensor slice with a tensor variable
        z5 = z.set((slice[0:-1, 1:2]), t)
        t3 = np.array([[3.,2.],[3.,2.],[3.,3.]])
        compare_values += [csdl_tests.TestingPair(z5, t3)]

        t = csdl.Variable(name = 't', value = 2.0*np.ones((2,)))
        # set a tensor slice at specific indices with a tensor variable
        z6 = z.set(((0,1), (1,1)), t)
        compare_values += [csdl_tests.TestingPair(z6, t3)]

        # set a tensor slice at specific indices with a scalar variable
        z7 = z.set(((0,1), (1,1)), y)
        compare_values += [csdl_tests.TestingPair(z7, t3)]

        # set a tensor slice at specific indices with a scalar constant
        z8 = z.set(((0,1), (1,1)), 2.0)
        compare_values += [csdl_tests.TestingPair(z8, t3)]

        self.run_tests(compare_values = compare_values,)

    def test_example(self,):
        self.prep()

        # docs:entry
        import csdl_alpha as csdl
        from csdl_alpha import slice
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        x = csdl.Variable(name = 'x', value = np.ones((3,2))*3.0)
        y = csdl.Variable(name = 'y', value = 2.0)
        z = csdl.Variable(name = 'z', value = np.ones((2,2))*2.0)

        # set a scalar slice with a scalar variable
        x1 = x.set(slice[0:-1], y)
        print(x1.value)

        # set a tensor slice with a scalar constant
        x2 = x.set(slice[0:-1], 2)
        print(x2.value)

        # set a tensor slice with a tensor variable
        x3 = x.set(slice[0:-1], z)
        print(x3.value)
        # docs:exit

        compare_values = []
        t = np.array([[2.,2.],[2.,2.],[3.,3.]])

        compare_values += [csdl_tests.TestingPair(x1, t)]
        compare_values += [csdl_tests.TestingPair(x2, t)]
        compare_values += [csdl_tests.TestingPair(x3, t)]

        self.run_tests(compare_values = compare_values,)

if __name__ == '__main__':
    test = TestSet()
    test.test_functionality()
    test.test_example()