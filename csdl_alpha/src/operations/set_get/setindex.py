from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable

from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.operations.set_get.slice import Slice
from csdl_alpha.src.operations.set_get.loop_slice import VarSlice
import pytest
from csdl_alpha.utils.typing import VariableLike

@set_properties(linear=True,)
class SetVarIndex(Operation):
    '''
    Elementwise setting of a slice s of a tensor x with another tensor y.
    '''

    def __init__(
            self,
            x:Variable,
            y:Variable,
            slice:VarSlice):
        '''
        Slice can be a tuple of slices or a single slice or list of index sets.
        '''
        super().__init__(x, y, *slice.vars)
        self.name = 'set_index'
        out_shapes = (x.shape,) 
        self.set_dense_outputs(out_shapes)
        self.slice = slice

    def compute_inline(self, x, y, *slice_args):
        out = x.copy()
        out[self.slice.evaluate(*slice_args)] = y
        return out

class BroadcastSetIndex(SetVarIndex):
    '''
    Setting all the elements of a slice s of a tensor x with a scalar y.
    '''

    def __init__(self, x, y, slice):
        super().__init__(x, y, slice)
        self.name = 'broadcast_set'

# class SparseSetIndex(ComposedOperation):

#     def __init__(self,x,y):
#         super().__init__(x,y)
#         self.name = 'sparse_set'

#     def compute_inline(self, x, y):
#         pass
    
# class SparseBroadcastSetIndex(ComposedOperation):

#     def __init__(self,x,y):
#         super().__init__(x,y)
#         self.name = 'sparse_broadcast_set'

#     def compute_inline(self, x, y):
#         pass

def set_index(x:Variable, s:Slice, y:VariableLike) -> Variable:
    x = validate_and_variablize(x)
    y = validate_and_variablize(y)

    if y.size != 1:
        import numpy as np
        # TODO: index out of bounds error from csdl instead of numpy
        slice_shape = np.zeros(x.shape)[s.evaluate_zeros()].shape

        # from csdl_alpha.utils.slice import get_slice_shape
        # slice_shape_ = get_slice_shape(s, x.shape)
        # print(slice_shape_, slice_shape)

        if slice_shape != y.shape:
            raise ValueError(f'Slice shape does not match value shape. {slice_shape} != {y.shape}')
        op = SetVarIndex(x, y, s)
    else:
        # TODO: use y.flatten() later once flatten() is implemented
        # op = BroadcastSet(x, y.flatten(), s)
        op = BroadcastSetIndex(x, y, s)
    
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
        ind_0 = csdl.Variable(name = 'ind0', value = 0)
        ind_1 = csdl.Variable(name = 'ind1', value = 1)

        compare_values = []
        # set a scalar slice with a scalar variable
        x1 = x.set(slice[0:1], y)
        x2 = x.set((0,), y)
        x2_v = x.set((ind_0,), y)
        t1 = np.array([2.])
        compare_values += [csdl_tests.TestingPair(x1, t1)]
        compare_values += [csdl_tests.TestingPair(x2, t1)]
        compare_values += [csdl_tests.TestingPair(x2_v, t1)]

        # set a scalar slice with a scalar constant
        x3 = x.set(slice[0:1], 2.0)
        x3_v = x.set([ind_0,], 2.0)
        compare_values += [csdl_tests.TestingPair(x3, t1)]
        compare_values += [csdl_tests.TestingPair(x3_v, t1)]

        z_val = 3.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # set a tensor slice with a tensor constant
        z1 = z.set(slice[0:-1:1], 2.0*np.ones((2,2)))
        z1_v = z.set([ind_0, ind_0+1], 2.0*np.ones((2,2)))
        t2 = np.array([[2.,2.],[2.,2.],[3.,3.]])
        compare_values += [csdl_tests.TestingPair(z1, t2)]
        compare_values += [csdl_tests.TestingPair(z1_v, t2)]

        # set a tensor slice with a scalar constant
        z2 = z.set(slice[0:-1:1], 2.0)
        z2_v = z.set([ind_0, ind_0+1], 2.0)
        compare_values += [csdl_tests.TestingPair(z2, t2)]
        compare_values += [csdl_tests.TestingPair(z2_v, t2)]

        # set a tensor slice with a scalar variable
        z3 = z.set(slice[0:-1:1], y)
        z3_v = z.set([ind_0, ind_0+1], y)
        compare_values += [csdl_tests.TestingPair(z3, t2)]
        compare_values += [csdl_tests.TestingPair(z3_v, t2)]

        t_val = 2.0*np.ones((2,2))
        t = csdl.Variable(name = 't', value = t_val)
        # set a tensor slice with a tensor variable
        z4 = z.set(slice[0:-1:1], t)
        z4_v = z.set([ind_0, ind_0+1], t_val)
        z4_var = z.set(slice[ind_0:ind_0+2:1], t_val)
        compare_values += [csdl_tests.TestingPair(z4, t2)]
        compare_values += [csdl_tests.TestingPair(z4_v, t2)]
        compare_values += [csdl_tests.TestingPair(z4_var, t2)]

        t = csdl.Variable(name = 't', value = 2.0*np.ones((2,1)))
        # set a tensor slice with a tensor variable
        z5 = z.set((slice[0:-1, 1:2]), t)
        z5_var = z.set((slice[0:-1, ind_1:ind_1+1]), t)
        z5_v = z.set(slice[0:-1, [ind_1,ind_1]], t_val) #strange...
        t3 = np.array([[3.,2.],[3.,2.],[3.,3.]])
        compare_values += [csdl_tests.TestingPair(z5, t3)]
        compare_values += [csdl_tests.TestingPair(z5_v, t3)]
        compare_values += [csdl_tests.TestingPair(z5_var, t3)]

        t = csdl.Variable(name = 't', value = 2.0*np.ones((2,)))
        # set a tensor slice at specific indices with a tensor variable
        z6 = z.set(([0,1], [1,1]), t)
        z6_v = z.set(([ind_0,1], [ind_1,ind_1]), t)
        compare_values += [csdl_tests.TestingPair(z6, t3)]
        compare_values += [csdl_tests.TestingPair(z6_v, t3)]

        # set a tensor slice at specific indices with a scalar variable
        z7 = z.set(([0,1], [1,1]), y)
        z7_v = z.set(([ind_0,1], [ind_1,ind_1]), y)
        compare_values += [csdl_tests.TestingPair(z7, t3)]
        compare_values += [csdl_tests.TestingPair(z7_v, t3)]

        # set a tensor slice at specific indices with a scalar constant
        z8 = z.set(([0,1], [1,1]), 2.0)
        z8_v = z.set(([ind_0,1], [ind_1,ind_1]), 2.0)
        compare_values += [csdl_tests.TestingPair(z8, t3)]
        compare_values += [csdl_tests.TestingPair(z8_v, t3)]

        # fixed/var and var step errors
        with pytest.raises(TypeError):
            z.set(slice[0:-1:ind_1], t)
        with pytest.raises(TypeError):
            z.set(slice[0:ind_1:1], 2.0)
        with pytest.raises(TypeError):
            z.set(slice[ind_0:-1:1], t)

        self.run_tests(compare_values = compare_values,turn_off_recorder=False)

        # change indices values to make sure they are updated.
        compare_values = []
        ind_1.value = ind_1.value - 1
        current_graph = csdl.get_current_recorder().active_graph
        current_graph.execute_inline()
        t3 = np.array([[2.,3.],[2.,3.],[3.,3.]])
        compare_values += [csdl_tests.TestingPair(z5_v, t3)]
        compare_values += [csdl_tests.TestingPair(z6_v, t3)]
        compare_values += [csdl_tests.TestingPair(z7_v, t3)]
        compare_values += [csdl_tests.TestingPair(z8_v, t3)]
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