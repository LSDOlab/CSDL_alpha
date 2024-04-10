from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
from csdl_alpha.src.operations.set_get.loop_slice import VarSlice
from csdl_alpha.src.operations.set_get.slice import Slice

import csdl_alpha.utils.test_utils as csdl_tests
import numpy as np
import pytest

@set_properties()
class GetVarIndex(Operation):
    def __init__(
            self,
            x:Variable,
            slice:VarSlice,
            slice_shape:tuple,
        ):
        super().__init__(x, *slice.vars)
        self.name = 'get_index'
        self.out_shape = slice_shape
        out_shapes = (self.out_shape,) 
        self.set_dense_outputs(out_shapes)
        self.slice = slice

    def compute_inline(self, x, *slice_args):
        return x[self.slice.evaluate(*slice_args)].reshape(self.out_shape)

def get_index(x:Variable, slices: Slice, shape = None):
    """
    doc strings
    """
    x = variablize(x)
    

    if isinstance(slices, VarSlice):
        # make sure shape is provided and is valid
        # if shape is None:
        #     raise TypeError("Shape must be provided when indexing with a CSDL variable")
        # else:
        #     from csdl_alpha.utils.error_utils.error_utils import check_if_valid_shape
        #     check_if_valid_shape(shape)

        shape = np.zeros(x.shape)[slices.evaluate_zeros()].shape
        if shape == ():
            shape = (1,)

        # Create operation
        op = GetVarIndex(x, slices, shape)

    return op.finalize_and_return_outputs()

class TestGet(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        shape_1 = (10,9,8)
        x_val = np.arange(np.prod(shape_1)).reshape(shape_1)
        shape_2 = (10,9,8,7,6)
        y_val = np.arange(np.prod(shape_2)).reshape(shape_2)
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        ind_var = csdl.Variable(name = 'ind_var', value = np.array([1]))
        ind_var2 = csdl.Variable(name = 'ind_var', value = np.array([2]))

        compare_values = []
        x1 = x.get(csdl.slice[ind_var,0,ind_var2])
        compare_values += [csdl_tests.TestingPair(x1, np.array([x_val[1,0,2]]))]
        x1 = x[ind_var,0,ind_var2]
        compare_values += [csdl_tests.TestingPair(x1, np.array([x_val[1,0,2]]))]

        x2 = x.get(csdl.slice[ind_var,0])
        compare_values += [csdl_tests.TestingPair(x2, x_val[1,0])]
        x2 = x[ind_var,0]
        compare_values += [csdl_tests.TestingPair(x2, x_val[1,0])]

        x3 = x.get(csdl.slice[[ind_var,0, ind_var2],[ind_var2,0, ind_var]])
        compare_values += [csdl_tests.TestingPair(x3, x_val[[1,0,2], [2,0,1]])]
        x3 = x[[ind_var,0, ind_var2],[ind_var2,0, ind_var]]
        compare_values += [csdl_tests.TestingPair(x3, x_val[[1,0,2], [2,0,1]])]

        x4 = x.get(csdl.slice[[ind_var2, ind_var, ind_var]])
        compare_values += [csdl_tests.TestingPair(x4, x_val[[2,1,1]])]
        x4 = x[[ind_var2, ind_var, ind_var]]
        compare_values += [csdl_tests.TestingPair(x4, x_val[[2,1,1]])]

        x5 = x.get(csdl.slice[:, [ind_var, ind_var2, ind_var]])
        compare_values += [csdl_tests.TestingPair(x5, x_val[:,[1,2,1]])]
        x5 = x[:, [ind_var, ind_var2, ind_var]]
        compare_values += [csdl_tests.TestingPair(x5, x_val[:,[1,2,1]])]

        x6 = x.get(csdl.slice[0:2, [ind_var2, 1, 1], 3])
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2,[2,1,1], 3])]
        x6 = x[0:2, [ind_var2, 1, 1], 3]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2,[2,1,1], 3])]

        x7 = y.get(csdl.slice[0:2, [1, 1, 1],[1, 2, 3], 3])
        compare_values += [csdl_tests.TestingPair(x7, y_val[0:2, [1, 1, 1],[1, 2, 3], 3])]
        x7 = y[0:2, [1, 1, 1],[1, 2, 3], 3]
        compare_values += [csdl_tests.TestingPair(x7, y_val[0:2, [1, 1, 1],[1, 2, 3], 3])]

        x8 = y.get(csdl.slice[0:2, [1, 1, 1],[1, 2, 3], 0:2])
        compare_values += [csdl_tests.TestingPair(x8, y_val[0:2, [1, 1, 1],[1, 2, 3], 0:2])]
        x8 = y[0:2, [1, 1, 1],[1, 2, 3], 0:2]
        compare_values += [csdl_tests.TestingPair(x8, y_val[0:2, [1, 1, 1],[1, 2, 3], 0:2])]

        with pytest.raises(IndexError):
            x_error = y[0:2, [1, 1, 1],[1, 2], 0:2]
        with pytest.raises(IndexError):
            x_error = y[0:2, [1, 1],[1, 2], 0:2, [1,1,2]]
        with pytest.raises(IndexError):
            x_error = y[100, [1, 1],[1, 2], 0:2]
        with pytest.raises(TypeError):
            x_error = y[[1, 1],ind_var:2]
        with pytest.raises(TypeError):
            x_error = y[0:2:ind_var]
        self.run_tests(compare_values = compare_values, turn_off_recorder=False)

        compare_values = []
        ind_var.value = ind_var.value + 1
        ind_var2.value = ind_var2.value + 1
        current_graph = csdl.get_current_recorder().active_graph
        current_graph.execute_inline()

        compare_values += [csdl_tests.TestingPair(x1, np.array([x_val[2,0,3]]))]
        compare_values += [csdl_tests.TestingPair(x2, x_val[2,0])]
        compare_values += [csdl_tests.TestingPair(x3, x_val[[2,0,3], [3,0,2]])]
        compare_values += [csdl_tests.TestingPair(x4, x_val[[3,2,2]])]
        compare_values += [csdl_tests.TestingPair(x5, x_val[:,[2,3,2]])]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2,[3,1,1], 3])]
        compare_values += [csdl_tests.TestingPair(x7, y_val[0:2, [1, 1, 1],[1, 2, 3], 3])]
        compare_values += [csdl_tests.TestingPair(x8, y_val[0:2, [1, 1, 1],[1, 2, 3], 0:2])]
        self.run_tests(compare_values = compare_values)


if __name__ == '__main__':
    test = TestGet()
    test.test_functionality()