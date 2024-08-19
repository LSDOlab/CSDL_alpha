from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
from csdl_alpha.src.operations.set_get.loop_slice import VarSlice
from csdl_alpha.src.operations.set_get.slice import Slice

import csdl_alpha.utils.testing_utils as csdl_tests
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

    def compute_jax(self, x, *slice_args):
        if self.slice.var_slice is True:
            from csdl_alpha.backends.jax.utils import fallback_to_inline_jax
            return fallback_to_inline_jax(self, x, *slice_args)[0]
        else:
            return x[self.slice.jnpevaluate(*slice_args)].reshape(self.out_shape)

    def evaluate_vjp(self, cotangents, x, *slice_args_and_outputs):
        import csdl_alpha as csdl
        x_indexed = slice_args_and_outputs[-1]
        slice_args = slice_args_and_outputs[:-1]
        if cotangents.check(x):
            # Old
            # new_var = csdl.Variable(value = np.zeros(x.shape))
            # x_cot =  new_var.set(self.slice, cotangents[x_indexed])
            # cotangents.accumulate(x, x_cot)

            # New:
            if cotangents[x] is None:
                new_var = csdl.Variable(value = np.zeros(x.shape))
                x_cot =  new_var.set(self.slice, cotangents[x_indexed])
                cotangents.accumulate(x, x_cot)
            else:
                new_var = cotangents[x]
                x_cot =  new_var.set(self.slice, new_var[self.slice] + cotangents[x_indexed])
                cotangents.accumulate(x, x_cot, replace = True)

def get_index(x:Variable, slices: Slice, shape = None):
    """
    doc strings
    """
    x = validate_and_variablize(x, raise_on_sparse=False)
    

    if isinstance(slices, VarSlice):
        # make sure shape is provided and is valid
        # if shape is None:
        #     raise TypeError("Shape must be provided when indexing with a CSDL variable")
        # else:
        #     from csdl_alpha.utils.error_utils.error_utils import check_if_valid_shape
        #     check_if_valid_shape(shape)

        shape = np.zeros(x.shape, dtype = bool)[slices.evaluate_zeros()].shape
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

        x4 = x.get(csdl.slice[[ind_var2, ind_var, ind_var+3]])
        compare_values += [csdl_tests.TestingPair(x4, x_val[[2,1,4]])]
        x4 = x[[ind_var2, ind_var, ind_var+3]]
        compare_values += [csdl_tests.TestingPair(x4, x_val[[2,1,4]])]

        x5 = x.get(csdl.slice[:, [ind_var, ind_var2, ind_var+3]])
        compare_values += [csdl_tests.TestingPair(x5, x_val[:,[1,2,4]])]
        x5 = x[:, [ind_var, ind_var2, ind_var+3]]
        compare_values += [csdl_tests.TestingPair(x5, x_val[:,[1,2,4]])]

        x6 = x.get(csdl.slice[0:2, [ind_var2, 1, 3], 3])
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2,[2,1,3], 3])]
        x6 = x[0:2, [ind_var2, 1, 3], 3]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2,[2,1,3], 3])]

        x7 = y.get(csdl.slice[0:2, [1, 1, 1],[1, 2, 3], 3])
        compare_values += [csdl_tests.TestingPair(x7, y_val[0:2, [1, 1, 1],[1, 2, 3], 3])]
        x7 = y[0:2, [1, 1, 1],[1, 2, 3], 3]
        compare_values += [csdl_tests.TestingPair(x7, y_val[0:2, [1, 1, 1],[1, 2, 3], 3])]

        x8 = y.get(csdl.slice[0:2, [1, 1, 1],[1, 2, 3], 0:2])
        compare_values += [csdl_tests.TestingPair(x8, y_val[0:2, [1, 1, 1],[1, 2, 3], 0:2])]
        x8 = y[0:2, [1, 1, 1],[1, 2, 3], 0:2]
        compare_values += [csdl_tests.TestingPair(x8, y_val[0:2, [1, 1, 1],[1, 2, 3], 0:2])]

        # slicing with CSDL variables
        int_1 = csdl.Variable(value = 2.0)
        int_2 = int_1+3
        x9 = y.get(csdl.slice[int_1:int_2, [1, 1, 1],[1, 2, 3], 0:2])
        compare_values += [csdl_tests.TestingPair(x9, y_val[2:5, [1, 1, 1],[1, 2, 3], 0:2])]
        x9 = y[int_1:int_2, [1, 1, 1],[1, 2, 3], 0:2]
        compare_values += [csdl_tests.TestingPair(x9, y_val[2:5, [1, 1, 1],[1, 2, 3], 0:2])]

        x10 = y.get(csdl.slice[int_1:int_2, [1, 1, 1],[1, 2, 3], int_2:int_2+2])
        compare_values += [csdl_tests.TestingPair(x10, y_val[2:5, [1, 1, 1],[1, 2, 3], 5:7])]
        x10 = y[int_1:int_2, [1, 1, 1],[1, 2, 3], int_2:int_2+2]
        compare_values += [csdl_tests.TestingPair(x10, y_val[2:5, [1, 1, 1],[1, 2, 3], 5:7])]

        x11 = y.get(csdl.slice[int_1:int_2, [int_1, 1, 1],[int_2, int_2, int_1], int_2:int_2+2])
        compare_values += [csdl_tests.TestingPair(x11, y_val[2:5, [2, 1, 1],[5, 5, 2], 5:7])]
        x11 = y[int_1:int_2, [int_1, 1, 1],[int_2, int_2, int_1], int_2:int_2+2]
        compare_values += [csdl_tests.TestingPair(x11, y_val[2:5, [2, 1, 1],[5, 5, 2], 5:7])]

        x12 = y.get(csdl.slice[0:1, [int_1, 1, 1],[int_2, int_2, int_1], int_2:int_2+2])
        compare_values += [csdl_tests.TestingPair(x12, y_val[0:1, [2, 1, 1],[5, 5, 2], 5:7])]
        x12 = y[0:1, [int_1, 1, 1],[int_2, int_2, int_1], int_2:int_2+2]
        compare_values += [csdl_tests.TestingPair(x12, y_val[0:1, [2, 1, 1],[5, 5, 2], 5:7])]

        with pytest.raises(IndexError):
            x_error = y[0:2, [1, 1, 1],[1, 2], 0:2]
        with pytest.raises(IndexError):
            x_error = y[0:2, [1, 1],[1, 2], 0:2, [1,1,2]]
        with pytest.raises(IndexError):
            x_error = y[100, [1, 1],[1, 2], 0:2]
        with pytest.raises(ValueError):
            x_error = y[0:2, [1, 1], 3]
        with pytest.raises(ValueError):
            x_error = y[0:2, [1, 1, 1], [0, 2, 0], 3]
        with pytest.raises(ValueError):
            x_error = y[0:2, [int_2, int_2, int_2], [0, 2, 0], 3]
        with pytest.raises(TypeError):
            x_error = y[[1, 1],ind_var:2]
        with pytest.raises(TypeError):
            x_error = y[0:2:ind_var]
        with pytest.raises(ValueError):
            x_error = y[int_1:csdl.sin(int_1)]
        with pytest.raises(ValueError):
            x_error = y[int_1:int_1*int_1]
        with pytest.raises(ValueError):
            x_error = y[int_1:int_1]
        with pytest.raises(ValueError):
            x_error = y[int_1:int_1/2]
        with pytest.raises(TypeError):
            x_error = y[int_1:int_1+0.5]
        self.run_tests(compare_values = compare_values, turn_off_recorder=False)

        compare_values = []
        ind_var.value = ind_var.value + 1
        ind_var2.value = ind_var2.value + 1
        int_1.value = int_1.value - 1
        current_graph = csdl.get_current_recorder().active_graph
        current_graph.execute_inline()

        compare_values += [csdl_tests.TestingPair(x1, np.array([x_val[2,0,3]]))]
        compare_values += [csdl_tests.TestingPair(x2, x_val[2,0])]
        compare_values += [csdl_tests.TestingPair(x3, x_val[[2,0,3], [3,0,2]])]
        compare_values += [csdl_tests.TestingPair(x4, x_val[[3,2,5]])]
        compare_values += [csdl_tests.TestingPair(x5, x_val[:,[2,3,5]])]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2,[3,1,3], 3])]
        compare_values += [csdl_tests.TestingPair(x7, y_val[0:2, [1, 1, 1],[1, 2, 3], 3])]
        compare_values += [csdl_tests.TestingPair(x8, y_val[0:2, [1, 1, 1],[1, 2, 3], 0:2])]
        compare_values += [csdl_tests.TestingPair(x9, y_val[1:4, [1, 1, 1],[1, 2, 3], 0:2])]
        compare_values += [csdl_tests.TestingPair(x10, y_val[1:4, [1, 1, 1],[1, 2, 3], 4:6])]
        compare_values += [csdl_tests.TestingPair(x11, y_val[1:4, [1, 1, 1],[4, 4, 1], 4:6])]
        compare_values += [csdl_tests.TestingPair(x12, y_val[0:1, [1, 1, 1],[4, 4, 1], 4:6])]

        self.run_tests(compare_values = compare_values)

    def test_deriv(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        shape_1 = (10,9,8)
        x_val = np.arange(np.prod(shape_1)).reshape(shape_1)
        # shape_2 = (10,9,8,7,6)
        # y_val = np.arange(np.prod(shape_2)).reshape(shape_2)
        x = csdl.Variable(name = 'x', value = x_val)
        # y = csdl.Variable(name = 'y', value = y_val)

        # ind_var = csdl.Variable(name = 'ind_var', value = np.array([1]))
        # ind_var2 = csdl.Variable(name = 'ind_var2', value = np.array([2]))

        compare_values = []
        # x6 = x[0:2, [2, 1, 1], 3]
        # compare_values += [csdl_tests.TestingPair(x6, x_val[0:2,[2,1,1], 3])]

        x6 = x[0,0,0]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0,0,0].flatten(), tag = 'x6')]

        x6 = x[[0,1],[0,1],[1,2]]
        compare_values += [csdl_tests.TestingPair(x6, x_val[[0,1],[0,1],[1,2]].flatten())]

        x6 = x[0:2]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2])]

        x6 = x[0:2, 1]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2, 1])]

        x6 = x[0:2, [1, 2]]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2, [1,2]])]

        x6 = x[0:2, [2, 1], 3]
        compare_values += [csdl_tests.TestingPair(x6, x_val[0:2, [2,1], 3])]

        x7 = x[0:2, [2, 1], 3] + x[0:2, [2, 1], 4] + x[1:3, [2, 1], 3]
        compare_values += [csdl_tests.TestingPair(x7, x_val[0:2, [2,1], 3]+x_val[0:2, [2,1], 4]+x_val[1:3, [2, 1], 3])]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)


if __name__ == '__main__':
    test = TestGet()
    test.overwrite_backend = 'jax'
    # test.overwrite_backend = 'inline'
    # test.test_functionality()
    test.test_deriv()