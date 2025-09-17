import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.inputs import get_type_string
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.loops.new_loop.loop_builder import enter_loop

import numbers
from typing import Callable, Tuple, Union
import pytest
import numpy as np

def batch_function(
        func:Callable,
        batch_size:int,
        batch_dims:Tuple[Union[int,None]],
        output_types:Tuple[Union[int,None]] = None)->Callable:

    # ==== Error checks ====:
    # - check if the function is callable
    # - check if batch_size is an integer, make sure integer is divisible by the batch_size
    # - check if the batch_dims is a tuple that matches the input arguments tuple
    # - - additionally, if the batch_dims is a tuple of integers, check if they are within the shape of that variable (and make sure they are variables in the first place)
    # - check if the output_sums is a tuple of integers
    
    # Checks:
    if not callable(func):
        raise TypeError(f'argument func must be a callable. Type {get_type_string(func)} given')
    if not isinstance(batch_dims, (tuple, list, numbers.Integral)):
        raise TypeError(f'argument batch_dims must be a tuple, list or integer. Type {get_type_string(batch_dims)} given')
    if not isinstance(batch_size, numbers.Integral):
        raise TypeError(f'argument batch_size must be an integer. Type {get_type_string(batch_size)} given')

    # parse the batch_dims
    if isinstance(batch_dims, numbers.Integral): batch_dims = (batch_dims,)
    elif isinstance(batch_dims, list): batch_dims = tuple(batch_dims)
    if len(batch_dims) == 0:
        raise ValueError(f"batch_dims must be a tuple of integers. {len(batch_dims)} given.")

    # create batched_func
    def batched_func(*args:tuple[Variable], **kwargs):
        # Check if the batch_dims is a tuple of integers that matches the inputs
        if len(args) != len(batch_dims):
            raise ValueError(f"batch_dims must be a tuple of integers that matches the inputs. {len(args)} arguments given, {len(batch_dims)} batch_dims provided.")
        
        # More checks:
        full_size = None
        for i, (dim, arg) in enumerate(zip(batch_dims, args)):
            # Check if the batch_dims are integers
            if not isinstance(dim, (numbers.Integral, type(None))):
                raise TypeError(f"batch_dims must be a tuple of integers or None. {get_type_string(dim)} given for batch_dim index {i}.")
            # Check if the batch_dims are within the shape of the variable
            elif isinstance(dim, numbers.Integral):
                if not isinstance(arg, Variable):
                    raise TypeError(f"Cannot batch a non-variable. {get_type_string(arg)} given for batch_dim index {i}.")
                else:
                    if dim >= len(arg.shape):
                        raise ValueError(f"Dimension {dim} is out of range for variable argument index {i} with shape {arg.shape}.")
                    if full_size is None:
                        full_size = arg.shape[dim]
                    elif full_size != arg.shape[dim]:
                        raise ValueError(f"Dimension {dim} of variable argument (shape {arg.shape} index {i} does not match the full size {full_size}.")
                # print(f"dim: {dim}, shape: {arg.shape}")
        
        # print(f"full_size: {full_size}, batch_size: {batch_size}")
        # Check if the batch_size is divisible by the full size
        if full_size % batch_size != 0:
            raise NotImplementedError(f"batch_size must be divisible by the full size {full_size}. Batch size {batch_size} given.")
        num_batches = full_size // batch_size 

        # Reshape inputs for batching
        reshaped_args = []
        for i, (dim, arg) in enumerate(zip(batch_dims, args)):
            arg:Variable
            if isinstance(dim, numbers.Integral):
                current_shape = arg.shape
                new_shape = []
                for j in range(len(current_shape)):
                    dim_size = current_shape[j]
                    if j == dim:
                        new_shape.append(num_batches)
                        new_shape.append(batch_size)
                    else:
                        new_shape.append(dim_size)
                reshaped_args.append(arg.reshape(*new_shape))
            else:
                reshaped_args.append(arg)

        # Finally do the batching loop
        with enter_loop(vals=[list(range(num_batches))]) as loop_builder:
            batch_ind = loop_builder.get_loop_indices()

            batched_args = []
            for i, (dim, arg) in enumerate(zip(batch_dims, args)):
                if isinstance(dim, numbers.Integral):
                    ndim = len(arg.shape)
                    indexes = [slice(None)] * ndim
                    indexes[dim] = batch_ind
                    batched_args.append(reshaped_args[i][tuple(indexes)])
                else:
                    batched_args.append(reshaped_args[i])

            # Call the function with the reshaped arguments
            single_outs = func(*batched_args, **kwargs)
        
        if not isinstance(single_outs, (tuple, list)):
            single_outs = (single_outs, )
            out_tuple = False
        else:
            out_tuple = True

        if output_types is None:
            output_types_post = (0,) * len(single_outs)
        else:
            if isinstance(output_types, numbers.Integral):
                output_types_post = (output_types,)
            elif isinstance(output_types, list):
                output_types_post = tuple(output_types)
            else:
                output_types_post = output_types
            if len(output_types) != len(single_outs):
                raise ValueError(f"output_types must be a tuple of integers that matches the outputs. {len(single_outs)} outputs given, {len(output_types)} output_types provided.")
        
        batched_outs = []
        for output_type, out in zip(output_types_post, single_outs):
            if not isinstance(out, Variable):
                raise TypeError(f"Function outputs must be a Variable. Type {get_type_string(out)} given.")
            if output_type == 0:
                batched_output = loop_builder.add_stack(out)
            elif output_type == 1:
                batched_output = loop_builder.add_pure_accrue(out)
            else:
                raise ValueError(f"output_types must be a tuple of integers that are either 0 or 1. {output_type} given.")
            batched_outs.append(batched_output)
        loop_builder.finalize()
        if out_tuple:
            return tuple(batched_outs)
        else:
            return batched_outs[0]
    return batched_func


class TestBatch(csdl_tests.CSDLTest):
    
    def test_functionality_simple(self,):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np
        
        compare_values = []

        def func(x,y,a):
            return x + y[0]*a
        
        # TEST 1
        batch_size = 3
        batched_func = csdl.experimental.batch_function(func, batch_size,[0,1, None])
        x_val, y_val = np.array([3.0, 4.0, 5.0]), np.array([[2.0, 3.0, 4.0]])
        x,y = csdl.Variable(name = 'x', value = x_val), csdl.Variable(name = 'y', value = y_val)
        s1 = batched_func(x,y,2.0)
        assert s1.shape == (1, batch_size), s1.shape
        real_value = func(x_val, y_val, 2.0).reshape((1, batch_size))
        compare_values += [csdl_tests.TestingPair(s1, real_value, tag = 's1')]

        # TEST 2
        batch_size = 2
        batched_func = csdl.experimental.batch_function(func, batch_size,[0,1, None])
        x_val, y_val = np.array([3.0, 4.0, 5.0, 6.0]), np.array([[2.0, 3.0, 4.0, -1.0]])
        x,y = csdl.Variable(name = 'x', value = x_val), csdl.Variable(name = 'y', value = y_val)
        s1 = batched_func(x,y,2.0)
        assert s1.shape == (2, batch_size), s1.shape
        real_value = func(x_val, y_val, 2.0).reshape((2, batch_size))
        compare_values += [csdl_tests.TestingPair(s1, real_value, tag = 's1')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_functionality(self,):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np
        
        compare_values = []

        # TEST 3,4,5,6: Double batch
        def matvec(theta_i,theta_j, vec):
            A_grid = csdl.outer(theta_i, theta_j)
            return A_grid @ vec
        n = 6
        batch_size_row = 2
        batch_size_col = 3
        theta_row_val, theta_col_val, vec_val = np.sin(np.arange(n)), np.cos(-np.arange(n)/3+0.1), 1.0/(np.arange(n)+1.0)
        theta_row, theta_col, vec = csdl.Variable(value = theta_row_val), csdl.Variable(value = theta_col_val), csdl.Variable(value = vec_val)
        batched_func_row = csdl.experimental.batch_function(matvec, batch_size_row,[0, None, None])
        Av_rows = batched_func_row(theta_row, theta_col, vec)

        # Compute real matvec
        A_np = np.outer(theta_row_val, theta_col_val)
        Av_np = A_np @ vec_val

        # TEST 3: Row batching only
        batched_func_row = csdl.experimental.batch_function(matvec, batch_size_row,[0, None, None])
        Av_rows = batched_func_row(theta_row, theta_col, vec).reshape((n,))
        
        # TEST 4: Col batching only
        batched_func_col = csdl.experimental.batch_function(matvec, batch_size_col,[None, 0, 0])
        Av_cols = csdl.sum(batched_func_col(theta_row, theta_col, vec), axes=(0,))

        # TEST 5: Grid batching row->col
        batched_func_row = csdl.experimental.batch_function(matvec, batch_size_row,[0, None, None])
        batched_func_row_col = csdl.experimental.batch_function(batched_func_row, batch_size_col,[None, 0, 0]) 
        Av_grid_rc = csdl.sum(batched_func_row_col(theta_row, theta_col, vec), axes=(0,)).reshape((n,))

        # TEST 6: Grid batching col->row
        batched_func_col = csdl.experimental.batch_function(matvec, batch_size_col,[None, 0, 0])
        batched_func_col_row = csdl.experimental.batch_function(batched_func_col, batch_size_row,[0, None, None]) 
        Av_grid_cr = csdl.sum(batched_func_col_row(theta_row, theta_col, vec), axes=(1,)).reshape((n,))

        # TEST 7: Grid batching col->col
        batched_func_col_inner = csdl.experimental.batch_function(matvec, 1,[None, 0, 0])
        batched_func_col_outer = csdl.experimental.batch_function(batched_func_col_inner, batch_size_col,[None, 0, 0]) 
        Av_grid_cc = csdl.sum(batched_func_col_outer(theta_row, theta_col, vec), axes=(0,1)).reshape((n,))

        print('Numpy:      ',Av_np)
        print('row:        ',Av_rows.value)
        print('col:        ',Av_cols.value)
        print('row->col:   ',Av_grid_rc.value)
        print('col->row:   ',Av_grid_cr.value)
        print('col->col:   ',Av_grid_cc.value)

        compare_values += [csdl_tests.TestingPair(Av_rows, Av_np, tag = 'row_batch')]
        compare_values += [csdl_tests.TestingPair(Av_cols, Av_np, tag = 'col_batch')]
        compare_values += [csdl_tests.TestingPair(Av_grid_rc, Av_np, tag = 'grid_batch')]
        compare_values += [csdl_tests.TestingPair(Av_grid_cr, Av_np, tag = 'grid_batch')]
        compare_values += [csdl_tests.TestingPair(Av_grid_cc, Av_np, tag = 'doublecol_batch')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_functionality_double_index(self,):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np
        
        compare_values = []

        # TEST 3,4,5,6: Double batch
        def matvec(A_grid, vec):
            print(f'SHAPES: A_grid: {A_grid.shape} vec: {vec.shape}')
            return A_grid @ vec

        n = 12
        batch_size_row = 2
        batch_size_col = 3
        theta_row_val, theta_col_val, vec_val = np.sin(np.arange(n)), np.cos(-np.arange(n)/3+0.1), 1.0/(np.arange(n)+1.0)
        theta_row, theta_col, vec = csdl.Variable(value = theta_row_val), csdl.Variable(value = theta_col_val), csdl.Variable(value = vec_val)

        # Compute real matvec
        A_np = np.outer(theta_row_val, theta_col_val)
        Av_np = A_np @ vec_val
        AA_np = A_np @ A_np

        A_full = csdl.outer(theta_row, theta_col)
        # TEST 8: Row batching only
        batched_func_row = csdl.experimental.batch_function(matvec, batch_size_row,[0, None])
        Av_rows = batched_func_row(A_full, vec).reshape((n,))

        # TEST 9: Col batching only
        batched_func_col = csdl.experimental.batch_function(matvec, batch_size_col,[1, 0])
        Av_cols = csdl.sum(batched_func_col(A_full, vec), axes=(0,))

        # TEST 9.1: Col batching only using output_sum
        batched_func_col = csdl.experimental.batch_function(matvec, batch_size_col,[1, 0], output_types = (1,))
        Av_cols_s = batched_func_col(A_full, vec)

        # TEST 10: Grid batching row->col
        batched_func_row = csdl.experimental.batch_function(matvec, batch_size_row,[0, None])
        batched_func_row_col = csdl.experimental.batch_function(batched_func_row, batch_size_col,[1, 0])
        Av_grid_rc = csdl.sum(batched_func_row_col(A_full, vec), axes=(0,)).reshape((n,))

        # TEST 11: Col batching only on matmat
        batched_func_col_matmat = csdl.experimental.batch_function(matvec, batch_size_col,[1, 0])
        asdf = batched_func_col_matmat(A_full, A_full)
        print(asdf.shape)
        AA_cols_matmat = csdl.sum(batched_func_col_matmat(A_full, A_full), axes=(0,))

        print('Numpy:      ',Av_np)
        print('row:        ',Av_rows.value)
        print('col:        ',Av_cols.value)
        print('col s:      ',Av_cols_s.value)
        print('row->col:   ',Av_grid_rc.value)
        print('Numpy matmat:',AA_np)
        print('matmat col: ', AA_cols_matmat.value)
        compare_values += [csdl_tests.TestingPair(Av_rows, Av_np, tag = 'row_batch')]
        compare_values += [csdl_tests.TestingPair(Av_cols, Av_np, tag = 'col_batch')]
        compare_values += [csdl_tests.TestingPair(Av_cols_s, Av_np, tag = 'col_batch_s')]
        compare_values += [csdl_tests.TestingPair(Av_grid_rc, Av_np, tag = 'grid_batch')]
        compare_values += [csdl_tests.TestingPair(AA_cols_matmat, AA_np, tag = 'col_batch_matmat')]
        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_errors(self):
        self.prep()

        import csdl_alpha as csdl

        # Check if the function is callable
        with pytest.raises(TypeError):
            batched_func = csdl.experimental.batch_function(1,1,(1,))

        def dummy_func(x):
            return x*2.0
        
        # Check if the batch_dims is a tuple of integers that matches the inputs
        batched_func = csdl.experimental.batch_function(dummy_func,1,[0,])
        batched_func = csdl.experimental.batch_function(dummy_func,1,(0,))
        batched_func = csdl.experimental.batch_function(dummy_func,1,0)


        # Check if the batch_dims is a tuple of integers that matches the inputs
        with pytest.raises(TypeError):
            batched_func = csdl.experimental.batch_function(dummy_func,1,'p')
        with pytest.raises(ValueError):
            batched_func = csdl.experimental.batch_function(dummy_func,1,(0,0))
            batched_func(1)

        # Check if the batch_dims is a tuple of integers that matches the inputs
        def dummy_func(x,y):
            return x*2.0*y
        batched_func = csdl.experimental.batch_function(dummy_func,1,(0,'str'))
        with pytest.raises(TypeError):
            batched_func(csdl.Variable(value = 1.0), csdl.Variable(value = 2.0))
        batched_func = csdl.experimental.batch_function(dummy_func,1,(None,0))
        batched_func(csdl.Variable(value = 1.0), csdl.Variable(value = 2.0))
        batched_func(0.0, csdl.Variable(value = 2.0))
        with pytest.raises(TypeError):
            batched_func(csdl.Variable(value = 2.0), 0.0)
        batched_func = csdl.experimental.batch_function(dummy_func,1,(None,5))
        with pytest.raises(ValueError):
            batched_func(csdl.Variable(value = 1.0), csdl.Variable(value = np.ones((2,2,2))))

        batched_func = csdl.experimental.batch_function(dummy_func,4,(0,1))
        with pytest.raises(ValueError):
            batched_func(csdl.Variable(value = np.ones((4,2,2))), csdl.Variable(value = np.ones((2,5,2))))
        with pytest.raises(NotImplementedError):
            batched_func(csdl.Variable(value = np.ones((5,2,2))), csdl.Variable(value = np.ones((2,5,2))))

if __name__ == '__main__':
    test = TestBatch()
    test.overwrite_backend = 'inline'
    # test.overwrite_backend = 'jax'
    test.test_functionality_simple()
    test.test_functionality()
    test.test_functionality_double_index()
    # test.test_errors()


    # ######################## TEST OF TESTS ######################## :
    # import numpy as np
    # n = 6
    # batch_per_dim = 3
    # batch_size = n // batch_per_dim

    # theta_row = np.arange(n)
    # theta_col = -np.arange(n)/3+0.1
    # full_A = np.outer(theta_row, theta_col)

    # def compute_A_grid(theta_i, theta_j):
    #     return np.outer(theta_i, theta_j)
    
    # A_to_assemble = np.zeros((n, n))
    # for i in range(batch_size):
    #     theta_i = theta_row[i*batch_per_dim:(i+1)*batch_per_dim]
    #     for j in range(batch_size):
    #         theta_j = theta_col[j*batch_per_dim:(j+1)*batch_per_dim]
    #         A_subgrid = compute_A_grid(theta_i, theta_j)
    #         A_to_assemble[i*batch_per_dim:(i+1)*batch_per_dim, j*batch_per_dim:(j+1)*batch_per_dim] = A_subgrid

    # # print(A_to_assemble)
    # # print(full_A)
    # assert np.allclose(A_to_assemble, full_A), "The assembled matrix does not match the expected full matrix."