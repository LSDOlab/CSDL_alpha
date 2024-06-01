from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests

import numpy as np
import pytest

class BLockMatrix(Operation):
    '''
    Assemble a block matrix from a list of matrices or a list of lists.
    '''

    def __init__(self, *args, num_row_blocks = None, shape=None):
        super().__init__(*args)
        self.name = 'block_matrix'
        self.num_row_blocks:list[int] = num_row_blocks
        out_shapes = (shape,)
        self.set_dense_outputs(out_shapes)

        # Build indices for each block
        self.indices = []
        if self.num_row_blocks is None:
            left_index = 0
            for arg in args:
                right_index = left_index + arg.shape[1]
                self.indices.append((left_index, right_index))
                left_index = right_index
        else:
            current_arg_index = 0
            lower_index = 0

            for cur_block_row_ind in range(len(self.num_row_blocks)):
                arg = args[current_arg_index]
                left_index = 0
                upper_index = lower_index + arg.shape[0]

                for cur_block_col_ind in range(self.num_row_blocks[cur_block_row_ind]):
                    # Current variable
                    arg = args[current_arg_index]
                    right_index = left_index + arg.shape[1]

                    # Save indices                  
                    self.indices.append((left_index, right_index, lower_index, upper_index))
                    
                    # update column indices
                    left_index = right_index
                    current_arg_index += 1 

                # update row indices
                lower_index = upper_index
            
            # print(self.indices)
            # exit()


    def compute_inline(self, *args):
        if self.num_row_blocks is None:
            return np.block([x for x in args])
        else:
            l = self.num_row_blocks
            row_idx = np.cumsum([0] + l)
            return np.block([list(args[row_idx[i]:row_idx[i+1]]) for i in range(len(l))])

    def evaluate_vjp(self, cotangents, *inputs_and_block):
        inputs = inputs_and_block[:-1]
        block = inputs_and_block[-1]

        block_out = cotangents[block]
        for i, input in enumerate(inputs):
            if cotangents.check(input):
                if self.num_row_blocks is None:
                    left = self.indices[i][0]
                    right = self.indices[i][1]
                    cotangents.accumulate(input, block_out[:, left:right])
                else:
                    left = self.indices[i][0]
                    right = self.indices[i][1]
                    lower = self.indices[i][2]
                    upper = self.indices[i][3]
                    cotangents.accumulate(input, block_out[lower:upper, left:right])

def blockmat(l)->Variable:
    """
    Assemble a block matrix from a list or list of lists of matrices.

    Parameters
    ----------
    l : list or list of lists of Variable or np.ndarray objects
        List or a list of lists of matrices to assemble into a block matrix.

    Returns
    -------
    Variable
        Block matrix assembled from the input list.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x_val = 3.0*np.ones((2,3))
    >>> z_val = np.ones((1,5))
    >>> x = csdl.Variable(name = 'x', value = x_val)
    >>> z = csdl.Variable(name = 'z', value = z_val)

    Create a block row matrix

    >>> b1 = csdl.blockmat([x, np.zeros((2,2))])
    >>> b1.value
    array([[3., 3., 3., 0., 0.],
           [3., 3., 3., 0., 0.]])

    Create a block matrix with block rows and columns

    >>> b2 = csdl.blockmat([[x, np.zeros((2,2))], [z]])
    >>> b2.value
    array([[3., 3., 3., 0., 0.],
           [3., 3., 3., 0., 0.],
           [1., 1., 1., 1., 1.]])

    """
    list_in_list = any(isinstance(x, list) for x in l)
    all_are_list = all(isinstance(x, list) for x in l)
    if list_in_list != all_are_list:
        raise ValueError('List depths are mismatched.')
    
    if list_in_list:
        num_rows = sum([x[0].shape[0] for x in l])
        num_cols = sum([x.shape[1] for x in l[0]])
        
        for i, elem_l in enumerate(l):
            num_rows_current = elem_l[0].shape[0]
            for x in elem_l:
                if x.shape[0] != num_rows_current:
                    raise ValueError(f'Number of columns are not the same for the blocks in the {i}th row. {x.shape[0]} given, {num_rows_current} expected')
            current_num_cols = sum([x.shape[1] for x in elem_l])
            if current_num_cols != num_cols:
                raise ValueError(f'Total number of columns are not the same for the 0th and {i}th block rows. {current_num_cols} given, {num_cols} expected')

        args = [validate_and_variablize(y) for x in l for y in x ]
        num_row_blocks = [len(x) for x in l]
    else:
        num_rows = l[0].shape[0]
        num_cols = sum([x.shape[1] for x in l])

        for x in l:
            if x.shape[0] != num_rows:
                raise ValueError(f'Number of rows are not the same for all the blocks. {x.shape[0]} given, {num_rows} expected')

        args = [validate_and_variablize(x) for x in l]
        num_row_blocks = None

    op = BLockMatrix(*args, num_row_blocks=num_row_blocks, shape=(num_rows, num_cols))
    
    return op.finalize_and_return_outputs()


class TestBlockMat(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.arange(6).reshape(2,3)+2.0
        y_val = np.arange(10).reshape(2,5)
        z_val = np.arange(32).reshape(4,8)*0.5 
        w_val = np.ones((2,1))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        w = csdl.Variable(name = 'w', value = w_val)

        compare_values = []

        # create a SINGLE ROW block row matrix
        b1 = csdl.blockmat([x,y])
        t1 = np.block([x_val, y_val])
        compare_values += [csdl_tests.TestingPair(b1, t1, tag = 'b1')]

        b1 = csdl.blockmat([x,y,w])
        t1 = np.block([x_val, y_val, w_val])
        compare_values += [csdl_tests.TestingPair(b1, t1, tag = 'b2')]

        b1 = csdl.blockmat([x])
        t1 = np.block([x_val])
        compare_values += [csdl_tests.TestingPair(b1, t1, tag = 'b3')]

        # Create a block matrix with block rows and columns
        b2 = csdl.blockmat([[x,y], [z]])
        t2 = np.block([[x_val, y_val], [z_val]])
        compare_values += [csdl_tests.TestingPair(b2, t2, tag = 'b4')]

        b2 = csdl.blockmat([[x,y], [z], [y,x]])
        t2 = np.block([[x_val, y_val], [z_val], [y_val, x_val]])
        compare_values += [csdl_tests.TestingPair(b2, t2, tag = 'b4')]

        b2 = csdl.blockmat([[x,x], [x, x]])
        t2 = np.block([[x_val, x_val], [x_val, x_val]])
        compare_values += [csdl_tests.TestingPair(b2, t2, tag = 'b5')]

        b2 = csdl.blockmat([[x.T()], [y.T()], [w.T()]])
        t2 = np.block([[x_val.T], [y_val.T], [w_val.T]])
        compare_values += [csdl_tests.TestingPair(b2, t2, tag = 'b5')]

        self.run_tests(
            compare_values = compare_values,
            verify_derivatives=True
        )


    def test_errors(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.arange(6).reshape(2,3)+2.0
        y_val = np.arange(10).reshape(2,5)
        z_val = np.arange(32).reshape(4,8)*0.5 
        w_val = np.ones((2,1))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        w = csdl.Variable(name = 'w', value = w_val)

        with pytest.raises(ValueError):
            b1 = csdl.blockmat([x, y.T()])

        with pytest.raises(ValueError):
            b1 = csdl.blockmat([[z], [x, y.T()]])

        with pytest.raises(ValueError):
            b1 = csdl.blockmat([[z.T()], [x, y]])

        with pytest.raises(ValueError):
            b1 = csdl.blockmat([[z], [x, y, x]])

    def test_example(self,):
        self.docstest(blockmat)


if __name__ == '__main__':
    test = TestBlockMat()
    test.test_functionality()
    test.test_errors()
    # test.test_example()