from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests

import numpy as np

class BLockMatrix(Operation):
    '''
    Assemble a block matrix from a list of matrices or a list of lists.
    '''

    def __init__(self, *args, num_row_blocks = None, shape=None):
        super().__init__(*args)
        self.name = 'block_matrix'
        self.num_row_blocks = num_row_blocks
        out_shapes = (shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, *args):
        if self.num_row_blocks is None:
            return np.block([x for x in args])
        else:
            l = self.num_row_blocks
            row_idx = np.cumsum([0] + l)
            return np.block([list(args[row_idx[i]:row_idx[i+1]]) for i in range(len(l))])

def blockmat(l):
    """
    doc strings
    """
    list_in_list = any(isinstance(x, list) for x in l)
    all_are_list = all(isinstance(x, list) for x in l)
    if list_in_list != all_are_list:
        raise ValueError('List depths are mismatched.')
    
    if list_in_list:
        num_rows = sum([x[0].shape[0] for x in l])
        num_cols = sum([x.shape[1] for x in l[0]])
        
        for i, elem_l in enumerate(l):
            if not all([x.shape[0] == elem_l[0].shape[0] for x in elem_l]):
                raise ValueError('Number of rows are not the same for the blocks in the {i}th row.')
            if sum([x.shape[1] for x in elem_l]) != num_cols:
                raise ValueError('Total number of columns are not the same for the 0th and {i}th block rows.')

        args = [variablize(y) for x in l for y in x ]
        num_row_blocks = [len(x) for x in l]
    else:
        num_rows = l[0].shape[0]
        num_cols = sum([x.shape[1] for x in l])

        if all([x.shape[0] != num_rows for x in l]):
            raise ValueError('Number of rows are not the same for all the blocks.')
        args = [variablize(x) for x in l]
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
        x_val = 3.0*np.ones((2,3))
        y_val = np.zeros((2,5))
        z_val = np.ones((4,8))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)

        compare_values = []
        # create a block row matrix
        b1 = csdl.blockmat([x,y])
        t1 = np.block([x_val, y_val])
        compare_values += [csdl_tests.TestingPair(b1, t1, tag = 'b1')]

        # Create a block matrix with block rows and columns
        b2 = csdl.blockmat([[x,y], [z]])
        t2 = np.block([[x_val, y_val], [z_val]])
        compare_values += [csdl_tests.TestingPair(b2, t2, tag = 'b1')]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.prep()

        # docs:entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.ones((2,3))
        y_val = np.zeros((2,5))
        z_val = np.ones((4,8))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)

        # create a block row matrix
        b1 = csdl.blockmat([x,y])
        print(b1.value)

        # Create a block matrix with block rows and columns
        b2 = csdl.blockmat([[x,y], [z]])
        print(b2.value)

        # Create a block matrix with block rows and columns using constant blocks
        b3 = csdl.blockmat([[x,y_val], [z]])
        print(b3.value)
        # docs:exit

        compare_values = []
        t1 = np.block([x_val, y_val])
        t2 = np.block([[x_val, y_val], [z_val]])

        compare_values += [csdl_tests.TestingPair(b1, t1)]
        compare_values += [csdl_tests.TestingPair(b2, t2)]
        compare_values += [csdl_tests.TestingPair(b3, t2)]

        self.run_tests(compare_values = compare_values,)


if __name__ == '__main__':
    test = TestBlockMat()
    test.test_functionality()
    test.test_example()