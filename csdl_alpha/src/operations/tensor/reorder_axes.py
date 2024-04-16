from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike

import numpy as np

@set_properties(linear=True)
class ReorderAxes(Operation):
    '''
    Reorders the axes of the input tensor as per the specified `action`.
    '''
    def __init__(self, x, out_shape, out_axes):
        super().__init__(x)
        self.name = 'reorder_axes'
        out_shapes = (out_shape,)
        self.set_dense_outputs(out_shapes)
        self.out_axes = out_axes

    def compute_inline(self, x):
        return np.transpose(x, self.out_axes)
    
    def evaluate_jacobian(self, x):
        rows = np.arange(x.size)
        cols = np.transpose(np.arange(x.size).reshape(x.shape), axes=self.out_axes)

        return csdl.Constant(rows=rows.flatten(), cols=cols.flatten(), val = 1.)
        
def reorder_axes(x, action):
    '''
    Reorders the axes of the input tensor as per the specified `action`.
    For example, `action='ijk->kji'` will transpose the input 3D tensor.
    The `action` argument is optional if the input is a scalar since the
    scalar will be simply broadcasted to the specified `out_shape`.

    Parameters
    ----------
    x : Variable or np.ndarray
        Input tensor that needs to have its axes reordered.
    action : str
        Specifies how the axes of the input tensor needs to be reordered,
        e.g.,`'ij->ji'` transposes the input matrix.

    Returns
    -------
    Variable
        Axes-reordered output tensor as per specfied `action`.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x_val = np.array([[1., 2., 3.], \
                          [4., 5., 6.]])
    >>> x = csdl.Variable(value = x_val)
    >>> y1 = csdl.reorder_axes(x, action='ij->ji')
    >>> y1.value
    array([[1., 4.],
           [2., 5.],
           [3., 6.]])
    
    Reorder the axes of a 3D tensor:
           
    >>> x_val = np.arange(24).reshape(2,3,4)
    >>> x_val
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> y2 = csdl.reorder_axes(x_val, action='ijk->kij')
    >>> y2.value
    array([[[ 0.,  4.,  8.],
            [12., 16., 20.]],
    <BLANKLINE>
           [[ 1.,  5.,  9.],
            [13., 17., 21.]],
    <BLANKLINE>
           [[ 2.,  6., 10.],
            [14., 18., 22.]],
    <BLANKLINE>
           [[ 3.,  7., 11.],
            [15., 19., 23.]]])
    '''

    x = variablize(x)

    if x.size == 1:
        raise ValueError('Cannot reorder axes of a scalar.')
    if action is None:
        raise ValueError('Cannot reorder axes of a tensor without "action" specified.')
    if not isinstance(action, str):
        raise ValueError('"action" must be a string.')
    if '->' not in action:
        raise ValueError('Invalid action string. Use "->" to separate the input and output subscripts.')

    in_str, out_str = action.split('->')
    in_shape = x.shape
    if len(in_str) != len(in_shape):
        raise ValueError('Input tensor shape does not match the input string in the action.')
    if len(out_str) != len(in_str):
        raise ValueError('Number of axes in the input and output must be the same.')
    
    if not all(in_str.count(char) == 1 for char in in_str):
        raise ValueError('Each character in the input string must appear exactly once.')
    if not all(out_str.count(char) == 1 for char in out_str):
        raise ValueError('Each character in the output string must appear exactly once.')      
    if not all(out_str.count(char) == 1 for char in in_str):
        raise ValueError('Each character in the input string must appear exactly once in the output string.')
    
    out_shape = tuple([in_shape[in_str.index(char)] for char in out_str])
    out_axes = tuple([in_str.index(char) for char in out_str])
        
    op = ReorderAxes(x, out_shape, out_axes)
    
    return op.finalize_and_return_outputs()

class TestReorderAxes(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.array([[1., 2., 3.], \
                          [4., 5., 6.]])
        y_val = np.arange(24).reshape(2,3,4)

        x = csdl.Variable(name = 'x', value = x_val)

        compare_values = []
        # transpose of a matrix variable
        s1 = csdl.reorder_axes(x, action='ij->ji')
        t1 = np.transpose(x_val)
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # reorder axes of a 3D tensor constant
        s2 = csdl.reorder_axes(y_val, action='ijk->kij')
        t2 = np.transpose(y_val, (2,0,1))
        compare_values += [csdl_tests.TestingPair(s2, t2, tag = 's2')]

        self.run_tests(compare_values = compare_values,)

    def test_example(self,):
        self.docstest(reorder_axes)

if __name__ == '__main__':
    test = TestReorderAxes()
    test.test_functionality()
    test.test_example()