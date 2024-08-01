from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike
import warnings

import numpy as np

@set_properties(linear=True)
class ScalarExpand(Operation):
    '''
    Expands the input scalar to the specified `out_shape`.
    '''
    def __init__(self, x,  out_shape):
        super().__init__(x)
        self.name = 'scalar_expand'
        self.out_shape = out_shape
        out_shapes = (out_shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x):
        return np.broadcast_to(x, self.out_shape)
    
    def compute_jax(self, x):
        import jax.numpy as jnp
        return jnp.broadcast_to(x, self.out_shape)  
    
    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            import csdl_alpha as csdl
            cotangents.accumulate(x, csdl.sum(cotangents[y]))
        
@set_properties(linear=True)
class TensorExpand(Operation):
    '''
    Expands the input tensor to the specified `out_shape` by 
    repeating the tensor along certain axes determined fom the `action` argument.
    '''
    def __init__(self, x, out_shape, ones_shape, einsum_str):
        super().__init__(x)
        self.name = 'tensor_expand'
        self.out_shape = out_shape
        self.ones_shape = ones_shape
        self.einsum_str = einsum_str

        out_shapes = (out_shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x):
        # NOTE : if csdl.einsum is implemented using csdl.[sum, expand, reorder_axes, mult] later,
        # then the line below should never call csdl.einsum since it just creates recursive calls.
        # print(self.einsum_str, (self.ones_shape))
        # exit()
        return np.einsum(self.einsum_str, x, np.ones(self.ones_shape))
    
    def compute_jax(self, x):
        import jax.numpy as jnp
        # NOTE : if csdl.einsum is implemented using csdl.[sum, expand, reorder_axes, mult] later,
        # then the line below should never call csdl.einsum since it just creates recursive calls.
        return jnp.einsum(self.einsum_str, x, jnp.ones(self.ones_shape))

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            import csdl_alpha as csdl
            in_str, out_str = self.einsum_str.split('->')
            in_str, ones_str = in_str.split(',')
            
            sum_str = out_str + '->' + in_str
            vjp = csdl.einsum(cotangents[y], action=sum_str)
            cotangents.accumulate(x, vjp)

    def evaluate_jacobian(self, x):
        # NOTE : if csdl.einsum is implemented using csdl.[sum, expand, reorder_axes, mult] later,
        # then the line below should never call csdl.einsum since it just creates recursive calls.
        rows = np.arange(np.prod(self.out_shape)).reshape(self.out_shape)
        cols = np.einsum(self.einsum_str, np.arange(x.size).reshape(x.shape), np.ones(self.ones_shape))

        return csdl.Constant(rows=rows.flatten(), cols=cols.flatten(), val = 1.)
        
def expand(x, out_shape, action=None):
    '''
    Expands the input scalar/tensor to the specified `out_shape` by 
    repeating the tensor along certain axes determined fom the `action` argument.
    For example, `action='i->ijk'` will expand a 1D tensor to a 3D tensor by repeating
    the input tensor along two new axes.
    The `action` argument is optional if the input is a scalar since the
    scalar will be simply broadcasted to the specified `out_shape`.

    Parameters
    ----------
    x : VariableLike
        Input scalar/tensor that needs to be expanded.
    out_shape : tuple of int
        Desired shape of the expanded output tensor.
    action : str, default=None
        Specifies the action to be taken when expanding the tensor,
        e.g.,`'i->ij'` expands a vector to a matrix by repeating the 
        input vector rowwise.

    Returns
    -------
    Variable
        Expanded output tensor as per the specified `out_shape` and `action`.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = 3.0)
    >>> y1 = csdl.expand(x, out_shape=(2,3))
    >>> y1.value
    array([[3., 3., 3.],
           [3., 3., 3.]])
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y2 = csdl.expand(x, out_shape=(2,3), action='i->ji')
    >>> y2.value
    array([[1., 2., 3.],
           [1., 2., 3.]])
    >>> y3 = csdl.expand(x, out_shape=(3,2), action='i->ij')
    >>> y3.value
    array([[1., 1.],
           [2., 2.],
           [3., 3.]])
    >>> y4 = csdl.expand(x, out_shape=(4,3,2), action='i->lij')
    >>> y4.value
    array([[[1., 1.],
            [2., 2.],
            [3., 3.]],
    <BLANKLINE>
           [[1., 1.],
            [2., 2.],
            [3., 3.]],
    <BLANKLINE>
           [[1., 1.],
            [2., 2.],
            [3., 3.]],
    <BLANKLINE>
           [[1., 1.],
            [2., 2.],
            [3., 3.]]])
    '''

    x = variablize(x)

    if not isinstance(out_shape, tuple):
        raise ValueError('"out_shape" must be a tuple.')

    if x.size != 1:
        if action is None:
            raise ValueError('Cannot expand a tensor without "action" specified.')
        else:
            if not isinstance(action, str):
                raise TypeError('"action" must be a string.')
            if '->' not in action:
                raise ValueError('Invalid action string. Use "->" to separate the input and output subscripts.')
            
            in_str, out_str = action.split('->')
            in_shape = x.shape
            if len(in_str) != len(in_shape):
                raise ValueError(f'Input tensor shape {in_shape} does not match the input string \'{in_str}\' in the action.')
            if len(out_str) != len(out_shape):
                raise ValueError('Output tensor shape does not match the output string in the action.')

            if not all(in_str.count(char) == 1 for char in in_str):
                raise ValueError('Each character in the input string must appear exactly once.')
            if not all(out_str.count(char) == 1 for char in out_str):
                raise ValueError('Each character in the output string must appear exactly once.')      
            if not all(out_str.count(char) == 1 for char in in_str):
                raise ValueError('Each character in the input string must appear exactly once in the output string.')
            
            if in_shape != tuple([out_shape[out_str.index(char)] for char in in_str]):
                raise ValueError(f'Input tensor shape {in_shape} is not compatible with the output shape {out_shape} specified in the action.')
            
            ones_str   = ''.join([char for char in out_str if char not in in_str])
            ones_shape = tuple([out_shape[out_str.index(char)] for char in ones_str])
            einsum_str = in_str + ',' + ones_str + '->' + out_str
            
        op = TensorExpand(x, out_shape, ones_shape, einsum_str)

    else:
        if action is not None:
            warnings.warn('"action" will have no effect when expanding a scalar.')
        
        x = x.flatten()
        op = ScalarExpand(x, out_shape)
    
    return op.finalize_and_return_outputs()

class TestExpand(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0
        y_val = np.array([1.0, 2.0, 3.0])
        y_tensor_val = np.arange(60).reshape(3,4,5)

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        y_tensor = csdl.Variable(name = 'yt', value = y_tensor_val)

        compare_values = []
        # expand a scalar constant
        s1 = csdl.expand(x_val, out_shape=(2,3,4))
        t1 = x_val * np.ones((2,3,4))
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # expand a scalar variable
        s2 = csdl.expand(x, out_shape=(2,3,4))
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # expand a tensor variable
        s3 = csdl.expand(y_tensor, out_shape=(4,3,4,2,5), action='ijk->aijbk')
        t3 = np.einsum('ijk,aijbk->aijbk', y_tensor_val, np.ones((4,3,4,2,5)))
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # expand a tensor variable
        s3 = csdl.expand(y_tensor, out_shape=(5,2,3,1,4), action='ijk->kaibj')
        t3 = np.einsum('ijk,kaibj->kaibj', y_tensor_val, np.ones((5,2,3,1,4)))
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # expand a vector variable
        s3 = csdl.expand(y, out_shape=(3,4), action='j->jk')
        t3 = np.einsum('j,jk->jk', y_val, np.ones((3,4)))
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # expand a vector constant
        s4 = csdl.expand(y_val, out_shape=(2,3,4), action='j->ijk')
        t4 = np.einsum('j,ijk->ijk', y_val, np.ones((2,3,4)))
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_example(self,):
        self.docstest(expand)

if __name__ == '__main__':
    test = TestExpand()
    test.test_functionality()
    test.test_example()