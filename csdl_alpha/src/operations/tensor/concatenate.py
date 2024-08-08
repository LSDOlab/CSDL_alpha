from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize, get_type_string
import csdl_alpha.utils.testing_utils as csdl_tests

import numpy as np
import pytest

class Concatenate(Operation):
    '''
    Assemble a block matrix from a list of matrices or a list of lists.
    '''

    def __init__(self, *args:list[Variable], axis:int, shape:tuple):
        super().__init__(*args)
        self.name = 'concat'
        self.axis:int = axis

        # Compute the location of the axis for each input
        args:list[Variable] = args
        axis_size = [0]
        for arg in args:
            axis_size.append(axis_size[-1]+arg.shape[axis])
        
        self.axis_size = axis_size
        self.shape_prefix = shape[:axis]
        self.shape_suffix = shape[axis+1:]
        out_shape = self.shape_prefix + (axis_size[-1],) + self.shape_suffix
        self.set_dense_outputs((out_shape,))

    def compute_inline(self, *args):
        return np.concatenate([x for x in args], axis = self.axis)

    def compute_jax(self, *args):
        import jax.numpy as jnp
        return jnp.concatenate([x for x in args], axis = self.axis)

    def evaluate_vjp(self, cotangents, *inputs_and_stack):
        inputs = inputs_and_stack[:-1]
        concat = inputs_and_stack[-1]

        concat_cot:Variable = cotangents[concat]
        index_prefix = [slice(None) for _ in self.shape_prefix]
        index_suffix = [slice(None) for _ in self.shape_suffix]
        for i,input in enumerate(inputs):
            if cotangents.check(input):
                # index the dimension 'self.axis' of concat_cot:
                cot = concat_cot[tuple(
                    index_prefix + [slice(self.axis_size[i], self.axis_size[i+1])] + index_suffix
                    )]
                cotangents.accumulate(input, cot)

def concatenate(arrays:tuple[Variable], axis:int = 0)->Variable:
    """
    concatenate arrays along an axis.

    Parameters
    ----------
    arrays : tuple of arrays to stack. Each array must have the same shape in all but the specified dimension.
    axis : int, optional
        The axis along which to concatenate the arrays. The default is 0.

    Returns
    -------
    Variable
        The concatenated array

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x_val = 3.0*np.ones((3,2))
    >>> z_val = np.ones((1,2))
    >>> x = csdl.Variable(name = 'x', value = x_val)
    >>> z = csdl.Variable(name = 'z', value = z_val)
    >>> y = csdl.concatenate((x, z), axis = 0)
    >>> y.value
    array([[3., 3.],
           [3., 3.],
           [3., 3.],
           [1., 1.]])
    >>> y = csdl.concatenate((x.flatten(), z.flatten()))
    >>> y.value
    array([3., 3., 3., 3., 3., 3., 1., 1.])
    """
    if not isinstance(arrays, (tuple, list)):
        raise ValueError('Input must be a list or tuple of variables to concatenate')
    if len(arrays) < 2:
        raise ValueError('Input list must have at least two variables to concatenate')

    args = [validate_and_variablize(x) for x in arrays]
    shape = args[0].shape
    if not isinstance(axis, (int, np.integer)):
        raise TypeError(f'Axis must be an integer. Got {get_type_string(axis)}')
    # check bounds of axis
    if axis >= len(shape) or axis < -len(shape):
        raise ValueError(f'Axis out of bounds. Axis must be in the range {-len(shape), len(shape)-1}. Got {axis}')
    if axis < 0:
        axis += len(shape)

    # Check to make sure all shapes are consistent except for the axis
    check_shape = shape[:axis] + shape[axis+1:]
    for ind, arg in enumerate(args):
        compare_shape = arg.shape[:axis] + arg.shape[axis+1:]
        if compare_shape != check_shape:
            raise ValueError(f'All arguments must have the same size across all dimensions except for dimension \'{axis}\'. Argument {ind} with shape {arg.shape} is incompatible with {shape}')

    # Create operation
    op = Concatenate(*args, axis = axis, shape = shape)
    return op.finalize_and_return_outputs()


class TestConcatenate(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep(inline = True)

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = np.arange(6).reshape(2,3)+2.0
        y_val = np.arange(3).reshape(1,3)
        a_val = np.arange(320).reshape(4,8,10)*0.5 
        b_val = np.arange(160).reshape(2,8,10)*0.5+7
        c_val = np.arange(80).reshape(1,8,10)*0.5+3
        v1_val = np.arange(10).reshape(10,)*12
        v2_val = np.arange(10).reshape(10,)*3+6


        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        a = csdl.Variable(name = 'a', value = a_val)
        b = csdl.Variable(name = 'b', value = b_val)
        c = csdl.Variable(name = 'c', value = c_val)
        v1 = csdl.Variable(name = 'v1', value = v1_val)
        v2 = csdl.Variable(name = 'v2', value = v2_val)

        compare_values = []

        # matrix from matrix
        c1 = csdl.concatenate((x, y))
        n1 = np.concatenate((x_val, y_val))
        compare_values += [csdl_tests.TestingPair(c1, n1, tag = 'v1')]
        
        c1 = csdl.concatenate((x, y, x))
        n1 = np.concatenate((x_val, y_val, x_val))
        compare_values += [csdl_tests.TestingPair(c1, n1, tag = 'v1')]
        
        # matrix from matrix
        c1 = csdl.concatenate((x, y),axis=-2)
        n1 = np.concatenate((x_val, y_val),axis=-2)
        compare_values += [csdl_tests.TestingPair(c1, n1, tag = 'v-2')]

        # 3d tensors
        c2 = csdl.concatenate((a, b, c), axis = 0)
        n2 = np.concatenate((a_val, b_val, c_val), axis = 0)
        compare_values += [csdl_tests.TestingPair(c2, n2, tag = 'v2')]

        # 1d vectors
        c3 = csdl.concatenate((v1, v2))
        n3 = np.concatenate((v1_val, v2_val))
        compare_values += [csdl_tests.TestingPair(c3, n3, tag = 'v3')]

        # mroe 3d tensors
        a_val = np.arange(240).reshape(8,3,10)*0.5 
        b_val = np.arange(80).reshape(8,1,10)*0.5+7
        c_val = np.arange(160).reshape(8,2,10)*0.5+3
        a = csdl.Variable(name = 'a', value = a_val)
        b = csdl.Variable(name = 'b', value = b_val)
        c = csdl.Variable(name = 'c', value = c_val)
        c2 = csdl.concatenate((a, b, c), axis = 1)
        n2 = np.concatenate((a_val, b_val, c_val), axis = 1)
        compare_values += [csdl_tests.TestingPair(c2, n2, tag = 'v4')]

        # more vectors
        a_val = np.arange(14).reshape(2,1,7)*0.5 
        b_val = np.arange(8).reshape(2,1,4)*0.5+7
        c_val = np.arange(6).reshape(2,1,3)*0.5+3
        a = csdl.Variable(name = 'a', value = a_val)
        b = csdl.Variable(name = 'b', value = b_val)
        c = csdl.Variable(name = 'c', value = c_val)
        c2 = csdl.concatenate((a, b, c), axis =(np.ones(1,dtype=np.int32)*2)[0])
        n2 = np.concatenate((a_val, b_val, c_val), axis = 2)
        compare_values += [csdl_tests.TestingPair(c2, n2, tag = 'v5')]

        self.run_tests(
            compare_values = compare_values,
            verify_derivatives=True,
            turn_off_recorder=False
        )

        recorder = csdl.get_current_recorder()
        recorder.start()
        with pytest.raises(TypeError):
            csdl.concatenate((x, y), axis = 1.5)
        with pytest.raises(ValueError):
            csdl.concatenate((x, y), axis = 2)
        with pytest.raises(ValueError):
            csdl.concatenate((x, y), axis = -3)
        with pytest.raises(ValueError):
            csdl.concatenate((x, y.T()))
        with pytest.raises(ValueError):
            csdl.concatenate((x[0], y))
        with pytest.raises(ValueError):
            csdl.concatenate((x,x,x,x,y.T()))

    def test_example(self,):
        self.docstest(concatenate)

if __name__ == '__main__':
    test = TestConcatenate()
    test.overwrite_backend = 'inline'
    test.test_functionality()
    test.test_example()
