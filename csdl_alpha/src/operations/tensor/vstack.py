from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests

import numpy as np
import pytest

class VStack(Operation):
    '''
    Assemble a block matrix from a list of matrices or a list of lists.
    '''

    def __init__(self, *args, shape=None):
        super().__init__(*args)
        self.name = 'stack'
        out_shapes = (shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, *args):
        return np.vstack([x for x in args])

    def compute_jax(self, *args):
        import jax.numpy as jnp
        return jnp.vstack([x for x in args])

    def evaluate_vjp(self, cotangents, *inputs_and_stack):
        inputs = inputs_and_stack[:-1]
        stack = inputs_and_stack[-1]

        stack_out = cotangents[stack]
        i = 0
        for input in inputs:
            if cotangents.check(input):
                if len(input.shape) == 1:
                    cotangents.accumulate(input, stack_out[i])
                    i += 1
                else:
                    cotangents.accumulate(input, stack_out[i:i+input.shape[0]])
                    i += input.shape[0]
            else:
                i += input.shape[0]



def vstack(arrays)->Variable:
    """
    Stack arrays in a sequence vartically (row wise).

    Parameters
    ----------
    arrays : tuple of arrays to stack. Each array must have the same shape in all but the first dimension.
             1D arrays must have the same length.

    Returns
    -------
    Variable
        The stacked array - at least 2D.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x_val = 3.0*np.ones((3,))
    >>> z_val = np.ones((3,))
    >>> x = csdl.Variable(name = 'x', value = x_val)
    >>> z = csdl.Variable(name = 'z', value = z_val)
    >>> y = csdl.vstack((x, z))
    >>> y.value
    array([[3., 3., 3.],
           [1., 1., 1.]])
    """
    if not isinstance(arrays, (tuple, list)):
        raise ValueError('Input must be a list or tuple of arrays to stack')
    if len(arrays) < 2:
        raise ValueError('Input list must have at least two arrays to stack')

    args = [validate_and_variablize(x) for x in arrays]
    shape = args[0].shape
    if len(shape) == 1:
        for arg in args:
            if arg.shape != shape:
                raise ValueError('All 1D arrays must have the same length')
        output_shape = (len(args), shape[0])
    else:
        for arg in args:
            if arg.shape[1:] != shape[1:]:
                raise ValueError('All arrays must have the same shape in all but the first dimension')
        output_shape = (sum([arg.shape[0] for arg in args]),) + shape[1:]
    

    op = VStack(*args, shape=output_shape)
    
    return op.finalize_and_return_outputs()


class TestStack(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

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
        c1 = csdl.vstack((x, y))
        n1 = np.vstack((x_val, y_val))
        compare_values += [csdl_tests.TestingPair(c1, n1, tag = 'v1')]

        # 3d tensors
        c2 = csdl.vstack((a, b, c))
        n2 = np.vstack((a_val, b_val, c_val))
        compare_values += [csdl_tests.TestingPair(c2, n2, tag = 'v2')]

        # 1d vectors
        c3 = csdl.vstack((v1, v2))
        n3 = np.vstack((v1_val, v2_val))
        compare_values += [csdl_tests.TestingPair(c3, n3, tag = 'v3')]

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

        with pytest.raises(ValueError):
            v1 = csdl.vstack([x, y.T()])

        with pytest.raises(ValueError):
            v2 = csdl.vstack([a, b, c.T()])

        with pytest.raises(ValueError):
            v3 = csdl.vstack([v1, v2[1:]])

    def test_example(self,):
        self.docstest(vstack)


# if __name__ == '__main__':
#     test = TestBlockMat()
#     test.test_functionality()
#     test.test_errors()
#     # test.test_example()
