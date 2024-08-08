from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.src.operations.sum import sum as csdl_sum
from csdl_alpha.src.operations.tensor.expand import expand as csdl_expand

class TensorDot(ComposedOperation):
    def __init__(self, x, y, axes=None):
        super().__init__(x,y)
        self.name = 'tensordot'
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        # Example: 'exp' stands for 'expanded'
        # x.shape = (3,2), y.shape = (2,5), axes = ([1],[0])
        # in1_str = 'ab', in2_str = 'bc', in2_unique_str = 'c'
        # exp_str = 'abc'
        # exp_shape = (3,2,5)
        # action1 = 'ab->abc'
        # action2 = 'bc->abc'
        # summation_axes = [1]

        rank1 = len(x.shape)
        rank2 = len(y.shape)
        in1_str = alphabet[:rank1]
        in2_str = alphabet[rank1:rank1+rank2]

        if axes is None:
            exp_str = in1_str + in2_str
            exp_shape = x.shape + y.shape
            self.summation_axes = None

        else:
            in2_unique_str = ''.join([in2_str[i] for i in range(rank2) if i not in axes[1]])
            exp_str   = in1_str + in2_unique_str
            exp_shape = x.shape + tuple([y.shape[i] for i in range(rank2) if i not in axes[1]])
        
            # replace subscripts in in2_str with in1_str at common axes locations
            for i in range(rank2):
                if i in axes[1]:
                    index_in_axes = axes[1].index(i)
                    axis_in_in1   = axes[0][index_in_axes]
                    in2_str       = in2_str[:i] + in1_str[axis_in_in1] + in2_str[i+1:]
            self.summation_axes = tuple(axes[0])

        self.exp_shape = exp_shape
        self.action1 = f'{in1_str}->{exp_str}'
        self.action2 = f'{in2_str}->{exp_str}'

    def evaluate_composed(self, x, y):
        return evaluate_tensordot(x, y, 
                                  self.exp_shape, 
                                  self.action1, 
                                  self.action2, 
                                  self.summation_axes)
    
def evaluate_tensordot(x, y, exp_shape, action1, action2, summation_axes):
    expand1 = csdl_expand(x, exp_shape, action=action1)
    expand2 = csdl_expand(y, exp_shape, action=action2)
    out = expand1 * expand2
    if summation_axes is not None:
        # more efficient summation for inner product
        if len(summation_axes) == len(exp_shape):
            out = csdl_sum(out)
        else:
            out = csdl_sum(out, axes=summation_axes)
    return out

def tensordot(x:VariableLike, y:VariableLike, axes=None)->Variable:
    '''
    Computes the tensor dot product of two tensors x and y
    along the specified axes.
    The axes argument is a tuple of two lists, where the 
    corresponding axes of x and y to multiply and sum over
    are specified.
    If `axes` is specified, the resulting tensor will have shape
    equal to the concatenation of the shapes of x and y,
    with the axes specified removed.
    For example, if x has shape (3,2) and y has shape (2,5),
    and axes = ([1],[0]), the result will have shape (3,5).

    The tensor dot product is a generalization of the 
    inner and outer product operations.
    If no axes is specified, the resulting tensor is the 
    outer product of x and y having shape (x.shape + y.shape).
    If x and y have same shape, and the axes is set to
    ([0,1,...,rank_x], [0,1,...,rank_y]),
    the result is the scalar inner product of x and y.
    Note that the rank_x = rank_y = len(x.shape) = len(y.shape).
    
    Parameters
    ----------
    x : VariableLike
        First input tensor.
    y : VariableLike
        Second input tensor.
    axes : tuple of 2 lists, default=None
        Axes along which to compute the tensor dot product of the input tensors.
        If not specified, the outer product of x and y is computed.
        If specified, the axes must be a tuple of 2 lists.
        The axes must be unique within each list.
        The axes must be non-negative integers within each list.
        Each list in the tuple must have the same length.
        Each corresponding pair of axes for x and y in the 2 lists specified 
        must have equal lengths.

    Returns
    -------
    Variable
        Tensor dot product of x and y.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1, 2, 3]))
    >>> y = csdl.Variable(value = np.array([4, 5]))

    Outer product of x and y:

    >>> csdl.tensordot(x, y).value
    array([[ 4.,  5.],
           [ 8., 10.],
           [12., 15.]])

    Outer product of x and z:

    >>> z = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
    >>> csdl.tensordot(x, z).value
    array([[[ 1.,  2.],
            [ 3.,  4.]],
    <BLANKLINE>
           [[ 2.,  4.],
            [ 6.,  8.]],
    <BLANKLINE>
           [[ 3.,  6.],
            [ 9., 12.]]])

    Dot product of y and z along one axis (same at matrix product z @ y):

    >>> csdl.tensordot(y, z, axes=([0], [1])).value
    array([14., 32.])

    Inner product of z and t:

    >>> t_np = np.array([[5, 6], [7, 8]])
    >>> csdl.tensordot(z, t_np, axes=([0,1], [0,1])).value
    array([70.])
    '''
    x = variablize(x)
    y = variablize(y)

    if axes is not None:
        if isinstance(axes, tuple):
            if len(axes) != 2:
                raise ValueError('`axes` must be a tuple of "two" lists.')
            if not isinstance(axes[0], list) or not isinstance(axes[1], list):
                raise ValueError('`axes` must be a tuple of two "lists".')
        else:
            raise ValueError('`axes` must be a "tuple" of two lists.')
    
        if len(axes[0]) != len(axes[1]):
            raise ValueError('Each list in `axes` must have the same length.')
        if len(axes[0]) != len(set(axes[0])) or len(axes[1]) != len(set(axes[1])):
            raise ValueError('Each list in `axes` must have unique elements.')
        if not all([isinstance(i, int) for i in axes[0]]) or not all([isinstance(i, int) for i in axes[1]]):
            raise ValueError('Each element in the lists of `axes` must be an integer.')
        if not all([i >= 0 for i in axes[0]]) or not all([i >= 0 for i in axes[1]]):
            raise ValueError('Each element in the lists of `axes` must be non-negative.')
        if not all([x.shape[i] == y.shape[j] for i,j in zip(axes[0], axes[1])]):
            raise ValueError('Each corresponding pair of axes in the \
                              2 lists of `axes` specified must have equal lengths.')
    
    op = TensorDot(x, y, axes=axes)

    return op.finalize_and_return_outputs()


class TestTensorDot(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.array([1, 2, 3])
        y_val = np.array([4, 5])
        z_val = np.array([[1, 2], [3, 4]])
        t_val = np.array([[5, 6], [7, 8]])
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        z = csdl.Variable(value = z_val)

        compare_values = []
        # Outer product of x and y:
        compare_values += [csdl_tests.TestingPair(csdl.tensordot(x, y), np.tensordot(x_val, y_val, axes=0))]
        # Outer product of x and z:
        compare_values += [csdl_tests.TestingPair(csdl.tensordot(x_val, z), np.tensordot(x_val, z_val, axes=0))]

        # Dot product of y and z along one axis:
        s1 = csdl.tensordot(y, z, axes=([0], [1]))
        t1 = np.tensordot(y_val, z_val, axes=([0], [1]))
        compare_values += [csdl_tests.TestingPair(s1, t1, tag='s1')]

        # Inner product of z and t:
        s2 = csdl.tensordot(z, t_val, axes=([0,1], [0,1]))
        t2 = np.tensordot(z_val, t_val, axes=([0,1], [0,1])).flatten()
        compare_values += [csdl_tests.TestingPair(s2, t2)]

        # Inner product of z and t:
        s2 = csdl.tensordot(1.0, 2.0)
        compare_values += [csdl_tests.TestingPair(s2, np.ones((1,1))*2.0)]

        self.run_tests(compare_values = compare_values,verify_derivatives=True)

    def test_tensordot_complex(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        def arange_np(*shape, ones = False):
            shape = tuple(shape)
            if not ones:
                val = np.arange(np.prod(shape)).reshape(shape)/np.prod(shape)*20.0+20.0
            else:
                val = np.ones(shape)*2.5
            return csdl.Variable(value=val), val

        grid_n = 4
        grid_n1 = grid_n - 1
        num_physical_dimensions = 1
        quadrature_order = 2

        grid_values, grid_values_np = arange_np(grid_n, grid_n, 3, ones=True)
        grid_values = grid_values*0.25
        # values, values_np = arange_np(grid_n1, grid_n1, num_physical_dimensions)
        quadrature_values, quadrature_values_np = arange_np(grid_n1, grid_n1, quadrature_order**2, 1, ones=False)
        quadrature_values = quadrature_values.T().reshape(quadrature_values.shape)
        quadrature_coord_weights, quadrature_coord_weights_np = arange_np(quadrature_order**2,)

        run_with_loop = False
        if run_with_loop:
            # compute the integral
            values = csdl.Variable(value=np.zeros((grid_n-1, grid_n-1, num_physical_dimensions)))
            for i in csdl.frange(grid_n-1):
                for j in csdl.frange(grid_n-1):
                    for k in csdl.frange(quadrature_order**2):
                        values = values.set(csdl.slice[i,j], values[i,j] + quadrature_values[i,j,k]*quadrature_coord_weights[k])

            # compute areas of the quadrilaterals
            output = csdl.Variable(value=np.zeros((grid_n-1, grid_n-1)))
            for i in csdl.frange(grid_n-1):
                for j in csdl.frange(grid_n-1):
                    area_1 = csdl.norm(csdl.cross(grid_values[i+1,j]-grid_values[i,j], grid_values[i,j+1]-grid_values[i,j]) + 1e-2)/2
                    area_2 = csdl.norm(csdl.cross(grid_values[i,j+1]-grid_values[i+1,j+1], grid_values[i+1,j]-grid_values[i+1,j+1]) + 1e-2)/2
                    output = output.set(csdl.slice[i,j], (area_1+area_2)*values[i,j])
        else:
            values = csdl.tensordot(quadrature_values, quadrature_coord_weights, axes = ([2],[0]))
            area_1 = csdl.norm(csdl.cross(grid_values[1:,:-1]-grid_values[:-1,:-1], grid_values[:-1,1:]-grid_values[:-1,:-1], axis=2) + 1e-2, axes=(2,))/2
            area_2 = csdl.norm(csdl.cross(grid_values[:-1,1:]-grid_values[1:,1:], grid_values[1:,:-1]-grid_values[1:,1:], axis=2) + 1e-2, axes=(2,))/2
            output = (area_1+area_2)*values.reshape(area_1.shape)
                
        run_timing = False
        if not run_timing:
            out_average = csdl.average(output)
            print(out_average.value)
            compare_values = []
            compare_values += [csdl_tests.TestingPair(out_average, np.ones((1,))*56.65516808, decimal=7)]
            self.run_tests(compare_values = compare_values, verify_derivatives=True, step_size=1e-8)
        else:
            ins = [grid_values, quadrature_values, quadrature_coord_weights]
            rec = csdl.get_current_recorder()
            jax_interface = csdl.jax.create_jax_interface(
                inputs = ins,
                outputs = [out_average] + [csdl.derivative(out_average,ins, as_block=True)]
            )
            import time
            start = time.time()
            outputs = jax_interface({input:input.value for input in ins})
            end1 = time.time()
            outputs = jax_interface({input:input.value for input in ins})
            end2 = time.time()
            print(np.linalg.norm(outputs[out_average]))

            print('Time taken for first run:', end1-start)
            print('Time taken for second run:', end2-end1)

    def test_docstring(self):
        self.docstest(tensordot)

if __name__ == '__main__':
    test = TestTensorDot()
    test.overwrite_backend = 'jax'
    test.test_functionality()
    test.test_docstring()
    test.test_tensordot_complex()