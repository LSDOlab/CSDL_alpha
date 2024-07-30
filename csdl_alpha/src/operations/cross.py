from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.operations.derivatives.utils import get_uncontract_action

import numpy as np
import pytest

class Cross(Operation):
    '''
    Computes the cross product of two arrays of 3D vectors.
    '''
    def __init__(self, x, y, axis):
        super().__init__(x, y)
        self.name = 'cross'
        self.axis = axis
        out_shapes = (x.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return np.cross(x, y, axis=self.axis)

    def compute_jax(self, x, y):
        import jax.numpy as jnp
        return jnp.cross(x, y, axis=self.axis)

    def evaluate_vjp(self, cotangents, x, y, z):
        import csdl_alpha as csdl
        if cotangents.check(x):
            cotangents.accumulate(x, -csdl.cross(cotangents[z], y, axis=self.axis))
        if cotangents.check(y):
            cotangents.accumulate(y, -csdl.cross(x, cotangents[z], axis=self.axis))

# class Cross2D(Operation):
#     '''
#     Computes the cross product of two arrays of 2D vectors.
#     '''
#     def __init__(self, x, y, axis):
#         super().__init__(x, y)
#         self.name = '2Dcross'
#         self.axis = axis
#         out_shape = x.shape[:axis] + x.shape[axis+1:]
#         # Uncomment the line below to debug the blanket reshaping in set_inline_values in operation
#         # out_shape = x.shape[:axis] + (1,) + x.shape[axis+1:]
#         # Or uncomment the 2 lines below to debug the blanket reshaping in set_inline_values in operation
#         if x.size == 2:
#             out_shape = (1,)
#         out_shapes = (out_shape,)
#         self.set_dense_outputs(out_shapes)

#     def compute_inline(self, x, y):
#         return np.cross(x, y, axis=self.axis)

#     def evaluate_vjp(self, cotangents, x, y, z):
#         import csdl_alpha as csdl

#         expanded_cotangents = csdl.expand(cotangents[z], out_shape=x.shape, action = get_uncontract_action(x.shape, (self.axis, )))

#         if x.size == 2:
#             if cotangents.check(x):
#                 initial = csdl.Variable(value = np.zeros(x.shape))
#                 initial = initial.set(csdl.slice[], cotangents[z])
#                 initial = initial.set(csdl.slice[], cotangents[z])

#                 cotangents.accumulate(x, )
#             if cotangents.check(y):
#                 cotangents.accumulate(y, -csdl.cross(x, expanded_cotangents, axis=self.axis))

def cross(x, y, axis=None):
    '''
    Computes the cross product of two arrays of 2D or 3D vectors.

    Parameters
    ----------
    x : Variable or np.ndarray
        First input tensor of shape (l,...,2,...,n) or (l,...,3,...,n).
    y : Variable or np.ndarray
        Second input tensor of the same shape as x.
    axis : int, default=None
        Axis along which the 2D or 3D vectors are stored in the input tensors.
        Need not be specified if the input tensors are 1D vectors of size 2 or 3.
        Needs to be specified if the input tensors are 2D or higher dimensional tensors.
        Axis must be a non-negative integer less than the number of dimensions in the input tensors.

    Returns
    -------
    Variable
        Tensor containing the cross product (x × y) of vectors in the input tensors.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([3.0, 4.0, 5.0]))
    >>> y = csdl.Variable(value = np.array([4.0, 5.0, 6.0]))
    >>> csdl.cross(x, y).value
    array([-1.,  2., -1.])

    >>> x = csdl.Variable(value = 3.0 * np.ones((2,3)))
    >>> y_val = np.arange(6).reshape(2,3)
    >>> csdl.cross(x, y_val, axis=1).value
    array([[ 3., -6.,  3.],
           [ 3., -6.,  3.]])
    '''
    x = validate_and_variablize(x)
    y = validate_and_variablize(y)

    if x.size == 1 or y.size == 1:
        raise ValueError("The input tensors must be atleast 1D vectors of size 3.")
    
    if x.shape != y.shape:
        raise ValueError("The input tensors must have the same shape.")

    if len(x.shape) == 1 and axis is None:
        axis = 0

    if not isinstance(axis, int):
        raise ValueError("The axis argument must be an integer.")
    
    if axis < 0:
        raise ValueError("The axis argument must be a non-negative integer.")
    
    if axis >= len(x.shape):
        raise ValueError("The axis argument must be less than the number of dimensions in the input tensors.")

    if x.shape[axis] != 3 and x.shape[axis] != 2:
        raise ValueError("The input tensors must have a size of 3 along the specified axis.")

    if x.shape[axis] == 3:
        op = Cross(x, y, axis)
    elif x.shape[axis] == 2:
        raise NotImplementedError("Cross product of 2D vectors is not yet implemented.")
        op = Cross2D(x, y, axis)
    
    return op.finalize_and_return_outputs()

class TestCross(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        compare_values = []

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        # x_val = np.array([3.0, 4.0])
        # y_val = np.array([4.0, 5.0])
        # x = csdl.Variable(value = x_val)
        # y = csdl.Variable(value = y_val)
        # s1 = csdl.cross(x, y)
        # t1 = np.cross(x_val, y_val).flatten()
        # compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        x_val = 3.0*np.ones((2,3))
        y_val = np.arange(6).reshape(2,3)
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        # s2 = csdl.cross(x, y_val, axis=0)
        # t2 = np.cross(x_val, y_val, axis=0)
        # compare_values += [csdl_tests.TestingPair(s2, t2, tag = 's2')]

        s3 = csdl.cross(x, y_val, axis=1)
        t3 = np.cross(x_val, y_val, axis=1)

        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        x_val = 3.0*np.ones((4, 3, 2))
        y_val = np.arange(24).reshape(4, 3, 2)
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        # s2 = csdl.cross(x, y_val, axis=0)
        # t2 = np.cross(x_val, y_val, axis=0)
        # compare_values += [csdl_tests.TestingPair(s2, t2, tag = 's2')]

        s3 = csdl.cross(x, y_val, axis=1)
        t3 = np.cross(x_val, y_val, axis=1)

        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        with pytest.raises(ValueError):
            s4 = csdl.cross(x, y_val, axis=0)
            t4 = np.cross(x_val, y_val, axis=0)

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_example(self,):
        self.docstest(cross)

if __name__ == '__main__':
    test = TestCross()
    test.test_functionality()
    test.test_example()