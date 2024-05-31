from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.operations.derivative.utils import get_uncontract_action

import numpy as np

class Sum(Operation):
    '''
    Sum entries in the input tensor along the specified axes.
    '''
    def __init__(self, x, axes=None, out_shape=None):
        super().__init__(x)
        self.name = 'sum'
        out_shapes = (out_shape,)
        self.axes = axes
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x):
        if self.axes is None:
            return np.sum(x)
        else:
            return np.sum(x, axis=self.axes)

    def evaluate_vjp(self, cotangents, x, y):
        import csdl_alpha as csdl
        if self.axes is None:
            if cotangents.check(x):
                cotangents.accumulate(x, csdl.expand(cotangents[y], out_shape=x.shape))
        else: 
            if cotangents.check(x):
                action = get_uncontract_action(x.shape, self.axes)
                cotangents.accumulate(x, csdl.expand(cotangents[y], out_shape=x.shape, action = action))

class ElementwiseSum(ComposedOperation):
    '''
    Elementwise sum of all the Variables in the arguments.
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'elementwise_sum'

    def evaluate_composed(self, *args):
        return evaluate_elementwise_sum(*args)

def evaluate_elementwise_sum(*args):
    out = args[0] + args[1]
    for i in range(2, len(args)):
        out = out + args[i]
    return out

def sum(*args, axes=None)->Variable:
    '''
    Computes the sum of all entries in the input tensor if a single argument is provided.
    Computes the sum of all entries along the specified axes if `axes` argument is given.
    Computes the elementwise sum of multiple variables of the same shape, 
    if multiple arguments are provided. Axes argument is not allowed in this case.

    Parameters
    ----------
    *args : tuple of Variable or np.ndarray objects
        Input tensor/s whose sum/s needs to be computed.
    axes : tuple of int, default=None
        Axes along which to compute the sum of the input tensor,
        if there's only one input tensor.

    Returns
    -------
    Variable
        Sum of all entries in the input tensor if a single argument is provided.
        Sum of entries along the specified axes if `axes` argument is given.
        Elementwise sum of multiple variables of the same shape, 
        if multiple arguments are provided.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y1 = csdl.sum(x)
    >>> y1.value
    array([6.])

    Sum of a single tensor variable along a specified axis

    >>> x_val = np.arange(6).reshape(2,3)
    >>> x = csdl.Variable(value = x_val)
    >>> y2 = csdl.sum(x, axes=(1,))
    >>> y2.value
    array([ 3., 12.])

    Elementwise sum of multiple tensor variables

    >>> y3 = csdl.sum(x, 2 * np.ones((2,3)), np.ones((2,3)))
    >>> y3.value
    array([[3., 4., 5.],
           [6., 7., 8.]])
    '''
    # Multiple Variables to sum
    if axes is not None and len(args) > 1:
        raise ValueError('Cannot sum multiple Variables along specified axes. \
                         Use X = sum(A,B,...) followed by out=sum(X, axes=(...)) instead.')
    if any(args[i].shape != args[0].shape for i in range(1, len(args))):
        raise ValueError('All Variables must have the same shape.')
    
    # Single Variable to sum
    if axes is not None:
        if any(np.asarray(axes) > (len(args[0].shape)-1)):
            raise ValueError('Specified axes cannot be more than the rank of the Variable summed.')
        if any(np.asarray(axes) < 0):
            raise ValueError('Axes cannot have negative entries.')

    if len(args) == 1:
        if axes is None:
            out_shape = (1,)
        else:
            out_shape = tuple([x for i, x in enumerate(args[0].shape) if i not in axes])
            if len(out_shape) == 0:
                # raise ValueError('It is inefficient to sum a tensor Variable along all axes. \
                #                  Use sum(A) to find the sum of all tensor entries.')
                out_shape = (1,)
                axes = None

        op = Sum(validate_and_variablize(args[0]), axes=axes, out_shape=out_shape)
    else:
        # axes is None for multiple variables
        args = [validate_and_variablize(x) for x in args]
        op = ElementwiseSum(*args)
    
    return op.finalize_and_return_outputs()

class TestSum(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.ones((2,3))
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))
        w_val = np.arange(60).reshape((3, 4, 5))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        w = csdl.Variable(name = 'w', value = w_val)

        compare_values = []
        # sum of a single tensor variable
        s1 = csdl.sum(x)
        t1 = np.array([18.0])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # sum of a single tensor variable
        s1 = csdl.sum(x, axes=(0,1))
        t1 = np.array([18.0])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # sum of a single tensor constant
        s2 = csdl.sum(x_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        s3w = csdl.sum(w, axes=(0,2))
        t3w = np.sum(w_val, axis=(0,2))
        compare_values += [csdl_tests.TestingPair(s3w, t3w, tag = 's3w')]

        s3w = csdl.sum(w, axes=(1,))
        t3w = np.sum(w_val, axis=(1,))
        compare_values += [csdl_tests.TestingPair(s3w, t3w, tag = 's3w')]

        # sum of a single tensor variable along specified axes
        s3 = csdl.sum(x, axes=(1,))
        t3 = np.array([9,9])
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # elementwise sum of multiple tensor variables
        s4 = csdl.sum(x, y, z)
        t4 = 6.0*np.ones((2,3))
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]

        # elementwise sum of multiple tensor constants
        s5 = csdl.sum(x_val, y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s5, t4, tag = 's5')]

        self.run_tests(compare_values = compare_values,verify_derivatives=True)

    def test_example(self,):
        self.docstest(sum)

if __name__ == '__main__':
    test = TestSum()
    test.test_functionality()
    test.test_example()