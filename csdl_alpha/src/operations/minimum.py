from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha as csdl

import numpy as np

class Minimum(ComposedOperation):
    '''
    Minimum entries in the input tensor along the specified axes.
    Or elementwise minimum if multiple variables of the same shape
    are provided.
    '''
    def __init__(self, *args, axes=None, rho=20.):
        super().__init__(*args)
        self.name  = 'minimum'
        self.axes  = axes
        self.rho   = rho

    def evaluate_composed(self, *args):
        return evaluate_minimum(args, self.axes, self.rho)
    
def evaluate_minimum(args, axes, rho):
    neg_args = [-arg for arg in args]
    out = csdl.maximum(*neg_args, axes=axes, rho=rho)
    return -out

def minimum(*args, axes=None, rho=20.):
    '''
    Computes the minimum entry in the input tensor if a single argument is provided.
    Computes the minimum entries along the specified axes if `axes` argument is given.
    Computes the elementwise minimum of multiple variables of the same shape, 
    if multiple arguments are provided. Axes argument is not allowed in this case.

    Parameters
    ----------
    *args : tuple of Variable or np.ndarray objects
        Input tensor/s whose minimum needs to be computed.
    axes : tuple of int, default=None
        Axes along which to compute the minimum of the input tensor,
        if there's only one input tensor.
    rho : float, default=20.
        Smoothing parameter for the minimum function.

    Returns
    -------
    Variable
        Minimum entry in the input tensor if a single argument is provided.
        Minimum entries along the specified axes if `axes` argument is given.
        Elementwise minimum of multiple variables of the same shape, 
        if multiple arguments are provided.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x_val = np.arange(6).reshape(2,3)
    >>> x = csdl.Variable(value = x_val)
    >>> y1 = csdl.minimum(x)
    >>> y1.value
    array([-1.03057685e-10])

    # Note that the value of y1 is not exactly 0.0 due to the smoothing term.
    # The value of y1 can be made closer to 0.0 by increasing the value of 
    # the smoothing parameter rho as shown below.
    
    >>> y = csdl.minimum(x, rho=200)
    >>> y.value
    array([-0.])

    # minimum of a single tensor variable along a specified axis

    >>> y2 = csdl.minimum(x, axes=(1,))
    >>> y2.value
    array([-1.03057685e-10,  3.00000000e+00])

    # minimum of multiple tensor variables

    >>> y3 = csdl.minimum(x, 2 * np.ones((2,3)), np.ones((2,3)))
    >>> y3.value
    array([[-1.03057685e-10,  9.65342641e-01,  1.00000000e+00],
           [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]])
    '''
    
    # Multiple Variables to find minimum
    if axes is not None and len(args) > 1:
        raise ValueError('Cannot find minimum of multiple variables along specified axes. \
                         Use X = min(A,B,...) followed by out=min(X, axes=(...)) instead.')
    if any(args[i].shape != args[0].shape for i in range(1, len(args))):
        raise ValueError('All Variables must have the same shape.')
    
    # Single Variable to find minimum
    if axes is not None:
        if any(np.asarray(axes) > (len(args[0].shape)-1)):
            raise ValueError('Specified axes cannot be more than the rank of the Variable.')
        if any(np.asarray(axes) < 0):
            raise ValueError('Axes cannot have negative entries.')

    if len(args) == 1:
        if axes is not None:
            out_shape = tuple([x for i, x in enumerate(args[0].shape) if i not in axes])
            if len(out_shape) == 0:
                raise ValueError('Cannot find minimum of a scalar variable along all axes. \
                                 Use minimum(A) to find the minimum of a tensor Variable.')
        
    args = [variablize(x) for x in args]
    op = Minimum(*args, axes=axes, rho=rho)
    
    return op.finalize_and_return_outputs()

class TestMinimum(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))
        d_val = np.arange(12).reshape(2,3,2)


        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        d = csdl.Variable(name = 'd', value = d_val)

        compare_values = []
        # minimum of a single tensor variable
        s1 = csdl.minimum(x)
        t1 = np.array([0.0])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # minimum of a single tensor constant
        s2 = csdl.minimum(x_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # minimum of a single tensor variable along a specified axes
        s3 = csdl.minimum(x, axes=(1,))
        t3 = np.min(x_val, axis=1)
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # minimum of a single tensor variable along 2 specified axes
        s4 = csdl.minimum(d, axes=(0,2))
        t4 = np.min(d_val, axis=(0,2))
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4', decimal=8)]

        # elementwise minimum of multiple tensor variables
        s5 = csdl.minimum(x, y, z)
        t5 = np.minimum(x_val, y_val)
        t5 = np.minimum(t5, z_val)
        compare_values += [csdl_tests.TestingPair(s5, t5, tag = 's5', decimal=8)]

        # elementwise minimum of multiple tensor constants
        s6 = csdl.minimum(x_val, y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s6, t5, tag = 's6', decimal=8)]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.docstest(minimum)

if __name__ == '__main__':
    test = TestMinimum()
    test.test_functionality()
    test.test_example()