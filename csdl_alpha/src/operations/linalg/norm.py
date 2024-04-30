from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
import csdl_alpha as csdl

import numpy as np

class Norm(ComposedOperation):
    '''
    Vector p-norm (p even) of an input tensor along the specified axes.
    Elementwise p-norm (p even) of all the Variables in the arguments.
    '''
    def __init__(self, *args, axes=None, ord=2):
        super().__init__(*args)
        self.name = 'norm'
        self.axes  = axes
        self.ord   = ord

    def evaluate_composed(self, *args):
        return evaluate_norm(args, self.axes, self.ord)
    
def evaluate_norm(args, axes, ord):
    power_sum = csdl.sum(*[arg**ord for arg in args], axes=axes)
    out = power_sum ** (1/ord)
    return out

def norm(*args, axes=None, ord=2):
    """
    Computes the even p-norm of all entries in the input tensor if a single argument is provided.
    Computes the even p-norm of all entries along the specified axes if `axes` argument is given.
    Computes the elementwise even p-norm of multiple variables of the same shape, 
    if multiple variable arguments are provided. 
    `axes` argument is not allowed in this case.

    Parameters
    ----------
    *args : tuple of Variable or np.ndarray objects
        Input tensor/s whose even p-norm/s needs to be computed.
    axes : tuple of int, default=None
        Axes along which to compute the even p-norm of the input tensor,
        if there's only one input tensor.
    ord : int (even), default=2
        Order of norm to compute. Currently only even p-norms are supported.

    Returns
    -------
    Variable
        p-norm of all entries in the input tensor if a single argument is provided.
        p-norm of entries along the specified axes if `axes` argument is given.
        Elementwise p-norm of multiple variables of the same shape, 
        if multiple variable arguments are provided.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([3.0, 4.0]))
    >>> y1 = csdl.norm(x)
    >>> y1.value
    array([5.])
    >>> y2 = csdl.norm(x, ord=4)
    >>> y2.value
    array([4.28457229])

    Norm of a single tensor variable along a specified axis

    >>> x_val = np.arange(6).reshape(2,3)
    >>> x = csdl.Variable(value = x_val)
    >>> y3 = csdl.norm(x, axes=(1,))
    >>> y3.value
    array([2.23606798, 7.07106781])

    Elementwise norm of multiple tensors

    >>> y4 = csdl.norm(x, 2 * np.ones((2,3)), np.ones((2,3)))
    >>> y4.value
    array([[2.23606798, 2.44948974, 3.        ],
           [3.74165739, 4.58257569, 5.47722558]])
    """
    if not isinstance(ord, int):
        raise ValueError('Order of norm must be an integer.')
    if ord % 2 != 0:
        raise ValueError('Currently only even p-norms are supported.')
    
    # Multiple Variables to find norm
    if axes is not None and len(args) > 1:
        raise ValueError('Cannot find norm of multiple Variables along specified axes. \
                         Use X = norm(A,B,...) followed by out=norm(X, axes=(...)) instead, \
                         if appropriate.')
    if any(args[i].shape != args[0].shape for i in range(1, len(args))):
        raise ValueError('All Variables must have the same shape.')
    
    # Single Variable to find norm
    if axes is not None:
        if any(np.asarray(axes) > (len(args[0].shape)-1)):
            raise ValueError('Specified axes cannot be more than the rank of the Variable.')
        if any(np.asarray(axes) < 0):
            raise ValueError('Axes cannot have negative entries.')

    if len(args) == 1:
        if axes is not None:
            out_shape = tuple([x for i, x in enumerate(args[0].shape) if i not in axes])
            if len(out_shape) == 0:
                raise ValueError('Cannot find norm of a scalar Variable along all axes. \
                                 Use norm(A) to find the norm of a tensor Variable.')
        
    args = [validate_and_variablize(x) for x in args] 
    op = Norm(*args, axes=axes, ord=ord)
    
    return op.finalize_and_return_outputs()

class TestNorm(csdl_tests.CSDLTest):
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
        # 2-norm of a single tensor variable
        s1 = csdl.norm(x)
        t1 = np.array([(np.sum(x_val**2))**0.5])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # 4-norm of a single tensor constant
        s2 = csdl.norm(x_val, ord=4)
        t2 = np.array([(np.sum(x_val**4))**0.25])
        compare_values += [csdl_tests.TestingPair(s2, t2, tag = 's2')]

        # norm of a single tensor variable along specified axes
        s3 = csdl.norm(x, axes=(1,))
        t3 = (np.sum(x_val**2, axis=(1,)))**0.5
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # norm of a single tensor variable along 2 specified axes
        s4 = csdl.norm(d, axes=(0,2))
        t4 = (np.sum(d_val**2, axis=(0,2)))**0.5
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4', decimal=8)]

        # elementwise norm of multiple tensor variables
        s5 = csdl.norm(x, y, z)
        t5 = (x_val**2 + y_val**2 + z_val**2) ** 0.5

        compare_values += [csdl_tests.TestingPair(s5, t5, tag = 's5', decimal=8)]

        # elementwise norm of multiple tensor constants
        s6 = csdl.norm(x_val, y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s6, t5, tag = 's6', decimal=8)]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)


    def test_example(self,):
        self.docstest(norm)

if __name__ == '__main__':
    test = TestNorm()
    test.test_functionality()
    test.test_example()