from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests

import numpy as np

class Product(Operation):
    '''
    Product of entries in the input tensor along the specified axes.
    '''
    def __init__(self, x, axes=None, out_shape=None):
        super().__init__(x)
        self.name = 'product'
        out_shapes = (out_shape,)
        self.axes = axes
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x):
        if self.axes is None:
            return np.prod(x)
        else:
            return np.prod(x, axis=self.axes)

class ElementwiseProduct(ComposedOperation):
    '''
    Elementwise product of all the Variables in the arguments.
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'elementwise_product'

    def evaluate_composed(self, *args):
        return evaluate_elementwise_product(*args)

def evaluate_elementwise_product(*args):
    out = args[0] * args[1]
    for i in range(2, len(args)):
        out = out * args[i]
    return out

def product(*args, axes=None):
    '''
    Computes the product of all entries in the input tensor if a single argument is provided.
    Computes the product of all entries along the specified axes if `axes` argument is given.
    Computes the elementwise product of multiple variables of the same shape, 
    if multiple arguments are provided. Axes argument is not allowed in this case.

    Parameters
    ----------
    *args : tuple of Variable or np.ndarray objects
        Input tensor/s whose product/s needs to be computed.
    axes : tuple of int, default=None
        Axes along which to compute the product of the input tensor,
        if there's only one input tensor.

    Returns
    -------
    Variable
        Product of all entries in the input tensor if a single argument is provided.
        Product of entries along the specified axes if `axes` argument is given.
        Elementwise product of multiple variables of the same shape, 
        if multiple arguments are provided.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y1 = csdl.product(x)
    >>> y1.value
    array([6.])

    Product of a single tensor variable along a specified axis

    >>> x_val = np.array([[1, 2, 3], [4, 5, 6]])
    >>> x = csdl.Variable(value = x_val)
    >>> y2 = csdl.product(x, axes=(1,))
    >>> y2.value
    array([  6., 120.])

    Elementwise product of multiple tensor variables

    >>> y3 = csdl.product(x, 2 * np.ones((2,3)), np.ones((2,3)))
    >>> y3.value
    array([[ 2.,  4.,  6.],
           [ 8., 10., 12.]])
    '''
    # Multiple Variables to product
    if axes is not None and len(args) > 1:
        raise ValueError('Cannot product more than one Variable along specified axes at once. \
                         Use X = product(A,B,...) followed by out=product(X, axes=(...)) instead.')
    if any(args[i].shape != args[0].shape for i in range(1, len(args))):
        raise ValueError('All Variables must have the same shape.')
    
    # Single Variable to product
    if axes is not None:
        if any(np.asarray(axes) > (len(args[0].shape)-1)):
            raise ValueError('Specified axes cannot be more than the rank of the \
                             Variable who product is required.')
        if any(np.asarray(axes) < 0):
            raise ValueError('Axes cannot have negative entries.')

    if len(args) == 1:
        if axes is None:
            out_shape = (1,)
        else:
            out_shape = tuple([x for i, x in enumerate(args[0].shape) if i not in axes])
            if len(out_shape) == 0:
                raise ValueError('It is inefficient to find the product of a tensor Variable along all axes. \
                                 Use product(A) to find the product of all tensor entries.')
        
        op = Product(validate_and_variablize(args[0]), axes=axes, out_shape=out_shape)
    else:
        # axes is None for multiple variables
        args = [validate_and_variablize(x) for x in args]
        op = ElementwiseProduct(*args)
    
    return op.finalize_and_return_outputs()

class TestProduct(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.ones((2,3))
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)

        compare_values = []
        # product of a single tensor variable
        s1 = csdl.product(x)
        t1 = np.array([729.0])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # product of a single tensor constant
        s2 = csdl.product(x_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # product of a single tensor variable along specified axes
        s3 = csdl.product(x, axes=(1,))
        t3 = np.array([27,27])
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # elementwise product of multiple tensor variables
        s4 = csdl.product(x, y, z)
        t4 = 6.0*np.ones((2,3))
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]

        # elementwise product of multiple tensor constants
        s5 = csdl.product(x_val, y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s5, t4, tag = 's5')]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.docstest(product)

if __name__ == '__main__':
    test = TestProduct()
    test.test_functionality()
    test.test_example()