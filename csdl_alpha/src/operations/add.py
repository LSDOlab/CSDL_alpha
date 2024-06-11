from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike

@set_properties(linear=True)
class Add(ElementwiseOperation):
    '''
    Elementwise addition of two tensors of the same shape.
    '''

    def __init__(self,x:Variable,y:Variable):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x + y

    def compute_jax(self, x, y):
        return self.compute_inline(x, y)

    def evaluate_vjp(self, cotangents, x, y, z):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[z])
        if cotangents.check(y):
            cotangents.accumulate(y, cotangents[z])

# TODO: Do we need a broadcast add? There's a lot of code duplication b/w both classes
class BroadcastAdd(Operation):
    '''
    Addition after the first input is broadcasted to the shape of the second input.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'broadcast_add'
        out_shapes = (y.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return x + y
    
    def compute_jax(self, x, y):
        z  = x + y
        return z

    def evaluate_vjp(self, cotangents, x, y, z):
        if cotangents.check(x):
            import csdl_alpha as csdl
            cotangents.accumulate(x, csdl.sum(cotangents[z]))
        if cotangents.check(y):
            cotangents.accumulate(y, cotangents[z])

class SparseAdd(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sparse_add'

    def compute_inline(self, x, y):
        pass
    
class SparseBroadcastAdd(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sparse_broadcast_add'

    def compute_inline(self, x, y):
        pass

def add(x:VariableLike,y:VariableLike)->Variable:
    """Elementwise addition of two tensors x and y.

    Parameters
    ----------
    x : Variable
    y : Variable

    Returns
    -------
    out: Variable

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.Variable(value = np.array([4.0, 5.0, 6.0]))
    >>> csdl.add(x, y).value
    array([5., 7., 9.])
    >>> (x + y).value # equivalent to the above
    array([5., 7., 9.])
    >>> (x + 2.0).value # broadcasting is also supported
    array([3., 4., 5.])
    """
    x = validate_and_variablize(x, raise_on_sparse=False)
    y = validate_and_variablize(y, raise_on_sparse=False)

    if x.shape == y.shape:
        op = Add(x,y)
    elif x.size == 1:
        op = BroadcastAdd(x.flatten(),y)
    elif y.size == 1:
        op = BroadcastAdd(y.flatten(),x)
    else:
        raise ValueError(f'Shapes not compatible for add operation. x shape: {x.shape}, y shape: {y.shape}')

    
    # TODO: add later
    # if (not x.is_sparse) and (not y.is_sparse):
    #     if x.shape == y.shape:
    #         op = Add(x,y)
    #     elif x.size == 1:
    #         op = BroadcastAdd(x.flatten(),y)
    #     elif y.size == 1:
    #         op = BroadcastAdd(y.flatten(),x)
    #     else:
    #         raise ValueError(f'Shapes not compatible for sparse add operation. x shape: {x.shape}, y shape: {y.shape}')
    # elif (x.is_sparse) and (y.is_sparse):
    #     if x.shape == y.shape:
    #         op = SparseAdd(x,y)
    #     else:
    #         raise ValueError(f'Shapes not compatible for sparse add operation. x shape: {x.shape}, y shape: {y.shape}')
    # elif (x.is_sparse) and (not y.is_sparse):
    #     if x.shape == y.shape:
    #         op = SparseDenseAdd(x,y)
    #     elif x.size == 1:
    #         op = BroadcastAdd(x.to_dense().flatten(),y)
    #     elif y.size == 1:
    #         op = SparseBroadcastAdd(y.flatten(),x)
    #     else:
    #         raise ValueError(f'Shapes not compatible for sparse add operation. x shape: {x.shape}, y shape: {y.shape}')        
    # elif (not x.is_sparse) and (y.is_sparse):
    #     if x.shape == y.shape:
    #         op = SparseDenseAdd(y,x)
    #     elif x.size == 1:
    #         op = BroadcastAdd(y.to_dense().flatten(),x)
    #     elif y.size == 1:
    #         op = SparseBroadcastAdd(x.flatten(),y)
    #     else:
    #         raise ValueError(f'Shapes not compatible for sparse add operation. x shape: {x.shape}, y shape: {y.shape}')        

    
    
    return op.finalize_and_return_outputs()


class TestAdd(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = 2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        compare_values = []
        # add scalar variables
        s1 = csdl.add(x,y)
        t1 = np.array([x_val + y_val])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # add scalar constants
        s2 = csdl.add(3.0, 2.0)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # add scalar constant and scalar variable
        s3 = csdl.add(3.0, y)
        compare_values += [csdl_tests.TestingPair(s3, t1, tag = 's3')]

        # add tensor constants
        s4 = csdl.add(3.0*np.ones((3,2)), 2.0*np.ones((3,2)))
        t2 = 5.0 * np.ones((3,2))
        compare_values += [csdl_tests.TestingPair(s4, t2, tag = 's4')]

        # add scalar constant and tensor constant
        s5 = csdl.add(3.0, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s5, t2, tag = 's5')]

        # add scalar variable and tensor constant
        s6 = csdl.add(x, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s6, t2, tag = 's6')]

        z_val = 2.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # add scalar variable and tensor variable
        s7 = csdl.add(x, z)
        compare_values += [csdl_tests.TestingPair(s7, t2, tag = 's7')]

        # add scalar constant and tensor variable
        s8 = csdl.add(3.0, z)
        compare_values += [csdl_tests.TestingPair(s8, t2, tag = 's8')]

        # add tensor variables
        s9 = csdl.add(x, z)
        compare_values += [csdl_tests.TestingPair(s9, t2, tag = 's9')]

        # add tensor variables
        s9 = csdl.add(csdl.Variable(value = np.ones((100,))), csdl.Variable(value = np.ones((100,))))
        temp = s9[0]
        temp.add_name('temp')
        compare_values += [csdl_tests.TestingPair(temp, 2*np.ones((100,))[0].flatten(), tag = 's10')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_errors(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.ones((2,2))
        y_val = np.ones((2,2))
        x = csdl.SparseMatrix(name = 'x', value = x_val)
        y = csdl.SparseMatrix(name = 'y', value = y_val)
        
        x+y


    def test_docstring(self):
        self.docstest(add)

if __name__ == '__main__':
    test = TestAdd()
    # test.overwrite_backend = 'jax'
    test.test_functionality()
    test.test_errors()