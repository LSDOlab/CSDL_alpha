from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests

import numpy as np

class Sum(Operation):
    '''
    Sum entries in the input tensor along the specified axes.
    '''
    def __init__(self, *args, axes=None, out_shape=None):
        super().__init__(*args)
        self.name = 'sum'
        out_shapes = (out_shape,)
        self.axes = axes
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, *args):
        if self.axes is None:
            return np.sum(args[0])
        else:
            return np.sum(args[0], axis=self.axes)

class MultiSum(ComposedOperation):
    '''
    Elementwise sum of all the Variables in the arguments.
    '''
    def __init__(self, *args, out_shape=None):
        super().__init__(*args)
        self.name = 'multi_sum'

    def evaluate_composed(self, *args):
        return evaluate_multisum(*args)

def evaluate_multisum(*args):
    out = args[0] + args[1]
    for i in range(2, len(args)):
        out = out + args[i]
    return out

def sum(*args, axes=None):
    """
    doc strings
    """
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
        
        op = Sum(*args, axes=axes, out_shape=out_shape)
    else:
        # axes is None for multiple variables
        args = [variablize(x) for x in args]
        out_shape = args[0].shape
        op = MultiSum(*args, out_shape=out_shape)
    
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

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)

        compare_values = []
        # sum of a single tensor variable
        s1 = csdl.sum(x)
        t1 = np.array([18.0])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # sum of a single tensor constant
        s2 = csdl.sum(x_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # sum of a single tensor variable along specified axes
        s3 = csdl.sum(x, axes=(1,))
        t3 = np.array([9,9])
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # sum of a multiple tensor variables
        s4 = csdl.sum(x, y, z)
        t4 = 6.0*np.ones((2,3))
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]

        # sum of a multiple tensor constants
        s5 = csdl.sum(x_val, y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s5, t4, tag = 's5')]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.prep()

        # docs:entry
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

        # sum of a single tensor variable
        s1 = csdl.sum(x)
        print(s1.value)

        # sum of a single tensor constant
        s2 = csdl.sum(x_val)
        print(s2.value)

        # sum of a single tensor variable along specified axes
        s3 = csdl.sum(x, axes=(1,))
        print(s3.value)

        # sum of a multiple tensor variables
        s4 = csdl.sum(x, y, z)
        print(s4.value)

        # sum of a multiple tensor constants and variables
        s5 = csdl.sum(x_val, y_val, z)
        print(s5.value)
        # docs:exit

        compare_values = []
        t1 = np.array([18.0])
        t3 = np.array([9,9])
        t4 = 6.0*np.ones((2,3))

        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]
        compare_values += [csdl_tests.TestingPair(s5, t4, tag = 's5')]
        
        self.run_tests(compare_values = compare_values,)

if __name__ == '__main__':
    test = TestSum()
    test.test_functionality()
    test.test_example()