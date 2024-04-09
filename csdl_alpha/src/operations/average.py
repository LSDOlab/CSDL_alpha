from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha as csdl

import numpy as np

class Average(Operation):
    '''
    Average entries in the input tensor along the specified axes.
    '''
    def __init__(self, *args, axes=None, out_shape=None):
        super().__init__(*args)
        self.name = 'average'
        out_shapes = (out_shape,)
        self.axes = axes
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, *args):
        if self.axes is None:
            return np.average(args[0])
        else:
            return np.average(args[0], axis=self.axes)

class MultiAverage(ComposedOperation):
    '''
    Elementwise average of all the Variables in the arguments.
    '''
    def __init__(self, *args, out_shape=None):
        super().__init__(*args)
        self.name = 'multi_average'

    def evaluate_composed(self, *args):
        return evaluate_multiaverage(*args)

def evaluate_multiaverage(*args):
    out = csdl.sum(*args)/len(args)
    return out

def average(*args, axes=None):
    """
    doc strings
    """
    # Multiple Variables to average
    if axes is not None and len(args) > 1:
        raise ValueError('Cannot average multiple Variables along specified axes. \
                         Use X = average(A,B,...) followed by out=average(X, axes=(...)) instead.')
    if any(args[i].shape != args[0].shape for i in range(1, len(args))):
        raise ValueError('All Variables must have the same shape.')
    
    # Single Variable to average
    if axes is not None:
        if any(np.asarray(axes) > (len(args[0].shape)-1)):
            raise ValueError('Specified axes cannot be more than the rank of the Variable averaged.')
        if any(np.asarray(axes) < 0):
            raise ValueError('Axes cannot have negative entries.')

    if len(args) == 1:
        if axes is None:
            out_shape = (1,)
        else:
            out_shape = tuple([x for i, x in enumerate(args[0].shape) if i not in axes])
        
        op = Average(*args, axes=axes, out_shape=out_shape)
    else:
        # axes is None for multiple variables
        args = [variablize(x) for x in args]
        out_shape = args[0].shape
        op = MultiAverage(*args, out_shape=out_shape)
    
    return op.finalize_and_return_outputs()

class TestAverage(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)

        compare_values = []
        # average of a single tensor variable
        s1 = csdl.average(x)
        t1 = np.array([7.5])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # average of a single tensor constant
        s2 = csdl.average(x_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # average of a single tensor variable along specified axes
        s3 = csdl.average(x, axes=(1,))
        t3 = np.average(x_val, axis=1)
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # average of a multiple tensor variables
        s4 = csdl.average(x, y, z)
        t4 = (x_val + y_val + z_val)/3
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]

        # average of a multiple tensor constants
        s5 = csdl.average(x_val, y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s5, t4, tag = 's5')]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.prep()

        # docs:entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)

        # average of a single tensor variable
        s1 = csdl.average(x)
        print(s1.value)

        # average of a single tensor constant
        s2 = csdl.average(x_val)
        print(s2.value)

        # average of a single tensor variable along specified axes
        s3 = csdl.average(x, axes=(1,))
        print(s3.value)

        # average of a multiple tensor variables
        s4 = csdl.average(x, y, z)
        print(s4.value)

        # average of a multiple tensor constants and variables
        s5 = csdl.average(x_val, y_val, z)
        print(s5.value)
        # docs:exit

        compare_values = []
        t1 = np.array([7.5])
        t3 = np.average(x_val, axis=1)
        t4 = (x_val + y_val + z_val)/3

        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4')]
        compare_values += [csdl_tests.TestingPair(s5, t4, tag = 's5')]
        
        self.run_tests(compare_values = compare_values,)

if __name__ == '__main__':
    test = TestAverage()
    test.test_functionality()
    test.test_example()