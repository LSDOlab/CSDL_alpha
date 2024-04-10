from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
import numpy as np
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests

class Power(ElementwiseOperation):
    '''
    Elementwise power of a tensor.
    Power of the first input to the second input. 
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'power'

    def compute_inline(self, x, y):
        return x ** y
    
class BroadcastPower(Operation):
    '''
    First input is broadcasted to the shape of the second input.
    '''

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'broadcast_power'
        out_shapes = (y.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, x, y):
        return x ** y
    
def power(x, y):
    """
    doc strings
    """
    x = variablize(x)
    y = variablize(y)

    if x.shape == y.shape:
        op = Power(x, y)
    # TODO: We need a broadcast power even when the methods are exactly the same 
    # because Broadcast should never inherit from ElementwiseOperation
    elif y.shape == (1,):
        op = Power(x, y)
    elif x.shape == (1,):
        op = BroadcastPower(x, y)
    else:
        raise ValueError('Shapes not compatible for the power operation.')
        
    return op.finalize_and_return_outputs()

class TestPower(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(6).reshape(2,3)
        y_val = 2.0
        z_val = 2.0*np.ones((2,3))
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        
        compare_values = []
        # power of a tensor variable to a scalar variable
        s1 = csdl.power(x, y)
        t1 = x_val ** y_val
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # power of a tensor variable to a tensor constant
        s2 = csdl.power(x, z_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # power of a scalar constant to a tensor variable
        s3 = csdl.power(3.0, x)
        t3 = 3.0 ** x_val
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        self.run_tests(compare_values = compare_values,)

    def test_example(self,):
        self.prep()

        # docs:entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        x_val = np.arange(6).reshape(2,3)
        y_val = 2.0
        z_val = 2.0*np.ones((2,3))
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        
        compare_values = []
        # power of a tensor variable to a scalar variable
        s1 = csdl.power(x, y)
        print(s1.value)

        # power of a tensor variable to a tensor constant
        s2 = csdl.power(x, z_val)
        print(s2.value)

        # power of a scalar constant to a tensor variable
        s3 = csdl.power(3.0, x)
        print(s3.value)
        # docs:exit

        compare_values = []
        t1 = x_val ** y_val
        t3 = 3.0 ** x_val

        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        self.run_tests(compare_values = compare_values,)


if __name__ == '__main__':
    test = TestPower()
    test.test_functionality()
    test.test_example()