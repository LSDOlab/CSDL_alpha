from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha as csdl

import numpy as np

class Exp(ComposedOperation):
    '''
    Exponential of the input tensor or scalar.
    '''
    def __init__(self, x):
        super().__init__(x)
        self.name  = 'exp'

    def evaluate_composed(self, x):
        return evaluate_exp(x)
    
def evaluate_exp(x):
    out = csdl.power(np.e, x)
    return out

def exp(x):
    """
    doc strings
    """
    op = Exp(x)
    return op.finalize_and_return_outputs()

class TestExp(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        x_val = np.arange(6).reshape(2,3)
        x = csdl.Variable(name = 'x', value = x_val)
        
        compare_values = []
        # exponential of a tensor variable
        s1 = csdl.exp(x)
        t1 = np.exp(x_val)
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # exponential of a scalar constant
        s2 = csdl.exp(3.0)
        t2 = np.array([np.exp(3.0)])
        compare_values += [csdl_tests.TestingPair(s2, t2, tag = 's2')]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.prep()

        # docs:entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        x_val = np.arange(6).reshape(2,3)
        x = csdl.Variable(name = 'x', value = x_val)
        
        # exponential of a tensor variable
        s1 = csdl.exp(x)
        print(s1.value)

        # exponential of a scalar constant
        s2 = csdl.exp(3.0)
        print(s2.value)

        # docs:exit
        compare_values = []
        t1 = np.exp(x_val)
        t2 = np.array([np.exp(3.0)])
        
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]
        compare_values += [csdl_tests.TestingPair(s2, t2, tag = 's2')]

        self.run_tests(compare_values = compare_values,)

if __name__ == '__main__':
    test = TestExp()
    test.test_functionality()
    test.test_example()