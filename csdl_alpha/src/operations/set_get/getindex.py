from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize

import csdl_alpha.utils.test_utils as csdl_tests

@set_properties(linear=True)
class SetIndex(Operation):
    '''
    '''

    def __init__(self, x, y, slice):
        '''
        Slice can be a tuple of slices or a single slice or list of index sets.
        '''
        super().__init__(x, y)
        self.name = 'set'
        out_shapes = (x.shape,) 
        self.set_dense_outputs(out_shapes)
        self.slice = slice

    def compute_inline(self, x, y):
        out = x.copy()
        out[self.slice] = y
        return out

def get_index(x:Variable, slices):
    """
    doc strings
    """
    x = variablize(x)

    





    

# class TestGet(csdl_tests.CSDLTest):
    
#     def test_functionality(self,):
#         self.prep()
#         import csdl_alpha as csdl
#         import numpy as np

#         shape_1 = (10,9,8)
#         x_val = np.arange(np.prod(shape_1)).reshape(shape_1)
#         y_val = np.ones((1,)).reshape((1,1,1))
#         x = csdl.Variable(name = 'x', value = x_val)
#         y = csdl.Variable(name = 'y', value = y_val)

#         compare_values = []
#         # set a scalar slice with a scalar variable
#         x1 = x[0:1]
#         compare_values += [csdl_tests.TestingPair(x1, x_val[0:1])]

#         self.run_tests(compare_values = compare_values,)

if __name__ == '__main__':
    test = TestGet()
    test.test_functionality()