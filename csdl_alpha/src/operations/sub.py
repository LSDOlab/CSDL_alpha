from csdl_alpha.src.operations.operation_subclasses import ComposedOperation, check_expand_subgraphs
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.typing import VariableLike

class Sub(ComposedOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'sub'

    def evaluate_composed(self,x,y):
        return evaluate_sub(x,y)


def evaluate_sub(x:Variable,y:Variable)->Variable:
    return x+(-y)

def sub(x:VariableLike,y:VariableLike)->Variable:
    """Elementwise subtraction of two tensors x and y.

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
    >>> x = csdl.Variable(value = np.array([4.0, 5.0, 6.0]))
    >>> y = csdl.Variable(value = np.array([3.0, 2.0, 1.0]))
    >>> csdl.sub(x, y).value
    array([1., 3., 5.])
    >>> (x - y).value # equivalent to the above
    array([1., 3., 5.])
    >>> (x - 2.0).value # broadcasting is also supported
    array([2., 3., 4.])
    """
    
    if check_expand_subgraphs():
        return evaluate_sub(x,y)
    else:
        op = Sub(x,y)
        return op.finalize_and_return_outputs()

import csdl_alpha.utils.test_utils as csdl_tests
class TestSub(csdl_tests.CSDLTest):
    
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
        s1 = csdl.sub(x,y)
        t1 = np.array([x_val - y_val])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        recorder = csdl.get_current_recorder()
        current_graph = recorder.active_graph
        assert len(current_graph.node_table) == 4

        # subtract scalar constants
        s2 = csdl.sub(3.0, 2.0)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # subtract scalar constant and scalar variable
        s3 = csdl.sub(3.0, y)
        compare_values += [csdl_tests.TestingPair(s3, t1, tag = 's3')]

        # subtract tensor constants
        s4 = csdl.sub(3.0*np.ones((3,2)), 2.0*np.ones((3,2)))
        t2 = 1.0 * np.ones((3,2))
        compare_values += [csdl_tests.TestingPair(s4, t2, tag = 's4')]

        # subtract scalar constant and tensor constant
        s5 = csdl.sub(3.0, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s5, t2, tag = 's5')]

        # subtract scalar variable and tensor constant
        s6 = csdl.sub(x, 2.0*np.ones((3,2)))
        compare_values += [csdl_tests.TestingPair(s6, t2, tag = 's6')]

        z_val = 2.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # add scalar variable and tensor variable
        s7 = csdl.sub(x, z)
        compare_values += [csdl_tests.TestingPair(s7, t2, tag = 's7')]

        # add scalar constant and tensor variable
        s8 = csdl.sub(3.0, z)
        compare_values += [csdl_tests.TestingPair(s8, t2, tag = 's8')]

        # add tensor variables
        s9 = csdl.sub(x, z)
        compare_values += [csdl_tests.TestingPair(s9, t2, tag = 's9')]

        self.run_tests(compare_values = compare_values,)


    def test_functionality_expand(self,):
        self.prep(expand_ops = True)

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = 2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        compare_values = []
        # add scalar variables
        s1 = csdl.sub(x,y)
        t1 = np.array([x_val - y_val])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        recorder = csdl.get_current_recorder()
        current_graph = recorder.active_graph
        assert len(current_graph.node_table) == 6
        # subtract scalar constants
        s2 = csdl.sub(3.0, 2.0)
        # compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]
        assert isinstance(s2, float)

        # subtract scalar constant and scalar variable
        s3 = csdl.sub(3.0, y)
        compare_values += [csdl_tests.TestingPair(s3, t1, tag = 's3')]

        # subtract tensor constants
        s4 = csdl.sub(3.0*np.ones((3,2)), 2.0*np.ones((3,2)))
        t2 = 1.0 * np.ones((3,2))
        assert isinstance(s4, np.ndarray)
        # compare_values += [csdl_tests.TestingPair(s4, t2, tag = 's4')]

        # subtract scalar constant and tensor constant
        s5 = csdl.sub(3.0, 2.0*np.ones((3,2)))
        assert isinstance(s5, np.ndarray)
        # compare_values += [csdl_tests.TestingPair(s5, t2, tag = 's5')]

        # subtract scalar variable and tensor constant
        s6 = csdl.sub(x, 2.0*np.ones((3,2)))
        # assert isinstance(s6, np.ndarray)
        compare_values += [csdl_tests.TestingPair(s6, t2, tag = 's6')]

        z_val = 2.0*np.ones((3,2))
        z = csdl.Variable(name = 'z', value = z_val)
        # add scalar variable and tensor variable
        s7 = csdl.sub(x, z)
        compare_values += [csdl_tests.TestingPair(s7, t2, tag = 's7')]

        # add scalar constant and tensor variable
        s8 = csdl.sub(3.0, z)
        compare_values += [csdl_tests.TestingPair(s8, t2, tag = 's8')]

        # add tensor variables
        s9 = csdl.sub(x, z)
        compare_values += [csdl_tests.TestingPair(s9, t2, tag = 's9')]

        self.run_tests(compare_values = compare_values,)

    def test_docstring(self):
        self.docstest(sub)
if __name__ == '__main__':
    
    test_instance = TestSub()
    test_instance.test_functionality()
    test_instance.test_functionality_expand()