from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import set_properties 
from csdl_alpha.src.graph.variable import Variable
import csdl_alpha.utils.test_utils as csdl_tests
import numpy as np
from csdl_alpha.utils.typing import VariableLike

@set_properties(linear=False)
class Sin(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'sin'

    def compute_inline(self, x):
        return np.sin(x)


@set_properties(linear=False)
class Cos(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'cos'

    def compute_inline(self, x):
        return np.cos(x)

@set_properties(linear=False)
class Tan(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'tan'

    def compute_inline(self, x):
        return np.tan(x)

def sin(x:VariableLike) -> Variable:
    """Elementwise sine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the sine of

    Returns
    -------
    y: Variable
        The elementwise sine of x
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.sin(x)
    >>> y.value
    array([0.84147098, 0.90929743, 0.14112001])

    """
    return Sin(x).finalize_and_return_outputs()


def cos(x:VariableLike) -> Variable:
    """Elementwise cosine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the cosine of

    Returns
    -------
    y: Variable
        The elementwise cosine of x

        
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.cos(x)
    >>> y.value
    array([ 0.54030231, -0.41614684, -0.9899925 ])
    """
    return Cos(x).finalize_and_return_outputs()


def tan(x:VariableLike) -> Variable:
    """Elementwise tangent of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the tangent of

    Returns
    -------
    y: Variable
        The elementwise tangent of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.tan(x)
    >>> y.value
    array([ 1.55740772, -2.18503986, -0.14254654])
    """
    return Tan(x).finalize_and_return_outputs()

class TestTrig(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = np.arange(10).reshape(2,5)
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []
        # sin/cos/tan scalar variables
        s1 = csdl.sin(x)
        t1 = np.sin(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s1, t1)]

        s2 = csdl.cos(x)
        t2 = np.cos(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s2, t2)]

        s3 = csdl.tan(x)
        t3 = np.tan(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s3, t3)]

        # sin/cos/tan tensor variables
        s4 = csdl.sin(y)
        t4 = np.sin(y_val)
        compare_values += [csdl_tests.TestingPair(s4, t4)]

        s5 = csdl.cos(y)
        t5 = np.cos(y_val)
        compare_values += [csdl_tests.TestingPair(s5, t5)]

        s6 = csdl.tan(y)
        t6 = np.tan(y_val)
        compare_values += [csdl_tests.TestingPair(s6, t6)]


        self.run_tests(compare_values = compare_values,)

    def test_examples(self):
        self.docstest(sin)
        self.docstest(cos)
        self.docstest(tan)

