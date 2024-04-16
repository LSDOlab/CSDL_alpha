from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import set_properties 
from csdl_alpha.src.graph.variable import Variable
import csdl_alpha.utils.test_utils as csdl_tests
import numpy as np
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.utils.inputs import validate_and_variablize

@set_properties(linear=False)
class Sin(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'sin'

    def compute_inline(self, x):
        return np.sin(x)

@set_properties(linear=False)
class ArcSin(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'asin'

    def compute_inline(self, x):
        return np.arcsin(x)


@set_properties(linear=False)
class Cos(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'cos'

    def compute_inline(self, x):
        return np.cos(x)

@set_properties(linear=False)
class ArcCos(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'acos'

    def compute_inline(self, x):
        return np.arccos(x)

@set_properties(linear=False)
class Tan(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'tan'

    def compute_inline(self, x):
        return np.tan(x)

@set_properties(linear=False)
class ArcTan(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'atan'

    def compute_inline(self, x):
        return np.arctan(x)

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
    x = validate_and_variablize(x, raise_on_sparse = False)
    return Sin(x).finalize_and_return_outputs()


def arcsin(x:VariableLike) -> Variable:
    """Elementwise arcsine of a CSDL Variable

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
    >>> x = csdl.Variable(value = np.array([1.0, -0.5, 0.5]))
    >>> y = csdl.arcsin(x)
    >>> y.value
    array([ 1.57079633, -0.52359878,  0.52359878])
    """
    x = validate_and_variablize(x, raise_on_sparse = False)
    return ArcSin(x).finalize_and_return_outputs()

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
    x = validate_and_variablize(x)
    return Cos(x).finalize_and_return_outputs()

def arccos(x:VariableLike) -> Variable:
    """Elementwise arccosine of a CSDL Variable

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
    >>> x = csdl.Variable(value = np.array([1.0, -0.5, 0.5]))
    >>> y = csdl.arccos(x)
    >>> y.value
    array([0.        , 2.0943951 , 1.04719755])
    """
    x = validate_and_variablize(x)
    return ArcCos(x).finalize_and_return_outputs()

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
    x = validate_and_variablize(x)
    return Tan(x).finalize_and_return_outputs()

def arctan(x:VariableLike) -> Variable:
    """Elementwise arctangent of a CSDL Variable

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
    >>> y = csdl.arctan(x)
    >>> y.value
    array([0.78539816, 1.10714872, 1.24904577])
    """
    x = validate_and_variablize(x)
    return ArcTan(x).finalize_and_return_outputs()

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

        compare_values = []
        # sin/cos/tan scalar variables
        s1a = csdl.arcsin(x)
        t1a = np.arcsin(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s1a, t1a)]

        s2a = csdl.arccos(x)
        t2a = np.arccos(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s2a, t2a)]

        s3a = csdl.arctan(x)
        t3a = np.arctan(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s3a, t3a)]

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

        # sin/cos/tan tensor variables
        s4a = csdl.arcsin(y)
        t4a = np.arcsin(y_val)
        compare_values += [csdl_tests.TestingPair(s4a, t4a)]

        s5a = csdl.arccos(y)
        t5a = np.arccos(y_val)
        compare_values += [csdl_tests.TestingPair(s5a, t5a)]

        s6a = csdl.arctan(y)
        t6a = np.arctan(y_val)
        compare_values += [csdl_tests.TestingPair(s6a, t6a)]

        self.run_tests(compare_values = compare_values,)

    def test_examples(self):
        self.docstest(sin)
        self.docstest(cos)
        self.docstest(tan)
        self.docstest(arcsin)
        self.docstest(arccos)
        self.docstest(arctan)
