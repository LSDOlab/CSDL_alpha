from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import set_properties 
from csdl_alpha.src.graph.variable import Variable
import csdl_alpha.utils.testing_utils as csdl_tests
import numpy as np
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.utils.inputs import validate_and_variablize

def np_softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def np_sigmoid(x):
    return 0.5 * (np.tanh(x / 2) + 1)

def np_relu(x):
    return np.maximum(0, x)

@set_properties(linear=False)
class SoftPlus(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'softplus'

    def compute_inline(self, x):
        return np_softplus(x)

    def compute_jax(self, x):
        import jax
        return jax.nn.softplus(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]*sigmoid(x))

@set_properties(linear=False)
class ReLU(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'ReLU'

    def compute_inline(self, x):
        return np_relu(x)

    def compute_jax(self, x):
        import jax
        return jax.nn.relu(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            # cotangents.accumulate(x, cotangents[y]*sigmoid(x))
            cotangents.accumulate(x, cotangents[y]*(relu(x)/x))

def softplus(x:VariableLike) -> Variable:
    """Elementwise softplus of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take softplus of

    Returns
    -------
    y: Variable
        The elementwise softplus of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([-2.0, 0.0, 2.0]))
    >>> y = csdl.softplus(x)
    >>> y.value
    array([0.12692801, 0.69314718, 2.12692801])
    """
    x = validate_and_variablize(x)
    return SoftPlus(x).finalize_and_return_outputs()

def relu(x:VariableLike) -> Variable:
    """Elementwise ReLU of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take ReLU of

    Returns
    -------
    y: Variable
        The elementwise ReLU of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([-2.0, 0.0, 2.0]))
    >>> y = csdl.relu(x)
    >>> y.value
    array([0., 0., 2.])
    """
    x = validate_and_variablize(x)
    return ReLU(x).finalize_and_return_outputs()

def softplus(x:VariableLike) -> Variable:
    """Elementwise softplus of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take softplus of

    Returns
    -------
    y: Variable
        The elementwise softplus of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([-2.0, 0.0, 2.0]))
    >>> y = csdl.softplus(x)
    >>> y.value
    array([0.12692801, 0.69314718, 2.12692801])
    """
    x = validate_and_variablize(x)
    return SoftPlus(x).finalize_and_return_outputs()

def sigmoid(x:VariableLike) -> Variable:
    """Elementwise sigmoid of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take sigmoid of

    Returns
    -------
    y: Variable
        The elementwise sigmoid of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([-2.0, 0.0, 2.0]))
    >>> y = csdl.sigmoid(x)
    >>> y.value
    array([0.11920292, 0.5       , 0.88079708])
    """
    import csdl_alpha
    return 0.5 * (csdl_alpha.tanh(x / 2) + 1)

class TestActivations(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep(always_build_inline=True)

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = np.array([-1000000.0, -1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0, 1000000.0])
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []
        # softplus/sigmoid scalar variables
        s1 = csdl.softplus(x)
        t1 = np_softplus(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s1, t1)]
        s2 = csdl.sigmoid(x)
        t2 = np_sigmoid(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s2, t2)]
        s3 = csdl.relu(x)
        t3 = np_relu(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s3, t3)]

        # softplus/sigmoid tensor variables
        s4 = csdl.softplus(y)
        t4 = np_softplus(y_val)
        compare_values += [csdl_tests.TestingPair(s4, t4)]
        s5 = csdl.sigmoid(y)
        t5 = np_sigmoid(y_val)
        compare_values += [csdl_tests.TestingPair(s5, t5)]
        s6 = csdl.relu(y)
        t6 = np_relu(y_val)
        compare_values += [csdl_tests.TestingPair(s6, t6)]

        assert np.all(np.isfinite(s1.value)), "Array contains NaN or Inf values!"
        assert np.all(np.isfinite(s2.value)), "Array contains NaN or Inf values!"
        assert np.all(np.isfinite(s3.value)), "Array contains NaN or Inf values!"
        assert np.all(np.isfinite(s4.value)), "Array contains NaN or Inf values!"
        assert np.all(np.isfinite(s5.value)), "Array contains NaN or Inf values!"
        assert np.all(np.isfinite(s6.value)), "Array contains NaN or Inf values!"

        sp_deriv = csdl.derivative(s1, y)
        compare_values += [csdl_tests.TestingPair(sp_deriv, sp_deriv.value)]
        assert np.all(np.isfinite(sp_deriv.value)), "Array contains NaN or Inf values!"

        sig_deriv = csdl.derivative(s2, y)
        compare_values += [csdl_tests.TestingPair(sig_deriv, sig_deriv.value)]
        assert np.all(np.isfinite(sig_deriv.value)), "Array contains NaN or Inf values!"

        sp_deriv2 = csdl.derivative(sp_deriv, y)
        compare_values += [csdl_tests.TestingPair(sp_deriv2, sp_deriv2.value)]
        assert np.all(np.isfinite(sp_deriv2.value)), "Array contains NaN or Inf values!"

        sig_deriv2 = csdl.derivative(sig_deriv, y)
        compare_values += [csdl_tests.TestingPair(sig_deriv2, sig_deriv2.value)]
        assert np.all(np.isfinite(sig_deriv2.value)), "Array contains NaN or Inf values!"

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_functionality_relu(self,):
        self.prep(always_build_inline=True)

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = np.array([-1000000.0, -1000.0, -100.0, -10.0, -1.0, -0.00001, 0.00001, 1.0, 10.0, 100.0, 1000.0, 1000000.0])
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []
        # softplus/sigmoid scalar variables
        s3 = csdl.relu(x)
        t3 = np_relu(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s3, t3)]

        # softplus/sigmoid tensor variables
        s6 = csdl.relu(y)
        t6 = np_relu(y_val)
        compare_values += [csdl_tests.TestingPair(s6, t6)]

        assert np.all(np.isfinite(s3.value)), "Array contains NaN or Inf values!"
        assert np.all(np.isfinite(s6.value)), "Array contains NaN or Inf values!"

        sp_deriv = csdl.derivative(s6, y)
        compare_values += [csdl_tests.TestingPair(sp_deriv, sp_deriv.value)]
        assert np.all(np.isfinite(sp_deriv.value)), "Array contains NaN or Inf values!"

        sig_deriv = csdl.derivative(s3, y)
        compare_values += [csdl_tests.TestingPair(sig_deriv, sig_deriv.value)]
        assert np.all(np.isfinite(sig_deriv.value)), "Array contains NaN or Inf values!"

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_examples(self):
        self.docstest(softplus)
        self.docstest(sigmoid)
        self.docstest(relu)

