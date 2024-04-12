from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike
from numbers import Number
import numpy as np
from typing import Union
import pytest
import csdl_alpha as csdl

import numpy as np

@set_properties()
class Bessel(ElementwiseOperation):
    '''
    Elementwise logarithm of a tensor.
    '''

    def __init__(
            self,
            x:Variable,
            kind:int = 1,
            order:Union[Number, np.ndarray] = 1,
        )->Variable:
        super().__init__(x)
        self.name = 'bessel'
        self.kind = kind
        self.order = order
        if kind == 1:
            from scipy.special import jv
            self.bessel_func = jv
        elif kind == 2:
            from scipy.special import yv
            self.bessel_func = yv

    def compute_inline(self, x):
        return self.bessel_func(self.order,x)


def bessel(
        x:VariableLike,
        kind:int = 1,
        order:Union[Number, np.ndarray] = 1,
        )->Variable:
    '''Elementwise bessel function of a tensor, uses scipy's bessel functions.
    Supports both J and Y Bessel functions by specifying kind = 1 or kind = 2 respectively.

    Parameters
    ----------
    x : VariableLike
        Input tensor to evaluate bessel function.
    kind : int
        The kind of Bessel function. The options are 1 (J) or 2 (Y)
    order: int
        Order of the Bessel function

    Returns
    -------
    Variable
        Elementwise bessel function of the input tensor.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> csdl.bessel(x).value
    array([0.44005059, 0.57672481, 0.33905896])

    specify kind and order:

    >>> csdl.bessel(x, kind = 2, order = 3).value
    array([-5.82151761, -1.12778378, -0.53854162])
    '''
    x = variablize(x)
    if kind not in [1,2]:
        raise TypeError(f"Bessel function kind must be an integer 1 or 2. {kind} given.")
    if not isinstance(order, (Number, np.ndarray)):
        raise TypeError(f"Bessel function order must be a integer/float or number array. {order} given.")
    if isinstance(order, np.ndarray):
        if x.shape != order.shape:
            raise ValueError(f"Bessel function order must be the same shape as the input. order shape {order.shape} given, {x.shape} expected.")

    return Bessel(x, kind, order).finalize_and_return_outputs()

class TestBessel(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()
        from scipy.special import jv
        from scipy.special import yv

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(6).reshape(2,3)
        x = csdl.Variable(value = x_val)
        compare_values = []

        b1 = csdl.bessel(x)
        compare_values += [csdl_tests.TestingPair(b1, jv(1,x_val))]
        b2 = csdl.bessel(x,kind = 2)
        compare_values += [csdl_tests.TestingPair(b2, yv(1,x_val))]

        b3 = csdl.bessel(x, order = 3)
        compare_values += [csdl_tests.TestingPair(b3, jv(3,x_val))]
        b4 = csdl.bessel(x, order = 3, kind = 2)
        compare_values += [csdl_tests.TestingPair(b4, yv(3,x_val))]

        b5 = csdl.bessel(x, order = np.arange(6).reshape(2,3)+1)
        compare_values += [csdl_tests.TestingPair(b5, jv(np.arange(6).reshape(2,3)+1,x_val))]
        b6 = csdl.bessel(x, order = np.arange(6).reshape(2,3)+1, kind = 2)
        compare_values += [csdl_tests.TestingPair(b6, yv(np.arange(6).reshape(2,3)+1,x_val))]

        b5 = csdl.bessel(x_val, order = np.arange(6).reshape(2,3)+1)
        compare_values += [csdl_tests.TestingPair(b5, jv(np.arange(6).reshape(2,3)+1,x_val))]
        b6 = csdl.bessel(x_val, order = np.arange(6).reshape(2,3)+1, kind = 2)
        compare_values += [csdl_tests.TestingPair(b6, yv(np.arange(6).reshape(2,3)+1,x_val))]

        self.run_tests(compare_values = compare_values,)

    def test_errors(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.arange(6).reshape(2,3)
        x = csdl.Variable(value = x_val)

        with pytest.raises(TypeError):
            csdl.bessel(x, kind = 3)

        with pytest.raises(TypeError):
            csdl.bessel(x, kind = 's')

        with pytest.raises(TypeError):
            csdl.bessel(x, order = 's')

        with pytest.raises(ValueError):
            csdl.bessel(x, order = x_val.reshape(3,2))

    def test_docstrings(self):
        self.docstest(bessel)

if __name__ == '__main__':
    test = TestBessel()
    test.test_functionality()
    test.test_errors()
    test.test_docstrings()