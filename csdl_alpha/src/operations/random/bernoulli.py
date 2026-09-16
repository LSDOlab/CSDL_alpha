import pytest
from csdl_alpha.src.operations.operation_subclasses import RandomOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import validate_and_variablize
from csdl_alpha.utils.typing import VariableLike
import csdl_alpha.utils.testing_utils as csdl_tests
import numpy as np


class Bernoulli(RandomOperation):
    def __init__(self, p:Variable, shape:tuple):
        """Initialize the Bernoulli distribution.

        Parameters
        ----------
        p : Variable
            The probability of success (i.e., the probability of getting 1).
        shape : tuple
            The shape of the output array.
        """
        super().__init__(p)
        self.shape = shape
        self.name = 'bernoulli'
        self.set_dense_outputs((self.shape,))

    def compute_inline(self, p):
        return np.random.binomial(1, p, self.shape)

    def compute_jax(self, key, p):
        """Compute the Bernoulli distribution using JAX.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The random key for JAX.

        Returns
        -------
        jax.numpy.ndarray
            An array of shape `self.shape` with values drawn from a Bernoulli distribution.
        """
        from jax import random
        
        return random.bernoulli(key, p, shape=self.shape)
    

def bernoulli(p:VariableLike, shape:tuple) -> Variable:
    """Create a Bernoulli distribution variable.

    Parameters
    ----------
    p : VariableLike
        The probability of success (i.e., the probability of getting 1).
    shape : tuple
        The shape of the output array.

    Returns
    -------
    Variable
        A variable representing the Bernoulli distribution, with shape `shape`.
    """
    p = validate_and_variablize(p)
    return Bernoulli(p, shape).finalize_and_return_outputs()
    


class TestBernoulli(csdl_tests.CSDLTest):
    def test_functionality(self):
        self.prep(always_build_inline=True)

        import csdl_alpha as csdl
        import numpy as np

        p_val = 0.7
        shape = (3, 4)

        p = csdl.Variable(value=p_val)
        bernoulli_var = bernoulli(p, shape)

        # Check the shape of the result
        assert bernoulli_var.shape == shape, f"Expected shape {shape}, got {bernoulli_var.shape}"

        # Check that the values are either 0 or 1
        assert np.all(np.isin(bernoulli_var.value, [0, 1])), f"Expected values to be 0 or 1, got {bernoulli_var.value}"

    def test_jax_interface(self):
        self.prep()

        import csdl_alpha as csdl
        import jax.numpy as jnp
        from jax import random

        p_val = 0.7
        shape = (3, 4)

        p = csdl.Variable(value=p_val)
        bernoulli_var = bernoulli(p, shape)

        interface = csdl.jax.create_jax_interface(p, bernoulli_var)
        outputs = interface({p:p.value}, prng_key=random.PRNGKey(42))

    def test_derivative_error(self):
        self.prep()

        import csdl_alpha as csdl

        p_val = 0.7
        shape = (3, 4)

        p = csdl.Variable(value=p_val)
        bernoulli_var = bernoulli(p, shape)

        # Check that the derivative raises an error
        with pytest.raises(ValueError):
            csdl.derivative(bernoulli_var, p)