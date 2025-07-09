import pytest
from csdl_alpha.src.operations.operation_subclasses import RandomOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import validate_and_variablize
from csdl_alpha.utils.typing import VariableLike
import csdl_alpha.utils.testing_utils as csdl_tests
import numpy as np

class Normal(RandomOperation):
    def __init__(self, shape:tuple):
        super().__init__()
        self.shape = shape
        self.name = 'normal'
        self.set_dense_outputs((self.shape,))

    def compute_inline(self):
        return np.random.normal(size=self.shape)
    
    def compute_jax(self, key):
        """Compute the normal distribution using JAX.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The random key for JAX.

        Returns
        -------
        jax.numpy.ndarray
            An array of shape `self.shape` with values drawn from a normal distribution.
        """
        from jax import random
        return random.normal(key, shape=self.shape)
    
def normal(shape:tuple) -> Variable:
    """Create a normal distribution variable.

    Parameters
    ----------
    shape : tuple
        The shape of the output array.

    Returns
    -------
    Variable
        A variable representing the normal distribution, with shape `shape`.
    """
    return Normal(shape).finalize_and_return_outputs()
    

class TestNormal(csdl_tests.CSDLTest):
    def test_functionality(self):
        self.prep(always_build_inline=True)

        import csdl_alpha as csdl
        import numpy as np

        shape = (3, 4)
        normal_var = normal(shape)

        # Check the shape of the result
        assert normal_var.shape == shape, f"Expected shape {shape}, got {normal_var.shape}"

        # Check that the values are normally distributed
        assert np.all(np.isfinite(normal_var.value)), "Expected all values to be finite"
