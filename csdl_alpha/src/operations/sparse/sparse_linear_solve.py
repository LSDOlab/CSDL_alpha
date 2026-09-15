from csdl_alpha.src.graph.operation import Operation, set_properties 
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.linalg.linear_solvers import DirectSolver
from csdl_alpha.src.operations.linalg.linear_solvers.linear_solver import LinearSolver
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.src.operations.linalg.utils import process_matA_vecb
import pytest
import numpy as np
# import jax
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import time

@set_properties()
class SparseLinearSolve(Operation):
    def __init__(self, A:Variable, b:Variable):
        super().__init__(b)
        self.name = 'sp_linear_solve'
        self.A = A
        self.set_dense_outputs((b.shape,))

    def compute_inline(self, b):
        # t1 = time.time()
        result = spsl.spsolve(self.A, b).reshape(b.shape)
        # t2 = time.time()
        # print(f"Time to solve sparse linear system: {t2-t1:.3f} seconds")
        return result
    
    def compute_jax(self, b):
        import jax.numpy as jnp
        from jax.experimental import sparse
        Acsr = self.A.tocsr()
        # CSDL right-hand sides are floating point, even for integer input data.
        data = jnp.asarray(Acsr.data, dtype=b.dtype)
        indices = jnp.asarray(Acsr.indices)
        indptr = jnp.asarray(Acsr.indptr)

        # JAX's sparse solver accepts one right-hand side at a time.
        if b.ndim == 1:
            return sparse.linalg.spsolve(data, indices, indptr, b)
        return jnp.stack([
            sparse.linalg.spsolve(data, indices, indptr, b[:, i])
            for i in range(b.shape[1])
        ], axis=1)



    def evaluate_vjp(self, cotangents, b, x):
        import csdl_alpha as csdl

        # solved_system =  csdl.solve_linear(A.T(), cotangents[x])
        solved_system =  csdl.sparse.solve_linear(self.A.T, cotangents[x])
        if cotangents.check(b):
            cotangents.accumulate(b, solved_system)
        # if cotangents.check(A):
        #     vjp = -csdl.outer(x,solved_system).T().reshape(A.shape)
        #     cotangents.accumulate(A, vjp)

def solve_linear(
        A:VariableLike,
        b:VariableLike,
    )->Variable:
    """Solve a linear system of equations Ax = b for x.

    Parameters
    ----------
    A : scipy sparse matrix or array
        Constant square 2D matrix.
    b : VariableLike
        Right-hand side with shape (n,) or (n, k).

    Returns
    -------
    x: Variable
        Solution with the same shape as b.


    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> A = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
    >>> b = csdl.Variable(value = np.array([5, 6]))
    >>> csdl.solve_linear(A, b).value
    array([-4. ,  4.5])
    >>> recorder.stop()

    Specify different solvers:

    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> A = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
    >>> b = csdl.Variable(value = np.array([5, 6]))
    >>> csdl.solve_linear(A, b, solver = csdl.linear_solvers.DirectSolver()).value
    array([-4. ,  4.5])
    >>> recorder.stop()
    """
    if not sps.issparse(A):
        raise TypeError(f"A must be a scipy sparse matrix or array. Got {type(A)}")
    b = validate_and_variablize(b)

    if len(b.shape) not in (1, 2):
        raise ValueError(f"b must be 1D or 2D, but has shape {b.shape}")
    if len(A.shape) != 2:
        raise ValueError(f"Matrix A must be 2D, but has shape {A.shape}")
    if A.shape[1] != b.shape[0]:
        raise ValueError(f"Number of columns of A must be equal to the number of rows of x. {A.shape[1]} != {b.shape[0]}")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix A must be square, but has shape {A.shape}")

    operation = SparseLinearSolve(A, b)
    output = operation.finalize_and_return_outputs()
    # output = SparseLinearSolve(A_mat, b_vec).finalize_and_return_outputs()

    if len(b.shape) == 2:
        return output
    if len(b.shape) == 1:
        return output.reshape((output.size,))
    
class TestSparseLinear(csdl_tests.CSDLTest):

    def test_inline_without_jax(self):
        import subprocess
        import sys
        from pathlib import Path

        # Use a fresh interpreter so an already-imported JAX cannot mask a regression.
        script = '''
import importlib.abc
import sys
class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise ModuleNotFoundError('JAX intentionally unavailable for this test')
sys.meta_path.insert(0, BlockJax())
import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sps
recorder = csdl.Recorder(inline=True)
recorder.start()
b = csdl.Variable(value=np.array([2., 6.]))
x = csdl.sparse.solve_linear(sps.csr_matrix([[2., 0.], [0., 3.]]), b)
np.testing.assert_allclose(x.value, [1., 2.])
np.testing.assert_allclose(csdl.derivative(x, b).value, np.diag([0.5, 1./3.]))
assert 'jax' not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, '-c', script],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        n = 4
        # condition number too high?
        # A_shape = (n,n)
        # b_shape = (n,1)
        # A_val = (np.arange(np.prod(A_shape)).reshape(A_shape)+1.0)**2.0
        # b_val = np.arange(np.prod(b_shape)).reshape(b_shape)

        main_diag = np.arange(n)+1
        A_val = np.diag(main_diag) + np.diag(main_diag[:-1]+1, 1) + np.diag(main_diag[:-1]+2, -1)
        A = sps.csr_array(A_val)
        b_val = 2*np.arange(n)


        # A = csdl.Variable(value = A_val)
        b = csdl.Variable(value = b_val)

        compare_values = []
        x = csdl.sparse.solve_linear(A,b)
        compare_values += [csdl_tests.TestingPair(x, spsl.spsolve(A, b_val))]
        # x = csdl.sparse.solve_linear(A_val,b)
        # compare_values += [csdl_tests.TestingPair(x, spsl.spsolve(A, b_val))]
        x = csdl.sparse.solve_linear(A,b_val)
        compare_values += [csdl_tests.TestingPair(x, spsl.spsolve(A, b_val))]

        b_shape = (4,)
        b_val = np.arange(np.prod(b_shape)).reshape(b_shape)
        b = csdl.Variable(value = b_val) 
        x = csdl.sparse.solve_linear(A,b)
        compare_values += [csdl_tests.TestingPair(x, spsl.spsolve(A, b_val).flatten())]
        x = csdl.sparse.solve_linear(A,b_val)
        compare_values += [csdl_tests.TestingPair(x, spsl.spsolve(A, b_val).flatten())]
        # x = csdl.sparse.solve_linear(A_val,b)
        # compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val).flatten())]

        # x = csdl.solve_linear(A,b, solver = csdl.linear_solvers.ScipyKrylovSolver())
        # compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]
        # x = csdl.solve_linear(A_val,b, solver = csdl.linear_solvers.ScipyKrylovSolver())
        # compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]
        # x = csdl.solve_linear(A,b_val, solver = csdl.linear_solvers.DirectSolver())
        # compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)
    
    @pytest.mark.parametrize('sparse_type', [
        sps.csr_matrix, sps.csc_matrix, sps.csr_array, sps.csc_array,
    ])
    @pytest.mark.parametrize('rhs_shape', [(3,), (3, 1), (3, 2), (3, 4)])
    def test_rhs_shapes(self, sparse_type, rhs_shape):
        self.prep()
        import csdl_alpha as csdl

        # Nonsymmetric A also exercises the transpose solve in the VJP.
        A_val = np.array([[4., 1., 0.], [0., 3., 1.], [1., 0., 2.]])
        b_val = np.arange(np.prod(rhs_shape), dtype=float).reshape(rhs_shape) + 1
        b = csdl.Variable(value=b_val)
        x = csdl.sparse.solve_linear(sparse_type(A_val), b)
        expected = np.linalg.solve(A_val, b_val)
        self.run_tests(
            compare_values=[csdl_tests.TestingPair(x, expected, decimal=8)],
            verify_derivatives=True,
        )

    def test_invalid_inputs(self):
        self.prep()
        import csdl_alpha as csdl

        with pytest.raises(TypeError, match='scipy sparse'):
            csdl.sparse.solve_linear(np.eye(3), np.ones(3))
        with pytest.raises(ValueError, match='b must be 1D or 2D'):
            csdl.sparse.solve_linear(sps.eye(3), np.ones((3, 1, 1)))

    def test_errors(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        A_shape = (4,4,3)
        b_shape = (4,1)
        A_val = (np.arange(np.prod(A_shape)).reshape(A_shape)+1.0)**2.0
        b_val = np.arange(np.prod(b_shape)).reshape(b_shape)
        
        # A = csdl.Variable(value = np.ones((2,2,3)))
        # A = sps.csr_array(A.value)
        # b = csdl.Variable(value = np.ones((2,2)))
        # with pytest.raises(ValueError):
        #     x = csdl.sparse.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,3)))
        A = sps.csr_array(A.value)
        b = csdl.Variable(value = np.ones((3,)))
        with pytest.raises(ValueError):
            x = csdl.sparse.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,2)))
        A = sps.csr_array(A.value)
        b = csdl.Variable(value = np.ones((3,)))
        with pytest.raises(ValueError):
            x = csdl.sparse.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,2)))
        A = sps.csr_array(A.value)
        b = csdl.Variable(value = np.ones((1,1)))
        with pytest.raises(ValueError):
            x = csdl.sparse.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,3)))
        A = sps.csr_array(A.value)
        b = csdl.Variable(value = np.ones((3,2)))
        with pytest.raises(ValueError):
            x = csdl.sparse.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,2)))
        A = sps.csr_array(A.value)
        b = csdl.Variable(value = np.ones((2,)))
        with pytest.raises(TypeError):
            x = csdl.sparse.solve_linear(A,b, 's')

    def test_docstrings(self):
        self.docstest(solve_linear)

if __name__ == '__main__':
    t = TestSparseLinear()
    t.overwrite_backend = 'inline'
    t.test_functionality()
    t.test_errors()
    t.test_docstrings()
    t.test_rhs_shapes(sps.csr_matrix, (3,))
    t.test_rhs_shapes(sps.csc_matrix, (3, 1))
    t.test_invalid_inputs()
