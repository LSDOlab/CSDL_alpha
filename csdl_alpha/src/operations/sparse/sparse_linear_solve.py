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
from jax.experimental import sparse
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
        # return jnp.linalg.solve(A, b)
        # Acoo = self.A.tocoo()
        # data = np.array(Acoo.data)
        # indices = np.array([Acoo.row, Acoo.col]).T
        # indptr = np.array(Acoo.indptr)
        # A = sparse.BCOO((data, indices), shape = self.A.shape)
        Acsr = self.A.tocsr()
        # A = sparse.BCSR.from_bcoo(A)
        # return sparse.linalg.spsolve(data=data, indices=indices, indptr=indptr, b=b).reshape(b.shape)
        # return sparse.linalg.spsolve(data=jnp.asarray(Acsr.data), indices=jnp.array(Acsr.indices), indptr=jnp.array(Acsr.indptr), b=b.flatten(),
        #                              reorder=1).reshape(b.shape)
        x = jnp.zeros((self.A.shape[0], 3))

        for i in range(b.shape[1]):
            x = x.at[:, i].set(
                sparse.linalg.spsolve(jnp.asarray(Acsr.data), jnp.array(Acsr.indices), jnp.array(Acsr.indptr), b[:, i])
            )
        return x



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
    A : VariableLike
        2D matrix
    b : VariableLike
        1D or 2D vector

    Returns
    -------
    x: Variable
        1D or 2D vector


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
    if not isinstance(A, sps.spmatrix):
        raise TypeError(f"A must be a scipy sparse matrix. Got {type(A)}")
    b = validate_and_variablize(b)

    # A_mat, b_vec = process_matA_vecb(A, b)
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