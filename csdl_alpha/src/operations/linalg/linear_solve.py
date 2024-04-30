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

@set_properties()
class LinearSolve(Operation):
    def __init__(self, A:Variable, b:Variable, solver:LinearSolver = DirectSolver) -> 'LinearSolve':
        super().__init__(A,b)
        self.name = 'linear_solve'
        self.set_dense_outputs((b.shape,))

    def compute_inline(self, A, b):
        return np.linalg.solve(A, b)
    
    def evaluate_vjp(self, cotangents, A, b, x):
        import csdl_alpha as csdl

        solved_system =  -csdl.solve_linear(A.T(), cotangents[x])
        if cotangents.check(b):
            cotangents.accumulate(b, solved_system)
        if cotangents.check(A):
            pass

def solve_linear(
        A:VariableLike,
        b:VariableLike,
        solver:LinearSolver = DirectSolver(),
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
    A = validate_and_variablize(A, raise_on_sparse=False)
    b = validate_and_variablize(b)
    A_mat, b_vec = process_matA_vecb(A, b)

    if A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError(f"Matrix A must be square, but has shape {A.shape}")

    if not isinstance(solver, LinearSolver):
        raise TypeError(f"Solver must be a LinearSolver. Got {type(solver)}.")

    output = LinearSolve(A_mat, b_vec, solver).finalize_and_return_outputs()

    if len(b.shape) == 2:
        return output
    if len(b.shape) == 1:
        return output.reshape((output.size,))
    
class TestLinear(csdl_tests.CSDLTest):

    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        A_shape = (4,4)
        b_shape = (4,1)
        A_val = (np.arange(np.prod(A_shape)).reshape(A_shape)+1.0)**2.0
        b_val = np.arange(np.prod(b_shape)).reshape(b_shape)
        A = csdl.Variable(value = A_val)
        b = csdl.Variable(value = b_val)

        compare_values = []
        x = csdl.solve_linear(A,b)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]
        x = csdl.solve_linear(A_val,b)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]
        x = csdl.solve_linear(A,b_val)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]

        b_shape = (4,)
        b_val = np.arange(np.prod(b_shape)).reshape(b_shape)
        b = csdl.Variable(value = b_val) 
        x = csdl.solve_linear(A,b)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val).flatten())]
        x = csdl.solve_linear(A,b_val)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val).flatten())]
        x = csdl.solve_linear(A_val,b)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val).flatten())]

        compare_values = []
        x = csdl.solve_linear(A,b, solver = csdl.linear_solvers.ScipyKrylovSolver())
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]
        x = csdl.solve_linear(A_val,b, solver = csdl.linear_solvers.ScipyKrylovSolver())
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]
        x = csdl.solve_linear(A,b_val, solver = csdl.linear_solvers.DirectSolver())
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)
    
    def test_errors(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        A_shape = (4,4,3)
        b_shape = (4,1)
        A_val = (np.arange(np.prod(A_shape)).reshape(A_shape)+1.0)**2.0
        b_val = np.arange(np.prod(b_shape)).reshape(b_shape)
        
        A = csdl.Variable(value = np.ones((2,2,3)))
        b = csdl.Variable(value = np.ones((2,2)))
        with pytest.raises(ValueError):
            x = csdl.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,3)))
        b = csdl.Variable(value = np.ones((3,)))
        with pytest.raises(ValueError):
            x = csdl.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,2)))
        b = csdl.Variable(value = np.ones((3,)))
        with pytest.raises(ValueError):
            x = csdl.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,2)))
        b = csdl.Variable(value = np.ones((1,1)))
        with pytest.raises(ValueError):
            x = csdl.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,3)))
        b = csdl.Variable(value = np.ones((3,2)))
        with pytest.raises(ValueError):
            x = csdl.solve_linear(A,b)

        A = csdl.Variable(value = np.ones((2,2)))
        b = csdl.Variable(value = np.ones((2,)))
        with pytest.raises(TypeError):
            x = csdl.solve_linear(A,b, 's')

    def test_docstrings(self):
        self.docstest(solve_linear)

if __name__ == '__main__':
    t = TestLinear()
    t.test_functionality()
    t.test_errors()
    t.test_docstrings()