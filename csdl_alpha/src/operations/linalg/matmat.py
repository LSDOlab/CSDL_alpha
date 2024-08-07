from csdl_alpha.src.graph.operation import Operation, set_properties 
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import pytest
import scipy.sparse as sp
from csdl_alpha.utils.typing import VariableLike

@set_properties()
class MatMat(Operation):

    def __init__(self, A:Variable, B:Variable) -> 'MatMat':
        super().__init__(A,B)
        self.name = 'matmat'
        self.set_dense_outputs(((A.shape[0], B.shape[1]),))

    def compute_inline(self, A, B):
        return A @ B
    
    def compute_jax(self, A, x):
        return A @ x

    def evaluate_vjp(self, cotangents, A, B, C):
        import csdl_alpha as csdl
        if cotangents.check(B):
            cotangents.accumulate(B, A.T()@cotangents[C])
        if cotangents.check(A):
            cotangents.accumulate(A, cotangents[C]@B.T())

def matmat(A:VariableLike, B:VariableLike) -> Variable:
    """matrix-matrix multiplication A*B. The number of columns of A must be equal to the number of rows of x.
    
    Parameters
    ----------
    A : Variable
        2D matrix
    B : Variable
        2D matrix (or 1D vector, in which case matvec is called instead)

    Returns
    -------
    C: Variable
        2D matrix

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> A = csdl.Variable(value = np.array([[1, 2], [3, 4], [5, 6]]))
    >>> B = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
    >>> (A @ B).value
    array([[ 7., 10.],
           [15., 22.],
           [23., 34.]])
    """
    from csdl_alpha.src.operations.linalg.matvec import matvec
    from csdl_alpha.src.operations.sparse.sparse_matvec import matmat as sparse_matmat

    if isinstance(A, sp.spmatrix):
        return sparse_matmat(A, B)

    A = validate_and_variablize(A, raise_on_sparse = False)
    B = validate_and_variablize(B, raise_on_sparse = False)

    # checks:
    # - A must be 2D
    # - B must be 2D 
    # - A.shape[1] == B.shape[0]
    if len(A.shape) != 2:
        raise ValueError(f"Matrix A must be 2D, but has shape {A.shape}")

    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Number of columns of A must be equal to the number of rows of B. {A.shape[1]} != {B.shape[0]}")
    
    if len(B.shape) == 1:
        return matvec(A, B)
    if len(B.shape) != 2:
        raise ValueError(f"Matrix B must be 2D, but has shape {B.shape}")

    return MatMat(A, B).finalize_and_return_outputs()

class TestMatMat(csdl_tests.CSDLTest):

    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        import scipy.sparse as sp

        A_shape = (3,4)
        B_shape = (4,1)
        A_val = np.arange(np.prod(A_shape)).reshape(A_shape)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        A = csdl.Variable(value = A_val)
        B = csdl.Variable(value = B_val)

        compare_values = []
        C = csdl.matmat(A,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C = A@B
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        B_shape = (4,3)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        B = csdl.Variable(value = B_val)
        C = csdl.matmat(A,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C = A@B
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        A_shape = (1,4)
        A_val = np.arange(np.prod(A_shape)).reshape(A_shape)
        A = csdl.Variable(value = A_val)
        C = csdl.matmat(A,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C = A@B
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        # with scalars
        B_shape = (4,3)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        C = csdl.matmat(A,B_val)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C = A@B_val
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        A_shape = (1,4)
        A_val = np.arange(np.prod(A_shape)).reshape(A_shape)
        C = csdl.matmat(A_val,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C =A_val@B
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        # matvec
        A_shape = (3,4)
        B_shape = (4,)
        A_val = np.arange(np.prod(A_shape)).reshape(A_shape)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        A = csdl.Variable(value = A_val)
        B = csdl.Variable(value = B_val)
        C = csdl.matmat(A,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C = A@B
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        # sparse matrix
        # NOTE: temporary
        A_data = np.array([1, 2, 3, 4, 5, 6])
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 0, 1])
        A = sp.csr_matrix((A_data, (row, col)), shape=A_shape)
        C = csdl.matmat(A,B)
        compare_values += [csdl_tests.TestingPair(C, A@B_val)]
        C = A@B
        compare_values += [csdl_tests.TestingPair(C, A@B_val)]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_errors(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np


        A = csdl.Variable(value = np.ones((2,2,3)))
        B = csdl.Variable(value = np.ones((2,1)))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B
            
        A = csdl.Variable(value = np.ones((2,3)))
        B = csdl.Variable(value = np.ones((2,1)))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B

        A = csdl.Variable(value = np.ones((2,3)))
        B = csdl.Variable(value = np.ones((2,)))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B

        A = csdl.Variable(value = np.ones((2,3)))
        B = csdl.Variable(value = np.ones((3,4,4)))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B

        # check with numpy arrays
        A = csdl.Variable(value = np.ones((2,2,3)))
        B = np.ones((2,1))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B
            
        A = csdl.Variable(value = np.ones((2,3)))
        B = np.ones((2,1))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B

        A = np.ones((2,3))
        B = csdl.Variable(value = np.ones((2,)))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B

        A = csdl.Variable(value = np.ones((2,3)))
        B = np.ones((3,4,4))
        with pytest.raises(ValueError):
            C = csdl.matmat(A, B)
        with pytest.raises(ValueError):
            C = A@B

    def test_docsstrings(self):
        self.docstest(matmat)

if __name__ == '__main__':
    t = TestMatMat()
    t.test_functionality()
    t.test_docsstrings()
    t.test_errors()