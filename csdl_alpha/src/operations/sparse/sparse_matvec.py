from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.operations.linalg.utils import process_matA_vecb

import numpy as np
import scipy.sparse as sp
import pytest

@set_properties(supports_sparse=True)
class SparseMatMat(Operation):

    def __init__(self, sparse_matrix, x:Variable) -> 'MatVec':
        super().__init__(x)
        self.name = 'sp_matvec'
        self.A = sparse_matrix
        self.set_dense_outputs(((sparse_matrix.shape[0], x.shape[1]),))

    def compute_inline(self, x):
        return self.A @ x
    
    def compute_jax(self, x):
        from jax.experimental import sparse
        import jax.numpy as jnp
        Acoo = self.A.tocoo()
        data = np.array(Acoo.data)
        indices = np.array([Acoo.row, Acoo.col]).T
        A = sparse.BCOO((data, indices), shape = self.A.shape)
        return A @ x

    def evaluate_vjp(self, cotangents, x, b):
        import csdl_alpha as csdl
        if cotangents.check(x):
            cotangents.accumulate(x, csdl.sparse.matmat(self.A.T, cotangents[b]))

# TODO: A will be variablized to sparse csdl matrix in the future
def matvec(A, x:Variable) -> Variable:
    """(TEMPORARY) sparse matrix-vector multiplication A*x.
    The number of columns of A must be equal to the number of rows of x.

    Parameters
    ----------
    A : scipy sparse matrix
        2D matrix

    x : Variable

    Returns
    -------
    Variable

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> row = np.array([0, 1, 2, 3, 4, 5])
    >>> col = np.array([5, 4, 3, 2, 1, 0])
    >>> A = sp.csr_matrix((data, (row, col)), shape=(6,6))
    >>> x_val = np.arange(6).reshape((6,1))
    >>> x = csdl.Variable(value = x_val)
    >>> csdl.sparse.matvec(A, x).value
    array([[5.],
           [8.],
           [9.],
           [8.],
           [5.],
           [0.]])
    """
    if not isinstance(A, sp.spmatrix):
        raise TypeError(f"A must be a scipy sparse matrix. Got {type(A)}")
    x_vec = validate_and_variablize(x)

    output = SparseMatMat(*process_matA_vecb(A, x_vec)).finalize_and_return_outputs()

    if len(x.shape) == 2:
        return output
    if len(x.shape) == 1:
        return output.reshape((output.size,))

# TODO: A will be variablized to sparse csdl matrix in the future
def matmat(A, x:Variable) -> Variable:
    """(TEMPORARY) sparse matrix-matrix multiplication A*x.
    The number of columns of A must be equal to the number of rows of x.

    Parameters
    ----------
    A : scipy sparse matrix
        2D matrix

    x : Variable

    Returns
    -------
    Variable

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> row = np.array([0, 1, 2, 3, 4, 5])
    >>> col = np.array([5, 4, 3, 2, 1, 0])
    >>> A = sp.csr_matrix((data, (row, col)), shape=(6,6))
    >>> x_val = np.arange(12).reshape((6,2))
    >>> x = csdl.Variable(value = x_val)
    >>> csdl.sparse.matmat(A, x).value
    array([[10., 11.],
           [16., 18.],
           [18., 21.],
           [16., 20.],
           [10., 15.],
           [ 0.,  6.]])
    """
    if not isinstance(A, sp.spmatrix):
        raise TypeError(f"A must be a scipy sparse matrix. Got {type(A)}")
    x = validate_and_variablize(x)

    # checks:
    # - A must be 2D
    # - B must be 2D 
    # - A.shape[1] == B.shape[0]
    if len(A.shape) != 2:
        raise ValueError(f"Matrix A must be 2D, but has shape {A.shape}")

    if A.shape[1] != x.shape[0]:
        raise ValueError(f"Number of columns of A must be equal to the number of rows of B. {A.shape[1]} != {x.shape[0]}")
    
    if len(x.shape) == 1:
        return matvec(A, x)
    if len(x.shape) != 2:
        raise ValueError(f"Matrix B must be 2D, but has shape {x.shape}")
    
    return SparseMatMat(A, x).finalize_and_return_outputs()


class TestSparseMatMat(csdl_tests.CSDLTest):

    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        data = np.array([1, 2, 3, 4, 5, 6])
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 0, 1])
        A = sp.csr_matrix((data, (row, col)), shape=(3, 2))
        x_val = np.array([1, 2])
        x = csdl.Variable(value = x_val)
        
        y = csdl.sparse.matvec(A, x)
        compare_values = []
        compare_values += [csdl_tests.TestingPair(y, A@x_val)]

        x_val = np.array([[1], [2]])
        x = csdl.Variable(value = x_val)
        y = csdl.sparse.matvec(A, x)
        compare_values = []
        compare_values += [csdl_tests.TestingPair(y, A@x_val)]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_matmat_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        # ================ test 1: (3,2)x(2,2) ================
        data = np.array([1, 2, 3, 4, 5, 6])
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 0, 1])
        A = sp.csr_matrix((data, (row, col)), shape=(3, 2))
        x_val = np.array([[1, 2],[3, 4]])
        x = csdl.Variable(value = x_val)
        
        y = csdl.sparse.matmat(A, x)
        assert y.shape == (3,2)
        y.add_name('y_1')
        compare_values = []
        compare_values += [csdl_tests.TestingPair(y, A@x_val)]

        # ================ test 2: (4,4)x(4,3) ================
        B_shape = (4,3)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        B = csdl.Variable(value = B_val)
        data = np.array([1, 2, 3, 4, 5, 6])
        row = np.array([0, 0, 1, 3, 2, 2])
        col = np.array([0, 1, 0, 3, 0, 1])
        A = sp.csr_matrix((data, (row, col)), shape=(4, 4))
        y = csdl.sparse.matmat(A, B)
        y.add_name('y_2')
        assert y.shape == (4,3)
        compare_values += [csdl_tests.TestingPair(y, A@B_val)]

        # ================ test 3: (10,4)x(4,10) ================
        B_shape = (4,10)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        B = csdl.Variable(value = B_val)
        data = np.array([1, 2, 3, 4, 5, 6])
        col = np.array([3, 0, 1, 3, 0, 2])
        row = np.array([0, 9, 9, 6, 2, 1])
        A = sp.csr_matrix((data, (row, col)), shape=(10, 4))
        y = csdl.sparse.matmat(A, B)
        y.add_name('y_3')
        assert y.shape == (10,10)
        compare_values += [csdl_tests.TestingPair(y, A@B_val)]
        compare_values += [csdl_tests.TestingPair(csdl.norm(y), np.linalg.norm(A@B_val).flatten())]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_errors(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        data = np.array([1, 2, 3, 4, 5, 6])
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 0, 1])
        A = sp.csr_matrix((data, (row, col)), shape=(3, 2))
        x_val = np.array([[1], [2]])
        x = csdl.Variable(value = x_val)

        with pytest.raises(TypeError):
            y = csdl.matvec(A, x)
        
        with pytest.raises(ValueError):
            x_val = np.array([[1], [2], [3]])
            x = csdl.Variable(value = x_val)
            y = csdl.sparse.matvec(A, x)

        with pytest.raises(ValueError):
            B_shape = (4,3)
            B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
            B = csdl.Variable(value = B_val)
            y = csdl.sparse.matmat(A, B)

        with pytest.raises(ValueError):
            B_shape = (4,3,3)
            B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
            B = csdl.Variable(value = B_val)
            y = csdl.sparse.matmat(A, B)

        with pytest.raises(TypeError):
            x_val = np.array([[1], [2]])
            x = csdl.Variable(value = x_val)
            A = np.ones((2,2))
            y = csdl.sparse.matvec(A, x)

        with pytest.raises(TypeError):
            x_val = np.array([[1], [2]])
            x = csdl.Variable(value = x_val)
            A = csdl.Variable(value = np.ones((2,2)))
            y = csdl.sparse.matvec(A, x)


    def test_docstrings(self):
        self.docstest(matvec)
        self.docstest(matmat)

if __name__ == '__main__':
    test = TestSparseMatMat()
    test.overwrite_backend = 'inline'
    test.test_functionality()
    test.test_matmat_functionality()
    test.test_errors()