from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.test_utils as csdl_tests

import numpy as np
import scipy.sparse as sp
import pytest

@set_properties(supports_sparse=True)
class SparseMatVec(Operation):

    def __init__(self, x:Variable, sparse_matrix:sp.sparray) -> 'MatVec':
        super().__init__(x)
        self.name = 'sp_matvec'
        self.A = sparse_matrix
        self.set_dense_outputs(((sparse_matrix.shape[0], 1),))

    def compute_inline(self, x):
        return self.A @ x

# TODO: A will be variablized to sparse csdl matrix in the future
def matvec(A:sp.sparray, x:Variable) -> Variable:
    """sparse matrix-vector multiplication A*x for Andrew.
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

    if len(x_vec.shape) != 2:
        raise ValueError(f"x must be a 2D vector. Got {len(x_vec.shape)} dimensions")
    if len(A.shape) != 2:
        raise ValueError(f"A must be a 2D matrix. Got {len(A.shape)} dimensions")

    if A.shape[1] != x_vec.shape[0]:
        raise ValueError(f"Number of columns of A ({A.shape[1]}) must be equal to the number of rows of x ({x_vec.shape[0]})")
    if x_vec.shape[1] != 1:
        raise ValueError(f"x must be a column vector. Got {x_vec.shape[1]} columns")

    output = SparseMatVec(x_vec, sparse_matrix = A).finalize_and_return_outputs()
    return output


class TestSparseMatVec(csdl_tests.CSDLTest):

    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        data = np.array([1, 2, 3, 4, 5, 6])
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 0, 1])
        A = sp.csr_matrix((data, (row, col)), shape=(3, 2))
        x_val = np.array([[1], [2]])
        x = csdl.Variable(value = x_val)
        
        y = csdl.sparse.matvec(A, x)
        compare_values = []
        compare_values += [csdl_tests.TestingPair(y, A@x_val)]


        self.run_tests(compare_values = compare_values,)

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

if __name__ == '__main__':
    test = TestSparseMatVec()
    test.test_functionality()
    test.test_errors()