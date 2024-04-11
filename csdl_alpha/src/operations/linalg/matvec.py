from csdl_alpha.src.graph.operation import Operation, set_properties 
import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import pytest

@set_properties()
class MatVec(Operation):

    def __init__(self, A:Variable, x:Variable) -> 'MatVec':
        super().__init__(A,x)
        self.name = 'matvec'
        self.set_dense_outputs(((A.shape[0], 1),))

    def compute_inline(self, A, x):
        return A @ x

def matvec(A:Variable, x:Variable) -> Variable:
    """matrix-vector multiplication A*x. The number of columns of A must be equal to the number of rows of x.
    If x is 1D, reshaped to 2D. 
    
    Parameters
    ----------
    A : Variable
        2D matrix
    x : Variable
        1D or 2D vector

    Returns
    -------
    y: Variable
        1D or 2D vector

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> A = csdl.Variable(value = np.array([[1, 2], [3, 4], [5, 6]]))
    >>> x = csdl.Variable(value = np.array([1, 2]))
    >>> csdl.matvec(A, x).value
    array([ 5, 11, 17])
    """

    A_mat = variablize(A)
    x_vec = variablize(x)

    # checks:
    # - A must be 2D
    # - x must be 2D (if 1D, reshaped to 2D)
    # - A.shape[1] == x.shape[0]
    if len(A_mat.shape) != 2:
        raise ValueError(f"Matrix A must be 2D, but has shape {A_mat.shape}")
    if len(x.shape) == 1:
        x_vec = x_vec.reshape((x_vec.size, 1))
    elif len(x.shape) == 2:
        if x_vec.shape[1] != 1:
            raise ValueError(f"x must have one column, but has shape {x_vec.shape}")
    elif len(x_vec.shape) != 2:
        raise ValueError(f"Vector x must be 1D or 2D, but has shape {x_vec.shape}")
    if A_mat.shape[1] != x_vec.shape[0]:
        raise ValueError(f"Number of columns of A must be equal to the number of rows of x. {A_mat.shape[1]} != {x_vec.shape[0]}")

    output = MatVec(A_mat, x_vec).finalize_and_return_outputs()

    if len(x.shape) == 2:
        return output
    if len(x.shape) == 1:
        return output.reshape((output.size,))

class TestMatVec(csdl_tests.CSDLTest):

    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        A_shape = (3,4)
        B_shape = (4,1)
        A_val = np.arange(np.prod(A_shape)).reshape(A_shape)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        A = csdl.Variable(value = A_val)
        B = csdl.Variable(value = B_val)

        compare_values = []
        C = csdl.matvec(A,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        B_shape = (4,)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        B = csdl.Variable(value = B_val)
        C = csdl.matvec(A,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        C = csdl.matvec(A_val,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C = csdl.matvec(A,B_val)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        B_shape = (4,)
        B_val = np.arange(np.prod(B_shape)).reshape(B_shape)
        C = csdl.matvec(A,B_val)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]
        C = csdl.matvec(A_val,B)
        compare_values += [csdl_tests.TestingPair(C, A_val@B_val)]

        self.run_tests(compare_values = compare_values,)

    def test_errors(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        A = csdl.Variable(value = np.ones((2,2)))
        B = csdl.Variable(value = np.ones((2,2)))
        with pytest.raises(ValueError):
            C = csdl.matvec(A, B)

        A = csdl.Variable(value = np.ones((2,2,3)))
        B = csdl.Variable(value = np.ones((2,1)))
        with pytest.raises(ValueError):
            C = csdl.matvec(A, B)

        A = csdl.Variable(value = np.ones((2,3)))
        B = csdl.Variable(value = np.ones((2,1)))
        with pytest.raises(ValueError):
            C = csdl.matvec(A, B)

        A = csdl.Variable(value = np.ones((2,3)))
        B = csdl.Variable(value = np.ones((2,)))
        with pytest.raises(ValueError):
            C = csdl.matvec(A, B)

        A = csdl.Variable(value = np.ones((2,3)))
        B = csdl.Variable(value = np.ones((3,4,4)))
        with pytest.raises(ValueError):
            C = csdl.matvec(A, B)

    def test_docsstrings(self):
        self.docstest(matvec)

if __name__ == '__main__':
    t = TestMatVec()
    t.test_functionality()
    t.test_docsstrings()
    t.test_errors()