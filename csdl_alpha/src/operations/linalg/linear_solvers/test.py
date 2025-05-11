import csdl_alpha.utils.testing_utils as csdl_tests

class TestLinear(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        compare_values = []

        n = 10
        theta_row = np.sin(np.arange(n)+0.3)
        theta_col = np.cos(-np.arange(n))
        A_val, b_val = np.outer(theta_row, theta_col), np.sin(np.arange(n)**2.0)
        A = csdl.Variable(value = A_val)
        b = csdl.Variable(value = b_val)

        def linear_operator(x):
            return A @ x
        
        def linear_transpose_operator(x):
            return A.T() @ x

        x_assembled = csdl.linear.solve_gmres(
            A = A,
            b = b,
            transpose_solve = lambda x: csdl.krylov_solve(A.T(), x), 
        )

        x_matrix_free = csdl.linear.solve_gmres(
            A = linear_operator,
            b = b,
            transpose_solve = lambda x: csdl.krylov_solve(linear_transpose_operator, x),
        )

        x = csdl.solve_linear(A,b)

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

if __name__ == '__main__':
    test = TestLinear()
    test.overwrite_backend = 'jax'
    test.test_functionality()
    test.test_docstring()