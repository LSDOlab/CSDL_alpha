import csdl_alpha.utils.test_utils as csdl_tests

class TestImplicit(csdl_tests.CSDLTest):
    def test_values(self,):

        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(name = 'a', value = 1.5)
        b = csdl.Variable(name = 'b', value = 0.5)
        c = csdl.Variable(name = 'c', value = -1.0)
        x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.34)

        ax2 = a*x*x
        y = x - (-ax2 - c)*b
        y.name = 'residual_x'

        # sum of solved states
        sum_states = x+x
        sum_states.name = 'state_sum'

        # apply coupling:
        # ONE SOLVER COUPLING:
        solver = csdl.GaussSeidel('gs_x_simpler')
        solver.add_state(x, y)
        solver.run()

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(y, np.array([0.0]), tag = 'residual', decimal = 9),
                csdl_tests.TestingPair(x, np.array([0.38742589]), tag = 'state', decimal = 7),
                csdl_tests.TestingPair(a, np.array([1.5]), tag = 'a'),
                csdl_tests.TestingPair(b, np.array([0.5]), tag = 'b'),
                csdl_tests.TestingPair(c, np.array([-1.0]), tag = 'c'),
            ],
        )