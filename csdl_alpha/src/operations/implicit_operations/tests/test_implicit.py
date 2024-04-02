import csdl_alpha.utils.test_utils as csdl_tests
import pytest

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
                csdl_tests.TestingPair(ax2, np.array([1.5*0.38742589**2]), tag = 'ax2', decimal=7),
            ],
        )


    def test_arg_errors(self,):
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

        # state must be CSDL implicit variable
        with pytest.raises(TypeError) as e_info:
            solver.add_state(c, y)
        
        # residual must be CSDL variable
        with pytest.raises(TypeError) as e_info:
            solver.add_state(x, np.ones((1,)))

        # residual must be CSDL variable
        # state residual variable shapes must match
        y_wrong = csdl.Variable(name = 'y', value = np.ones(3,))
        with pytest.raises(ValueError) as e_info:
            solver.add_state(x, y_wrong)

        # there must be a state/residual added to nonlinear solver
        with pytest.raises(ValueError) as e_info:
            solver.run()


    def test_no_state_res_dependence(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(name = 'a', value = 1.5)
        x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.34)
        y = csdl.Variable(name = 'y', value = 1.5)

        solver = csdl.GaussSeidel('gs_x_simpler')
        # y must depend on x
        solver.add_state(x, y)
            
        with pytest.raises(ValueError) as e_info:
            solver.run()

    def test_insufficient_res_state_dependence_1(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(name = 'a', value = 1.5)
        x1 = csdl.ImplicitVariable(shape=(1,), name='x1', value=0.34)
        x2 = csdl.ImplicitVariable(shape=(1,), name='x2', value=0.34)
        
        y1 = x2*2
        y2 = x2*3

        solver = csdl.GaussSeidel('gs_x_simpler')
        solver.add_state(x1, y1)
        solver.add_state(x2, y2)

        # x1 must affect atleast one residual
        with pytest.raises(ValueError) as e_info:
            solver.run()

    def test_insufficient_res_state_dependence_2(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(name = 'a', value = 1.5)
        x1 = csdl.ImplicitVariable(shape=(1,), name='x1', value=0.34)
        x2 = csdl.ImplicitVariable(shape=(1,), name='x2', value=0.34)
        
        y1 = x1*2+x2
        y2 = a*3

        solver = csdl.GaussSeidel('gs_x_simpler')
        solver.add_state(x1, y1)
        solver.add_state(x2, y2)

        # x1 must affect atleast one residual
        with pytest.raises(ValueError) as e_info:
            solver.run()

if __name__ == '__main__':
    t = TestImplicit()
    # t.test_values()
    # t.test_arg_errors()
    # t.test_insufficient_res_state_dependence_1()
    # t.test_insufficient_res_state_dependence_2()
