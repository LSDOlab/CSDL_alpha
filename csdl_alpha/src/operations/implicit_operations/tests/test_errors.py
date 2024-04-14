import csdl_alpha.utils.test_utils as csdl_tests
import pytest
import csdl_alpha as csdl
import numpy as np

class TestErrorsBracket(csdl_tests.CSDLTest):

    def test_gauss_seidel(self):
        self.prep()

        a = csdl.Variable(value=1.5)
        b = csdl.Variable(value=2)
        c = csdl.Variable(value=-1)
        x = csdl.ImplicitVariable((1,), value=0.34)

        ax2 = a*x**2
        y = x - (-ax2 - c)/b

        solver = csdl.nonlinear_solvers.GaussSeidel()
        with pytest.raises(ValueError):
            solver.add_state(x, y, state_update=1)
            solver.run()

    def test_bracket(self):
        self.prep()

        a = csdl.Variable(value=1.5)
        b = csdl.Variable(value=2)
        c = csdl.Variable(value=-1)
        x = csdl.ImplicitVariable((1,), value=0.34)

        ax2 = a*x**2
        y = x - (-ax2 - c)/b

        solver = csdl.nonlinear_solvers.BracketedSearch()

        with pytest.raises(TypeError):
            solver.add_state(x, y)
        with pytest.raises(TypeError):
            x = csdl.ImplicitVariable((1,), value=0.34)
            solver.add_state(x, y, bracket=4)
        with pytest.raises(TypeError):
            x = csdl.ImplicitVariable((1,), value=0.34)
            solver.add_state(x, y, bracket=(4, 's'))
        with pytest.raises(ValueError):
            x = csdl.ImplicitVariable((1,), value=0.34)
            solver.add_state(x, y, bracket=(4, 5, 6))
        with pytest.raises(ValueError):
            x = csdl.ImplicitVariable((1,), value=0.34)
            solver.add_state(x, y, bracket=(np.array([4,1]), 5))
        with pytest.raises(ValueError):
            x = csdl.ImplicitVariable((1,), value=0.34)
            solver.add_state(x, y, bracket=(csdl.Variable(value = np.array([4,1])), 5))
        with pytest.raises(TypeError):
            x = csdl.ImplicitVariable((1,), value=0.34)
            solver.add_state(x, y, bracket=(4, 5), tolerance='s')
        with pytest.raises(ValueError):
            x = csdl.ImplicitVariable((1,), value=0.34)
            solver.add_state(x, y, bracket=(4, 5), tolerance=np.array([4,1]))

    def test_general(self):
        self.prep()

        a = csdl.Variable(value=1.5)
        b = csdl.Variable(value=2)
        c = csdl.Variable(value=-1)
        x = csdl.ImplicitVariable((1,), value=0.34)

        ax2 = a*x**2
        y = x - (-ax2 - c)/b

        solver = csdl.nonlinear_solvers.GaussSeidel()
        with pytest.raises(TypeError):
            solver.add_state(3.0, y)
        with pytest.raises(TypeError):
            solver.add_state(x, 3.0)

        solver = csdl.nonlinear_solvers.GaussSeidel()
        with pytest.raises(ValueError):
            solver.add_state(x, y)
            solver.run()
            solver.add_state(x, y)

        solver = csdl.nonlinear_solvers.GaussSeidel()
        with pytest.raises(ValueError):
            solver.add_state(x, y)
            solver.add_state(x, y)
