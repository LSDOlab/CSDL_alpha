import csdl_alpha.utils.test_utils as csdl_tests
import pytest
import csdl_alpha as csdl
import numpy as np

def nl_model():
    a = csdl.Variable(value=1.5)
    b = csdl.Variable(value=2)
    c = csdl.Variable(value=-1)
    x = csdl.ImplicitVariable((1,), value=0.34) #solution: [0.38743]

    ax2 = a*x**2
    y = x - (-ax2 - c)/b
    return x, y


def nl_model_vectorized():
    a = csdl.Variable(value=np.array([1.3, 1.5]))
    b = csdl.Variable(value=np.array([2.1, 2.0]))
    c = csdl.Variable(value=-1)
    x = csdl.ImplicitVariable(value=np.array([0.34, 0.30])) # solution:[0.38462, 0.38743]

    ax2 = a*x**2
    y = x - (-ax2 - c)/b
    return x, y

class TestSimpleImplicit(csdl_tests.CSDLTest):

    def test_solvers_simple_vectorized(self):
        cases = []
        self.prep()
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': 1e-10}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': csdl.Variable(value=1e-8)}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': csdl.Variable(value=np.ones((2,))*1e-8)}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.27)}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.28}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.28), 'tolerance': 1e-10}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.27, 'tolerance': csdl.Variable(value=1e-8)}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4)}))

        compare_values = []
        for i, (solver, kwargs) in enumerate(cases):
            print('running case', i, solver, kwargs)

            x, y = nl_model_vectorized()

            # apply coupling:
            solver = solver()
            solver.add_state(x, y, **kwargs)
            solver.run()

            compare_values += [csdl_tests.TestingPair(x, np.array([0.38461538, 0.38742589]), tag = f'residual_{i}', decimal = 5)]

        self.run_tests(compare_values)

    def test_solvers_simple(self):
        cases = []
        self.prep()

        # Gauss Seidel tests
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': 1e-10}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': csdl.Variable(value=1e-8)}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.27)}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.28}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.28), 'tolerance': 1e-10}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.27, 'tolerance': csdl.Variable(value=1e-8)}))

        # Bracketed Search tests
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4)}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=0.0), 4)}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4.0))}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4.0)),'tolerance': csdl.Variable(value=1e-8)}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=0.0), csdl.Variable(value=4.0)),'tolerance': 1e-7}))

        compare_values = []
        for i, (solver, kwargs) in enumerate(cases):
            print('running case', i, solver, kwargs)

            x, y = nl_model()

            # apply coupling:
            solver = solver()
            solver.add_state(x, y, **kwargs)
            solver.run()

            compare_values += [csdl_tests.TestingPair(x, np.array([0.38742589]), tag = f'residual_{i}', decimal = 5)]

        self.run_tests(compare_values)

if __name__ == '__main__':
    t = TestSimpleImplicit()
    # t.test_solvers_simple()
    t.test_solvers_simple_vectorized()