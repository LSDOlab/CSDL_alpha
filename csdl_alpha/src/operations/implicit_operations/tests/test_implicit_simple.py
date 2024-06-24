import csdl_alpha.utils.testing_utils as csdl_tests
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

def nl_model_vectorized_double():
    a = csdl.Variable(value=np.array([1.3, 1.5]))
    b = csdl.Variable(value=np.array([2.1, 2.0]))
    c = csdl.Variable(value=-1)
    x = csdl.ImplicitVariable(value=np.array([0.34, 0.30])) # solution:[0.38462, 0.38743]
    x2 = csdl.Variable(value=np.array([0.34, 0.30])) # solution:[0.38462, 0.38743]

    ax2 = a*((x+x2)/2)**2
    y = x - (-ax2 - c)/b
    y2 = x2 - (-ax2 - c)/(2*b)
    return (x, y), (x2, y2), (a, b, c)

class TestSimpleImplicit(csdl_tests.CSDLTest):

    def test_solvers_simple_vectorized(self):
        cases = []
        self.prep()

        # Gauss Seidel tests:
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': csdl.Variable(value=1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': csdl.Variable(value=np.ones((2,))*1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.27)}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.28}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.28), 'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.27, 'tolerance': csdl.Variable(value=1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {'residual_jac_kwargs': {'elementwise':True}}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {'residual_jac_kwargs': {'loop':False}}))
        
        # Jacobi tests:
        cases.append((csdl.nonlinear_solvers.Jacobi, {}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': 1.1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': csdl.Variable(value=1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': csdl.Variable(value=np.ones((2,))*1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.271)}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.281}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {}, {'residual_jac_kwargs': {'elementwise':True}}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {}, {'residual_jac_kwargs': {'loop':False}}))

        # Newtons method tests:
        cases.append((csdl.nonlinear_solvers.Newton, {}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': 1.1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': csdl.Variable(value=1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': csdl.Variable(value=np.ones((2,))*1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.271)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.281}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {}, {'residual_jac_kwargs': {'elementwise':True}}))
        cases.append((csdl.nonlinear_solvers.Newton, {}, {'residual_jac_kwargs': {'concatenate_ofs':True}}))

        # Bracketed search tests:
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4)}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=np.zeros((2,))), 4)}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4*np.ones((2,))))}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4*np.ones((2,)))),'tolerance': csdl.Variable(value=1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=np.zeros((2,))), csdl.Variable(value=4*np.ones((2,)))),'tolerance': 1e-7}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4)}, {'residual_jac_kwargs': {'elementwise':True}}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {}, {'residual_jac_kwargs': {'loop':False}}))

        compare_values = []
        for i, (solver, kwargs, solver_kwargs) in enumerate(cases):
            print('running case', i, solver, kwargs)

            x, y = nl_model_vectorized()

            # apply coupling:
            solver = solver(**solver_kwargs)
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

        # Jacobi tests
        cases.append((csdl.nonlinear_solvers.Jacobi, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': 1.1e-10}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': csdl.Variable(value=1.1e-8)}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.271)}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.281}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1.1e-10}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}))

        # Newton tests
        cases.append((csdl.nonlinear_solvers.Newton, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': 1.1e-10}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': csdl.Variable(value=1.1e-8)}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.271)}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.281}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1.1e-10}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}))

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

    def test_solvers_vectorized_double(self):
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
        
        # Jacobi tests:
        cases.append((csdl.nonlinear_solvers.Jacobi, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': 1.1e-10}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': csdl.Variable(value=1.1e-8)}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': csdl.Variable(value=np.ones((2,))*1.1e-8)}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.271)}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.281}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1e-10}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}))

        # Newtons method tests:
        cases.append((csdl.nonlinear_solvers.Newton, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': 1.1e-10}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': csdl.Variable(value=1.1e-8)}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': csdl.Variable(value=np.ones((2,))*1.1e-8)}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.271)}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.281}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1e-10}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}))

        # Bracketed search tests:
        # WARNING: In general, bracketed search does NOT work for coupled systems
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4)}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=np.zeros((2,))), 4)}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4*np.ones((2,))))}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4*np.ones((2,)))),'tolerance': csdl.Variable(value=1e-8)}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=np.zeros((2,))), csdl.Variable(value=4*np.ones((2,)))),'tolerance': 1e-7}))

        compare_values = []
        for i, (solver_class, kwargs) in enumerate(cases):
            print('running case', i, solver_class, kwargs)

            pair_1, pair_2, abc = nl_model_vectorized_double()

            x, y = pair_1[0], pair_1[1]
            x1, y1 = pair_2[0], pair_2[1]

            # apply coupling:
            solver = solver_class(print_status=False)
            solver.add_state(x, y, **kwargs)
            solver.add_state(x1, y1, **kwargs)
            solver.run()

            compare_values += [csdl_tests.TestingPair(x, np.array([0.41594571, 0.4241156]), tag = f'residual_{i}', decimal = 4)]
            compare_values += [csdl_tests.TestingPair(x1, np.array([0.20797276, 0.21205783]), tag = f'residual_{i}', decimal = 4)]

        self.run_tests(compare_values)

    def test_solvers_vectorized_double_deriv(self):
        cases = []
        self.prep()

        # Gauss Seidel tests
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {'residual_jac_kwargs': {'concatenate_ofs':True}}))

        # Jacobi tests:
        cases.append((csdl.nonlinear_solvers.Jacobi, {}, {}))

        # Newtons method tests:
        cases.append((csdl.nonlinear_solvers.Newton, {}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {}, {'residual_jac_kwargs': {'concatenate_ofs':True}}))

        # Bracketed search tests:
        # WARNING: In general, bracketed search does NOT work for coupled systems
        # cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 2), 'tolerance': 1e-12}))

        compare_values = []
        for i, (solver_class, kwargs, solver_kwargs) in enumerate(cases):
            print('running case', i, solver_class, kwargs)

            pair_1, pair_2, abc = nl_model_vectorized_double()

            x, y = pair_1[0], pair_1[1]
            x1, y1 = pair_2[0], pair_2[1]
            x.add_name(f'x_{solver_class.__name__}')
            x1.add_name(f'x1_{solver_class.__name__}')

            abc[0].add_name(f'a_{solver_class.__name__}')
            abc[1].add_name(f'b_{solver_class.__name__}')
            abc[2].add_name(f'c_{solver_class.__name__}')
            # apply coupling:
            solver = solver_class(print_status=False, **solver_kwargs)
            solver.add_state(x1, y1, **kwargs)
            solver.add_state(x, y, **kwargs)
            solver.run()

            compare_values += [csdl_tests.TestingPair(x, np.array([0.41594571, 0.4241156]), tag = f'residual_{i}', decimal = 4)]
            compare_values += [csdl_tests.TestingPair(x1, np.array([0.20797276, 0.21205783]), tag = f'residual_{i}', decimal = 4)]
            compare_values += [csdl_tests.TestingPair(x+x1, np.array([
                0.41594571+0.20797276, 0.4241156+0.21205783                
            ]), tag = f'residual_{i}', decimal = 4)]

        self.run_tests(compare_values, verify_derivatives=True, ignore_constants=True)

    def test_solvers_vectorized_nested(self):
        cases = []
        self.prep()

        # Gauss Seidel tests
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': csdl.Variable(value=1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.27)}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.28}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': csdl.Variable(value=0.28), 'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.27, 'tolerance': csdl.Variable(value=1e-8)}, {}))

        # Jacobi tests:
        cases.append((csdl.nonlinear_solvers.Jacobi, {}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': 1.1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': csdl.Variable(value=1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': csdl.Variable(value=np.ones((2,))*1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.271)}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.281}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}, {}))

        # Newtons method tests:
        cases.append((csdl.nonlinear_solvers.Newton, {}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': 1.1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': csdl.Variable(value=1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'tolerance': csdl.Variable(value=np.ones((2,))*1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.271)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.281}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': csdl.Variable(value=0.281), 'tolerance': 1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': 0.271, 'tolerance': csdl.Variable(value=1.1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {}, {'residual_jac_kwargs': {'concatenate_ofs':True}}))

        # Bracketed search tests:
        # WARNING: In general, bracketed search does NOT work for coupled systems
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4)}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=np.zeros((2,))), 4)}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4*np.ones((2,))))}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0.0, csdl.Variable(value=4*np.ones((2,)))),'tolerance': csdl.Variable(value=1e-8)}, {}))
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (csdl.Variable(value=np.zeros((2,))), csdl.Variable(value=4*np.ones((2,)))),'tolerance': 1e-7}, {}))

        compare_values = []
        for i, (solver_class, kwargs, solver_kwargs) in enumerate(cases):
            print('running case', i, solver_class, kwargs)

            pair_1, pair_2, abc = nl_model_vectorized_double()

            x, y = pair_1[0], pair_1[1]
            x1, y1 = pair_2[0], pair_2[1]

            # apply coupling:
            solver = solver_class(print_status=False, **solver_kwargs)
            solver.add_state(x, y, **kwargs)
            solver.run()

            solver = solver_class(print_status=False, **solver_kwargs)
            solver.add_state(x1, y1, **kwargs)
            solver.run()

            compare_values += [csdl_tests.TestingPair(x, np.array([0.41594571, 0.4241156]), tag = f'residual_{i}', decimal = 5)]
            compare_values += [csdl_tests.TestingPair(x1, np.array([0.20797276, 0.21205783]), tag = f'residual_{i}', decimal = 5)]

        self.run_tests(compare_values)


    def test_solvers_vectorized_nested_derivs2(self):
        cases = []
        self.prep()

        # Gauss Seidel tests
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}, {'residual_jac_kwargs': {'concatenate_ofs':True}}))

        # Newtons method tests:
        temp = csdl.Variable(name = 'initial', value=0.271)
        tol = csdl.Variable(name = 'tolerance', value=1.1e-10)
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': temp, 'tolerance':tol}, {}))

        # Bracketed search tests:
        # WARNING: In general, bracketed search does NOT work for coupled systems
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4), 'tolerance':1e-12}, {}))
        
        compare_values = []
        for i, (solver_class1, kwargs1, sl_kwargs1) in enumerate(cases):

            solver_class2, kwargs2, sl_kwargs2 = cases[i-1]

            print('running case', i, solver_class1, kwargs1, solver_class2, kwargs2)

            pair_1, pair_2, abc = nl_model_vectorized_double()

            x, y = pair_1[0], pair_1[1]
            x1, y1 = pair_2[0], pair_2[1]

            x.add_name(f'x_{solver_class1.__name__}')
            x1.add_name(f'x1_{solver_class2.__name__}')

            abc[0].add_name(f'a_{solver_class1.__name__}')
            abc[1].add_name(f'b_{solver_class1.__name__}')
            abc[2].add_name(f'c_{solver_class1.__name__}')

            # apply coupling:
            solver = solver_class1(print_status=False, **sl_kwargs1)
            solver.add_state(x, y, **kwargs1)
            solver.run()

            solver = solver_class2(print_status=False, **sl_kwargs2)
            solver.add_state(x1, y1, **kwargs2)
            solver.run()

            compare_values += [csdl_tests.TestingPair(x, np.array([0.41594571, 0.4241156]), tag = f'residual_{i}', decimal = 5)]
            compare_values += [csdl_tests.TestingPair(x1, np.array([0.20797276, 0.21205783]), tag = f'residual_{i}', decimal = 5)]

        self.run_tests(compare_values, verify_derivatives=True, ignore_derivative_fd_error={tol}, ignore_constants = True)

    def test_solvers_vectorized_nested_derivs_2(self):
        cases = []
        self.prep()

        # Gauss Seidel tests
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {}))

        # Newtons method tests:
        temp = csdl.Variable(name = 'initial', value=0.271)
        tol = csdl.Variable(name = 'tolerance', value=1.1e-10)
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': temp, 'tolerance':tol}))

        # Bracketed search tests:
        # WARNING: In general, bracketed search does NOT work for coupled systems
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 4), 'tolerance':1e-12}))

        compare_values = []
        for i, (solver_class, kwargs) in enumerate(cases):
            print('running case', i, solver_class, kwargs)

            pair_1, pair_2, abc = nl_model_vectorized_double()

            x, y = pair_1[0], pair_1[1]
            x1, y1 = pair_2[0], pair_2[1]

            x.add_name(f'x_{solver_class.__name__}')
            x1.add_name(f'x1_{solver_class.__name__}')

            abc[0].add_name(f'a_{solver_class.__name__}')
            abc[1].add_name(f'b_{solver_class.__name__}')
            abc[2].add_name(f'c_{solver_class.__name__}')

            # apply coupling:
            solver = solver_class(print_status=False)
            solver.add_state(x, y, **kwargs)
            solver.run()

            solver = solver_class(print_status=False)
            solver.add_state(x1, y1, **kwargs)
            solver.run()

            compare_values += [csdl_tests.TestingPair(x, np.array([0.41594571, 0.4241156]), tag = f'residual_{i}', decimal = 5)]
            compare_values += [csdl_tests.TestingPair(x1, np.array([0.20797276, 0.21205783]), tag = f'residual_{i}', decimal = 5)]

        self.run_tests(compare_values, verify_derivatives=True, ignore_derivative_fd_error={tol}, ignore_constants = True)


    def test_solvers_vectorized_double_deriv_bracket(self):
        cases = []
        self.prep(always_build_inline = True)

        # Bracketed search tests:
        # WARNING: In general, bracketed search does NOT work for coupled systems
        cases.append((csdl.nonlinear_solvers.BracketedSearch, {'bracket': (0, 2), 'tolerance': 1e-15}))

        compare_values = []
        for i, (solver_class, kwargs) in enumerate(cases):
            print('running case', i, solver_class, kwargs)

            a = csdl.Variable(value=np.array([1.3, 1.5]))
            b = csdl.Variable(value=np.array([2.1, 2.0]))
            c = csdl.Variable(value=-1)
            x = csdl.ImplicitVariable(value=np.array([0.34, 0.30])) # solution:[0.38462, 0.38743]
            x2 = csdl.ImplicitVariable(value=np.array([0.34, 0.30])) # solution:[0.38462, 0.38743]

            y = x - (-a*((x)/2)**2 - c)/b
            y2 = x2 - (-a*((x2)/2)**2 - c)/(2*b)

            x.add_name(f'x_{solver_class.__name__}')
            x2.add_name(f'x2_{solver_class.__name__}')

            a.add_name(f'a_{solver_class.__name__}')
            b.add_name(f'b_{solver_class.__name__}')
            c.add_name(f'c_{solver_class.__name__}')

            # apply coupling:
            solver = solver_class(print_status=True, max_iter=1000)
            solver.add_state(x, y, **kwargs)
            solver.add_state(x2, y2, **kwargs)
            solver.run()

            # print(x.value)
            # exit()

            # Analytical Jacobian: dx_da 
            # [[-0.03593611 -0.        ]
            # [-0.         -0.03725726]]
            # Finite Difference Jacobian: 
            # [[ 0.          0.        ]
            # [ 0.         -0.17881393]]

            compare_values += [csdl_tests.TestingPair(x, x.value, tag = f'residual_{i}', decimal = 4)]
            compare_values += [csdl_tests.TestingPair(x2, x2.value, tag = f'residual_{i}', decimal = 4)]
            compare_values += [csdl_tests.TestingPair(x+x2, x.value + x2.value, tag = f'residual_{i}', decimal = 4)]

        self.run_tests(compare_values, verify_derivatives=True, ignore_constants=True, step_size=1e-6)

if __name__ == '__main__':
    t = TestSimpleImplicit()
    # t.overwrite_backend = 'jax'
    # t.test_solvers_simple()
    # t.test_solvers_simple_vectorized()
    # t.test_solvers_vectorized_nested()
    # t.test_solvers_vectorized_double()
    # t.test_solvers_vectorized_nested_derivs()
    # t.test_solvers_vectorized_double_deriv()
    # t.test_solvers_vectorized_double_deriv_bracket()
    # t.test_solvers_vectorized_nested_derivs_2()

