import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
import csdl_alpha as csdl
import numpy as np

def nl_model():
    size_1 = 2
    shape = (1, size_1, 3)
    size = np.prod(shape)
    v1 = np.array([[[0.5488135, 0.71518937, 0.60276338], [0.54488318, 0.4236548,  0.64589411]]]).reshape(shape)
    v2 = np.array([[[0.43758721, 0.891773, 0.96366276], [0.38344152, 0.79172504, 0.52889492]]]).reshape(shape)
    v3 = np.array([[[0.56804456, 0.92559664, 0.07103606], [0.0871293, 0.0202184, 0.83261985]]]).reshape(shape)

    p_eval = csdl.Variable(shape, value=v1)
    p1 = csdl.Variable(shape, value=v2)
    p2 = csdl.Variable(shape, value=v3)

    # find cross product and norm
    r1 = p_eval-p1
    r2 = p_eval-p2
    input_shape = r1.shape
    xyz_dim = len(input_shape) - 1

    r1r2_cross = csdl.cross(r1, r2, axis=xyz_dim)
    r1r2_cross = r1+r2
    r1r2_cross_norm = csdl.norm(r1r2_cross+1.e-1, axes=(xyz_dim,))
    r1r2_cross_norm_exp = csdl.expand(r1r2_cross_norm, r1.shape, 'ij->ija')

    # compute norms of r1 and r2
    r1_norm = csdl.norm(r1+1.e-1, axes=(xyz_dim, ))
    r1_norm_exp = csdl.expand(r1_norm, r1.shape, 'ij->ija')
    r2_norm = csdl.norm(r2+1.e-1, axes=(xyz_dim, ))
    r2_norm_exp = csdl.expand(r2_norm, r2.shape, 'ij->ija')

    # compute dot products 
    r0 = r1-r2
    r0r1_dot = csdl.sum(r0*r1, axes=(xyz_dim,))
    r0r1_dot_exp = csdl.expand(r0r1_dot, r1.shape, 'ij->ija')
    r0r2_dot = csdl.sum(r0*r2, axes=(xyz_dim,))
    r0r2_dot_exp = csdl.expand(r0r2_dot, r2.shape, 'ij->ija')

    temp = (r1r2_cross_norm_exp + 1.e-1)**2
    induced_vel = 1.0/(4*3.14159)*r1r2_cross
    induced_vel = induced_vel/temp
    induced_vel = induced_vel* (r0r1_dot_exp/r1_norm_exp - r0r2_dot_exp/r2_norm_exp)
    
    induced_vel.name = 'induced_vel'
    state_update = p_eval - 0.3*induced_vel
    return p_eval, induced_vel, state_update, induced_vel

class TestComplex(csdl_tests.CSDLTest):

    def test_solvers_complex(self):
        x_value_solved = np.array([[[0.50281589, 0.90868482, 0.51734941], [0.2352856, 0.40597173, 0.68075736]]])

        cases = []
        self.prep(debug = False, expand_ops = True, inline = False)

        # Gauss Seidel tests:
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'state_update': None}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'tolerance': 1.1e-10, 'state_update': None}, {}))
        cases.append((csdl.nonlinear_solvers.GaussSeidel, {'initial_value': 0.281, 'state_update': None}, {}))
        
        # Jacobi tests:
        cases.append((csdl.nonlinear_solvers.Jacobi, {'state_update': None}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'tolerance': 1.1e-10, 'state_update': None}, {}))
        cases.append((csdl.nonlinear_solvers.Jacobi, {'initial_value': 0.281, 'state_update': None}, {}))

        # Newtons method tests:
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': x_value_solved-0.03,}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': x_value_solved-0.03,'tolerance': 1.1e-10}, {}))
        cases.append((csdl.nonlinear_solvers.Newton, {'initial_value': x_value_solved-0.03}, {}))

        states = {}
        for i, (solver, kwargs, solver_kwargs) in enumerate(cases):
            print('running case', i, solver, kwargs)

            x, y, state_update, other = nl_model()

            # apply coupling:
            if 'state_update' in kwargs:
                kwargs['state_update'] = state_update
            solver = solver(**solver_kwargs)
            solver.add_state(x, y, **kwargs)
            solver.run()

            states[f'x_{i}'] = x

        from csdl_alpha.backends.jax.utils import verify_inline_vs_jax
        current_recorder = csdl.get_current_recorder()
        comparisons = verify_inline_vs_jax(
            current_recorder,
            rel_threshold = 1e-10,
            raise_error = True,
            print_errors = False,
        )
        
        for tag, x in states.items():
            print(tag, x.value, x_value_solved)
            assert np.allclose(x.value, x_value_solved)
        print('jax/inline verification complete')
        

if __name__ == '__main__':
    t = TestComplex()
    t.test_solvers_complex()
    print('done running test')