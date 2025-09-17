from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation, subgraph_operationify
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.linalg.linear_solvers.utils import process_linsolve_A_b, return_b, build_matvec_subgraph

import pytest
import numpy as np
from typing import Callable, Union
import jax

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            if np.mod(self.niter, 5) == 0:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))


@set_properties()
class MatFreeLinearSolve(SubgraphOperation):
    def __init__(self, A_inputs, b, Av, v, A_subgraph, transpose_solve, A_func, x0):
        super().__init__(*A_inputs,b) # pass in inputs
        self.set_dense_outputs((b.shape,))
        self.assign_subgraph(A_subgraph)
        self.name = 'gmres'

        self.subgraph_Av:Variable = Av
        self.subgraph_v:Variable = v
        self.subgraph_A_inputs:list[Variable] = A_inputs
        self.n = b.size
        self.transpose_solve = transpose_solve
        self.A_func = A_func
        self.x0 = x0

        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        matvec_func = create_jax_function(
            self.get_subgraph(),
            [self.subgraph_Av],
            [self.subgraph_v] + self.subgraph_A_inputs,
        )
        # potentially jit to the GPU
        jax.config.update("jax_enable_x64", True)
        self.matvec_func = jax.jit(matvec_func, device=jax.devices("cpu")[0])
        # replace gpu with cpu to jit between the two

    def compute_inline(self, *inputs):
        from scipy.sparse.linalg import gmres
        from scipy.sparse.linalg import LinearOperator
        import jax

        subgraph_inputs = inputs[:-1]
        b = inputs[-1]

        subgraph = self.get_subgraph()
        for input_var, input_val in zip(subgraph.inputs, subgraph_inputs):
            input_var.value = input_val
        
        
        matvec_func = self.matvec_func
        
        matvec_func_single = lambda v: matvec_func(v, *[jax.numpy.array(x) for x in subgraph_inputs])[0]

        def matvec_func_inline(v):
            # self.subgraph_v.value = v
            # subgraph.execute_inline()
            # Av = self.subgraph_Av.value
            v=jax.numpy.array(v)
            Av = matvec_func_single(v)
            return np.array(Av)

        counter = gmres_counter()
        # x, int = gmres(LinearOperator((self.n,self.n), matvec=matvec_func_inline), b, tol = 1e-12)
        x, int = gmres(LinearOperator((self.n,self.n), matvec=matvec_func_inline), b, x0=self.x0, tol = 1e-12, callback=counter)
        # print('GMRES # iterations:', int)
        print('GMRES # iterations:', counter.niter)
        # print(f'counter: {counter.niter}')
        return x.reshape(b.shape)
    
    def compute_jax(self, *inputs):
        from csdl_alpha.backends.jax.utils import fallback_to_inline_jax
        return fallback_to_inline_jax(self,*inputs)
        # JAX stuff
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import gmres
        import jax

        # get inputs
        subgraph_inputs = inputs[:-1]
        b = inputs[-1]

        # build matvec function
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        matvec_func = create_jax_function(
            self.get_subgraph(),
            [self.subgraph_Av],
            [self.subgraph_v] + self.subgraph_A_inputs,
        )
        
        matvec_func_single = lambda v: matvec_func(v, *subgraph_inputs)[0]
        # x_solved, _ = gmres(matvec_func_single, b.flatten(), tol=1e-12, maxiter=None)
        x_solved, _ = gmres(matvec_func_single, b.flatten(), tol=1e-6, maxiter=1, restart=1)
        
        return x_solved

    def evaluate_vjp(self, cotangents, *inputs_outputs):
        import csdl_alpha as csdl
        inputs = inputs_outputs[:self.num_inputs]
        outputs = inputs_outputs[self.num_inputs:]

        A_inputs = inputs[:-1]
        b = inputs[-1]
        x = outputs[0]

        solved_system = self.transpose_solve(cotangents[x].reshape((self.n,))).reshape((self.n,))
        if cotangents.check(b):
            cotangents.accumulate(b, solved_system)

        # Now for the operator inputs
        rec = csdl.get_current_recorder()
        from csdl_alpha.src.operations.derivatives.reverse import vjp

        def matvec_vjp_func(cot_vec,x):
            cot_vec = cot_vec.reshape((self.n,))
            Ax = self.A_func(x).reshape((self.n,))
            wrt_cotangents = vjp([(Ax, cot_vec)], A_inputs, rec.active_graph)
            outs = []
            for input in A_inputs:
                outs.append(-wrt_cotangents[input])
            
            return tuple(outs)

        matvec_vjp_func = subgraph_operationify(matvec_vjp_func)
        matvec_vjps = matvec_vjp_func(solved_system,x)

        # rec.visualize_graph(visualize_style='hierarchical')
        # rec.visualize_graph()

        for in_vjp, matvec_param_input in zip(matvec_vjps, A_inputs):
            if cotangents.check(matvec_param_input):
                cotangents.accumulate(matvec_param_input, in_vjp)

class GMRESSolve(MatFreeLinearSolve):
    pass

def solve_gmres(
        A:Union[Variable,Callable],
        b:Variable,
        x0,
        transpose_solve:Callable = None, 
    )->Variable:

    A,b,A_func_bool,b_shape = process_linsolve_A_b(A, b)
    b = b.flatten()
        
    if not A_func_bool: # A is a variable, not a function
        A_func = lambda x: A @ x
        if transpose_solve is None:
            linear_transpose = lambda x: A.T() @ x
            transpose_solve = lambda x: solve_gmres(linear_transpose, x)
    else: # A is a function
        A_func = A
    from csdl_alpha.api import get_current_recorder
    recorder = get_current_recorder()
    A_subgraph, A_inputs, Av, v = build_matvec_subgraph(A_func, b, recorder, 'gmres_matvec_subgraph')

    output = GMRESSolve(A_inputs, b, Av, v, A_subgraph, transpose_solve, A_func, x0).finalize_and_return_outputs()
    return return_b(output, b_shape)

class TestGmres(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        compare_values = []

        n = 10
        diags = (np.arange(n)/2.0+0.3)
        A_val, b_val = np.diag(diags)+0.01, np.sin(np.arange(n)**2.0)
        A = csdl.Variable(value = A_val,name = 'A')
        b = csdl.Variable(value = b_val,name= 'b')
        x_real = np.linalg.solve(A_val**2.0, b_val)

        def linear_operator(x):
            return A**2.0 @ x
        
        def linear_transpose_operator(x):
            return (A**2.0).T() @ x

        x_assembled = solve_gmres(
            A = A**2.0,
            b = b,
        )

        # print('real x:      ', x_real)
        # print('assembled x: ', x_assembled.value)

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')

        x_matrix_free = solve_gmres(
            A = linear_operator,
            b = b,
            transpose_solve = lambda x: solve_gmres(linear_transpose_operator, x),
        )

        # x = csdl.solve_linear(A,b)
        compare_values += [csdl_tests.TestingPair(x_assembled, x_real)]
        compare_values += [csdl_tests.TestingPair(x_matrix_free, x_real)]
        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_functionality2(self,):
        self.prep(debug=False)

        import csdl_alpha as csdl
        import numpy as np

        compare_values = []

        n = 2
        diags = (np.arange(n)/2.0+0.3)
        A_val, b_val, theta_val, theta2_val = np.diag(diags), np.sin((np.arange(n)+0.1)**2.0), np.pi, np.pi/2.0
        A = csdl.Variable(value = A_val,name = 'A')
        b = csdl.Variable(value = b_val,name= 'b')
        theta = csdl.Variable(value = theta_val, name= 'theta')
        theta2 = csdl.Variable(value = theta2_val, name= 'theta2')
        x_real = np.linalg.solve(A_val*theta2_val, b_val*theta_val)

        def linear_operator(x):
            return A*theta2 @ x
        
        def linear_transpose_operator(x):
            return (A*theta2).T() @ x

        x_matrix_free = solve_gmres(
            A = linear_operator,
            b = b*theta,
            transpose_solve = lambda x: solve_gmres(linear_transpose_operator, x),
        )

        # x = csdl.solve_linear(A,b)
        # deriv = csdl.derivative(x_matrix_free, theta)
        # print('deriv:', deriv.value)
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')

        # from csdl_alpha.src.operations.derivatives.derivative_utils import verify_derivatives
        # print(x_real)
        # print(x_matrix_free.value)
        # A_val = np.diag(diags)+1e-9
        # A.value = A_val
        # recorder.execute()
        # x_real = np.linalg.solve(A_val, b_val*theta_val)
        # print(x_real)
        # print(x_matrix_free.value)

        # verify_derivatives(x_matrix_free[0], A, step_size=1e-9, raise_on_error=True, backend='inline')
        # recorder.visualize_graph(visualize_style='hierarchical')
        # exit()

        compare_values += [csdl_tests.TestingPair(x_matrix_free, x_real)]
        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_functionality3(self,):
        self.prep(debug=False)

        import csdl_alpha as csdl
        import numpy as np

        compare_values = []

        n = 2
        diags = (np.arange(n)/2.0+0.3)
        diags_val, b_val, theta_val, theta2_val = (diags), np.sin((np.arange(n)+0.1)**2.0), np.pi, np.pi/2.0
        diags = csdl.Variable(value = diags_val, name = 'A_diag')
        b = csdl.Variable(value = b_val,name= 'b')
        theta = csdl.Variable(value = theta_val, name= 'theta')
        scale = 1.10
        x_real = np.linalg.solve(np.diag(diags_val)*theta_val, b_val*theta_val**scale)

        def linear_operator(x):
            return diags*theta*x
        
        def linear_transpose_operator(x):
            return diags*theta*x

        x_matrix_free = solve_gmres(
            A = linear_operator,
            b = b*theta**scale,
            transpose_solve = lambda x: solve_gmres(linear_transpose_operator, x),
        )

        # x = csdl.solve_linear(A,b)
        # deriv = csdl.derivative(x_matrix_free, theta)
        # print('deriv:', deriv.value)
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')

        # from csdl_alpha.src.operations.derivatives.derivative_utils import verify_derivatives
        # print(x_real)
        # print(x_matrix_free.value)
        # A_val = np.diag(diags)+1e-9
        # A.value = A_val
        # recorder.execute()
        # x_real = np.linalg.solve(A_val, b_val*theta_val)
        # print(x_real)
        # print(x_matrix_free.value)

        # verify_derivatives(x_matrix_free[0], A, step_size=1e-9, raise_on_error=True, backend='inline')
        # recorder.visualize_graph(visualize_style='hierarchical')
        # exit()

        compare_values += [csdl_tests.TestingPair(x_matrix_free, x_real)]
        self.run_tests(compare_values = compare_values, verify_derivatives=True)

if __name__ == '__main__':
    test = TestGmres()
    test.overwrite_backend = 'jax'
    # test.overwrite_backend = 'inline'
    test.test_functionality()
    test.test_functionality2()
    test.test_functionality3()
    # test.test_docstring()