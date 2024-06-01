import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
import csdl_alpha as csdl
import numpy as np

class TestImplicit(csdl_tests.CSDLTest):

    def test_very_simple(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(name = 'a', value = 1.5)
        x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.34)

        y = x - a
        y.name = 'residual_x'

        # sum of solved states
        sum_states = x+x
        sum_states.name = 'state_sum'

        # apply coupling:
        # ONE SOLVER COUPLING:
        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x_simpler')
        solver.add_state(x, y)
        solver.run()

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(y, np.array([0.0]), tag = 'residual', decimal = 9),
                csdl_tests.TestingPair(x, np.array([1.5]), tag = 'state', decimal = 7),
                csdl_tests.TestingPair(a, np.array([1.5]), tag = 'a'),
                csdl_tests.TestingPair(sum_states, np.array([3.0]), tag = 'ax2', decimal=7),
            ],
            verify_derivatives=True
        )

    def test_value_newton(self,):
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
        solver = csdl.nonlinear_solvers.Newton('gs_x_simpler')
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
            verify_derivatives=True
        )        

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
        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x_simpler')
        solver.add_state(x, y)
        current_graph = csdl.get_current_recorder().active_graph
        num_nodes_right_now = len(current_graph.node_table)
        solver.run()
        num_nodes_after_run = len(current_graph.node_table)
        # assert num_nodes_after_run < num_nodes_right_now

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph()

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(y, np.array([0.0]), tag = 'residual', decimal = 9),
                csdl_tests.TestingPair(x, np.array([0.38742589]), tag = 'state', decimal = 7),
                csdl_tests.TestingPair(a, np.array([1.5]), tag = 'a'),
                csdl_tests.TestingPair(b, np.array([0.5]), tag = 'b'),
                csdl_tests.TestingPair(c, np.array([-1.0]), tag = 'c'),
                csdl_tests.TestingPair(ax2, np.array([1.5*0.38742589**2]), tag = 'ax2', decimal=7),
            ],
            verify_derivatives=True
        )

    def test_double_state_nest1(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.1)
        y = csdl.ImplicitVariable(shape=(1,), name='y', value=0.1)
        param = csdl.Variable(shape=(1,), name='param', value=np.ones((1,))*1.0)
        test = param+param # should be ignored
        test.name = 'ignore'
        
        # simple 2d root finding problem: https://balitsky.com/teaching/phys420/Nm4_roots.pdf
        residual_1 = csdl.square(y)*(param - x) - x*x*x
        residual_2 = csdl.square(x) + csdl.square(y) - param*param

        residual_1.name = 'residual_1'
        residual_2.name = 'residual_2'

        # sum of solved states
        sum_states = x + y
        sum_states.name = 'states_sum'

        # apply coupling:
        x_update = x-residual_1/(-csdl.square(y)-3.0*x*x)
        y_update = y-residual_2/(2.0*y)

        # NESTED (x) SOLVER COUPLING:
        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x')
        solver.add_state(x, residual_1, state_update=x_update)

        # Make sure nodes are deleted
        current_graph = csdl.get_current_recorder().active_graph
        num_nodes_right_now = len(current_graph.node_table)
        solver.run()
        num_nodes_after_run = len(current_graph.node_table)
        # assert num_nodes_after_run < num_nodes_right_now

        solver = csdl.nonlinear_solvers.GaussSeidel('gs_y')
        solver.add_state(y, residual_2, state_update=y_update)

        # Make sure nodes are deleted
        current_graph = csdl.get_current_recorder().active_graph
        num_nodes_right_now = len(current_graph.node_table)
        solver.run()
        num_nodes_after_run = len(current_graph.node_table)
        # assert num_nodes_after_run < num_nodes_right_now

        x_sol = np.array([(np.sqrt(5)-1)/2])
        y_sol = np.array([np.sqrt((-1+np.sqrt(5))/2)])

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(y, y_sol, tag = 'state_y', decimal = 11),
                csdl_tests.TestingPair(x, x_sol, tag = 'state_x', decimal = 11),
            ],
            verify_derivatives=True
        )

    def test_double_state_nest2(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.1)
        y = csdl.ImplicitVariable(shape=(1,), name='y', value=0.1)
        param = csdl.Variable(shape=(1,), name='param', value=np.ones((1,))*1.0)
        test = param+param # should be ignored
        test.name = 'ignore'
        
        # simple 2d root finding problem: https://balitsky.com/teaching/phys420/Nm4_roots.pdf
        residual_1 = csdl.square(y)*(param - x) - x*x*x
        residual_2 = csdl.square(x) + csdl.square(y) - param*param

        residual_1.name = 'residual_1'
        residual_2.name = 'residual_2'

        # sum of solved states
        sum_states = x + y
        sum_states.name = 'states_sum'

        # apply coupling:
        x_update = x-residual_1/(-csdl.square(y)-3.0*x*x)
        y_update = y-residual_2/(2.0*y)

        # NESTED (x) SOLVER COUPLING:
        solver = csdl.nonlinear_solvers.GaussSeidel('gs_y')
        solver.add_state(y, residual_2, state_update=y_update)

        # Make sure nodes are deleted
        current_graph = csdl.get_current_recorder().active_graph
        num_nodes_right_now = len(current_graph.node_table)
        solver.run()
        num_nodes_after_run = len(current_graph.node_table)
        # assert num_nodes_after_run < num_nodes_right_now

        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x')
        solver.add_state(x, residual_1, state_update=x_update)
        
        num_nodes_right_now = len(current_graph.node_table)
        solver.run()
        num_nodes_after_run = len(current_graph.node_table)
        # assert num_nodes_after_run < num_nodes_right_now
        x_sol = np.array([(np.sqrt(5)-1)/2])
        y_sol = np.array([np.sqrt((-1+np.sqrt(5))/2)])

        with csdl.namespace('total_deriv'):
            deriv = csdl.derivative(sum_states, [param])

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(y, y_sol, tag = 'state_y', decimal = 9),
                csdl_tests.TestingPair(x, x_sol, tag = 'state_x', decimal = 9),
                csdl_tests.TestingPair(deriv[param], deriv[param].value, tag = 'deriv', decimal = 9),
            ],
            verify_derivatives=True
        )

    def test_double_state(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.1)
        y = csdl.ImplicitVariable(shape=(1,), name='y', value=0.1)
        param = csdl.Variable(shape=(1,), name='param', value=np.ones((1,))*1.0)
        test = param+param # should be ignored
        test2 = param*param
        test2.add_name('test2')
        test.name = 'ignore'
        
        # simple 2d root finding problem: https://balitsky.com/teaching/phys420/Nm4_roots.pdf
        temp = y**2.0
        temp.add_name('temp')
        temp2 = x**2.0 + temp
        temp2.add_name('temp2')

        residual_1 = (temp)*(param - x) - x*x*x
        residual_2 = temp2 - test2

        residual_1.name = 'residual_1'
        residual_2.name = 'residual_2'

        # sum of solved states
        sum_states = x + y*temp + temp2
        sum_states.name = 'states_sum'

        # apply coupling:
        x_update = x-residual_1/(-(y**2.0)-3.0*x*x)
        y_update = y-residual_2/(2.0*y)

        # NESTED (x) SOLVER COUPLING:
        solver = csdl.nonlinear_solvers.GaussSeidel('gs_xy')
        solver.add_state(y, residual_2, state_update=y_update)
        solver.add_state(x, residual_1, state_update=x_update)
        
        # Make sure nodes are deleted
        current_graph = csdl.get_current_recorder().active_graph
        num_nodes_right_now = len(current_graph.node_table)
        solver.run()
        num_nodes_after_run = len(current_graph.node_table)
        # assert num_nodes_after_run < num_nodes_right_now

        x_sol = np.array([(np.sqrt(5)-1)/2])
        y_sol = np.array([np.sqrt((-1+np.sqrt(5))/2)])

        with csdl.namespace('total_deriv'):
            deriv = csdl.derivative(sum_states, [param])

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(y, y_sol, tag = 'state_y', decimal = 9),
                csdl_tests.TestingPair(x, x_sol, tag = 'state_x', decimal = 9),
                csdl_tests.TestingPair(deriv[param], deriv[param].value, tag = 'deriv', decimal = 9),
            ],
            verify_derivatives=True
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
        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x_simpler')

        # state must be CSDL implicit variable
        with pytest.raises(TypeError) as e_info:
            solver.add_state(1.0*c, y)
        
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

        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x_simpler')
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

        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x_simpler')
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

        solver = csdl.nonlinear_solvers.GaussSeidel('gs_x_simpler')
        solver.add_state(x1, y1)
        solver.add_state(x2, y2)

        # x1 must affect atleast one residual
        with pytest.raises(ValueError) as e_info:
            solver.run()

    def test_linear_system(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        # solver = csdl.nonlinear_solvers.Newton()
        # x = csdl.ImplicitVariable((1,), value=0.34)
        # y = x*2.0+1.0
        # solver.add_state(x, y)
        # solver.run()

        solver = csdl.nonlinear_solvers.Newton()
        x = csdl.ImplicitVariable((1,), value=0.34, name = 'x')
        x1 = csdl.ImplicitVariable((2,), value=0.34, name = 'x1')
        y = x+1.0
        y.add_name('residual')
        y1 = x1*2.0+1.0
        y1.add_name('residual2')
        solver.add_state(x, y)
        solver.add_state(x1, y1)
        solver.run()

        compare_values = []
        compare_values += [csdl_tests.TestingPair(y, np.array([0.0]), tag = 'residual')]
        compare_values += [csdl_tests.TestingPair(x, np.array([-1.0]), tag = 'state')]
        compare_values += [csdl_tests.TestingPair(y1, np.array([0.0, 0.0]), tag = 'residual2')]
        compare_values += [csdl_tests.TestingPair(x1, np.array([-0.5, -0.5]), tag = 'state2')]
        self.run_tests(
            compare_values = compare_values,
            verify_derivatives=True
        )

if __name__ == '__main__':
    t = TestImplicit()
    t.test_arg_errors()
    t.test_insufficient_res_state_dependence_1()
    t.test_insufficient_res_state_dependence_2()
    
    t.test_double_state_nest1()
    t.test_double_state()
    t.test_double_state_nest2()
    t.test_values()
    t.test_value_newton()

    t.test_very_simple()

