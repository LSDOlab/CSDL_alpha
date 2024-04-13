

if __name__ == '__main__':
    import csdl_alpha as csdl
    import numpy as np

    import time as time
    start = time.time()
    recorder = csdl.build_new_recorder(inline = True)
    recorder.start()

    x1 = csdl.ImplicitVariable(name='x1', value=1.5)
    x2 = csdl.ImplicitVariable(name='x2', value=3.5)
    a = csdl.Variable(name='a', value=10.0)

    residual_1 = x1-(csdl.sqrt(a-x1*x2))
    residual_2 = x2-(csdl.sqrt((57-x2)/(3*x1)))
    print('residual_1:', residual_1.value)
    print('residual_2:', residual_2.value)

    residual_1.name = 'residual_1'
    residual_2.name = 'residual_2'

    # sum of solved states
    sum_states = x1 + x2
    sum_states.name = 'states_sum'

    # apply coupling:
    # x_update = x-residual_1/(-csdl.square(y)-3.0*x*x)
    # y_update = y-residual_2/(2.0*y)

    # ONE SOLVER COUPLING:
    # solver = csdl.nonlinear_solvers.GaussSeidel('gs_xy')
    # solver.add_state(x1, residual_1)
    # solver.add_state(x2, residual_2)
    # solver.run()

    # NESTED (x) SOLVER COUPLING:
    solver = csdl.nonlinear_solvers.GaussSeidel('gs_x')
    solver.add_state(x1, residual_1)
    solver.run()

    solver = csdl.nonlinear_solvers.GaussSeidel('gs_y')
    solver.add_state(x2, residual_2)
    solver.run()

    # NESTED (y) SOLVER COUPLING:
    # solver = csdl.nonlinear_solvers.GaussSeidel('gs_y')
    # solver.add_state(y, residual_2, state_update=y_update)
    # solver.run()

    # solver = csdl.nonlinear_solvers.GaussSeidel('gs_x')
    # solver.add_state(x, residual_1, state_update=x_update)
    # solver.run()


    print('x1 Value:', x1.value)
    print('x2 Value:', x2.value)
    print('param Value:', a.value)

    # A solution: 
    x_sol = (np.sqrt(5)-1)/2
    y_sol = np.sqrt((-1+np.sqrt(5))/2)
    print('an x solution:', x_sol)
    print('a y solution:', y_sol)

    recorder.active_graph.visualize('FINAL_top_level')
