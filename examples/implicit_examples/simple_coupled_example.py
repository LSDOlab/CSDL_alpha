

if __name__ == '__main__':
    import csdl_alpha as csdl
    import numpy as np

    import time as time
    start = time.time()
    recorder = csdl.build_new_recorder(inline = True)
    recorder.start()

    # x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.6180339)
    # y = csdl.ImplicitVariable(shape=(1,), name='y', value=0.7861513777)
    x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.61803)
    y = csdl.ImplicitVariable(shape=(1,), name='y', value=0.78)
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
    # ONE SOLVER COUPLING:
    solver = csdl.GaussSeidel('gs_xy')
    solver.add_state(x, residual_1)
    solver.add_state(y, residual_2)
    solver.run()

    # NESTED (x) SOLVER COUPLING:
    # solver = csdl.GaussSeidel('gs_x')
    # solver.add_state(x, residual_1, initial_value=0.0)
    # solver.run()

    # solver = csdl.GaussSeidel('gs_y')
    # solver.add_state(y, residual_2, initial_value=0.0)
    # solver.run()

    # NESTED (y) SOLVER COUPLING:
    # solver = csdl.GaussSeidel('gs_y')
    # solver.add_state(y, residual_2, initial_value=0.0)
    # solver.run()

    # solver = csdl.GaussSeidel('gs_x')
    # solver.add_state(x, residual_1, initial_value=0.0)
    # solver.run()


    print('x Value:', x.value)
    print('y Value:', y.value)
    print('param Value:', param.value)

    # A solution: 
    x_sol = (np.sqrt(5)-1)/2
    y_sol = np.sqrt((-1+np.sqrt(5))/2)
    print('an x solution:', x_sol)
    print('a y solution:', y_sol)

    recorder.active_graph.visualize('FINAL_top_level')
