'''
Addition:
'''
if __name__ == '__main__':
    import csdl
    import numpy as np


    import time as time
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    m = csdl.Model()
    x = m.create_input(name = 'x', val = 3.0)
    z = m.create_input(name = 'z', val = 2.0)

    start = time.time()
    for i in range(60000):
        z = x+z
    z = m.register_output('z1', z)
    g = csdl.GraphRepresentation(m)

    import python_csdl_backend
    sim = python_csdl_backend.Simulator(g)
    sim.run()
    print(sim['z1'])
    end = time.time()
    print(f'Time: {end - start}')
    profiler.disable()
    profiler.dump_stats('output')
    # for item in z.trace:
    #     print(item)