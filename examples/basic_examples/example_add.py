if __name__ == '__main__':
    import csdl_alpha as csdl
    import numpy as np


    import time as time
    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()

    start = time.time()
    recorder = csdl.build_new_recorder(inline = True)
    recorder.start()
    x = csdl.Variable(name = 'x', value = 3.0)
    z = csdl.Variable(name = 'z', value = 2.0)

    for i in range(10):
        csdl.enter_namespace('test'+str(i))
        z = csdl.add(x,z)
        z.name = f'z_{i}'
    for i in range(10):
        csdl.exit_namespace()

    print(z.value)

    recorder.active_graph.visualize()
    # recorder.active_graph.visualize_n2()

    recorder.stop()
    # profiler.disable()
    # profiler.dump_stats('output')

    end = time.time()
    print(f'Time: {end - start}')