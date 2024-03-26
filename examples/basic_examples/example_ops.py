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
    x = csdl.Variable((1,), name = 'x', value = np.ones((1,))*3.0)
    z = csdl.Variable((1,), name = 'z', value = np.ones((1,))*2.0)

    # for i in range(5_000_000):
    for i in range(10):
        z = x-csdl.square(z)
        z.name = f'z_{i}'

    print(z.value)

    # post processing analysis
    # recorder.active_graph.visualize()
    # recorder.stop()
    # profiler.disable()
    # profiler.dump_stats('output')

    end = time.time()
    print(f'Time: {end - start}')