'''
Operations:
'''
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

    z1 = x-z
    z2 = x+z1
    z3 = x*z2
    z4 = x-csdl.square(z3)

    z1.name = f'z1'
    z2.name = f'z2'
    z3.name = f'z3'
    z4.name = f'z4'

    print(z1.value)
    print(z2.value)
    print(z3.value)
    print(z4.value)

    recorder.stop()

    # post processing analysis
    recorder.active_graph.visualize()
    # profiler.disable()
    # profiler.dump_stats('output')

    end = time.time()
    print(f'Time: {end - start}')


        # for i in range(5_000_000):
    # for i in range(2):
    #     z1 = x-z
    #     z2 = x+z1
    #     z3 = x*z2
    #     z4 = x-csdl.square(z3)

    #     z1.name = f'z1_{i}'
    #     z2.name = f'z2_{i}'
    #     z3.name = f'z3_{i}'
    #     z4.name = f'z4_{i}'