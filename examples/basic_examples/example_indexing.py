'''
Indexing example:
'''
import csdl_alpha as csdl
import numpy as np

recorder = csdl.build_new_recorder(inline = True)
recorder.start()

x = csdl.Variable(name = 'x', value = np.arange(1000).reshape(10,10,10))
print('\nindex get scalar')
print(x[0,0,0].shape)
print(x[0,0,0].value)
print('\nindex first dimension')
print(x[0].shape)
print(x[0].value)
print('\nindex coords')
print(x[0:5,[0,1,2,3],[0,1,2,3]].shape)
print(x[0:5,[0,1,2,3],[0,1,2,3]].value)


# test times
import time
# import cProfile
# profiler = cProfile.Profile()
# profiler.enable()

# x = np.arange(1000).reshape(10,10,10)
# ind = csdl.Variable(name = 'i', value = np.array([0]))
start = time.time()
for i in range(10000):
    y = x[0:5,[0,1,2,3],[0,1,2,3]]
    # y = x+1
end = time.time()
print('time to get scalar:', end-start)
# print(y.value)
# profiler.disable()
# profiler.dump_stats('output')