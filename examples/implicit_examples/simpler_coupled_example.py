'''
Simpler Coupled Example:
'''

import csdl_alpha as csdl
import numpy as np

import time as time
start = time.time()
recorder = csdl.build_new_recorder(inline = True)
recorder.start()


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
solver = csdl.GaussSeidel('gs_x_simpler')
solver.add_state(x, y)
solver.run()

print('x Value:', x.value)
print('res Value:', y.value)
print('param Value:', a.value)
print('b Value:', b.value)
print('c Value:', c.value)
print('sum states:', sum_states.value)


# A solution: 
x_sol = (np.sqrt(5)-1)/2

# recorder.active_graph.visualize('FINAL_simpler_top_level')
