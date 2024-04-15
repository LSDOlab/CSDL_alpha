'''
Simple model:
'''
import csdl_alpha as csdl
import numpy as np

# Start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# make variables
x = csdl.Variable(value=0)
y = csdl.Variable(value=0, name='y')

# define model
f = (x - 3)**2 + x*y + (y + 4)**2 - 3

# finish up
recorder.stop()
print(f.value)
recorder.active_graph.visualize()
