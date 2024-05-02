'''
Sellar example:
'''

import csdl_alpha as csdl
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()

# Sellar problem from OpenMDAO: 
# https://openmdao.org/newdocs/versions/latest/basic_user_guide/multidisciplinary_optimization/sellar.html 

# Define variables: using openmdao solved optimization values
z1 = csdl.Variable(name = 'z1', value = 1.97763888)
z2 = csdl.Variable(name = 'z2', value = 8.83056605e-15)
x = csdl.Variable(name = 'x', value = 0.0)
y2 = csdl.ImplicitVariable(name = 'y2', value = 1.0)

# Define each "component" from the example
with csdl.namespace('Discipline 1'):
    y1 = z1**2 + z2 + x - 0.2*y2
    y1.add_name('y1')

with csdl.namespace('Discipline 2'):
    residual = csdl.sqrt(y1) + z1 + z2 - y2
    residual.add_name('residual')

with csdl.namespace('Objective'):
    f = x**2 + z2 + y1 + csdl.exp(-y2)
    f.add_name('f')

with csdl.namespace('Constraint 1'):
    g1 = 3.16 - y1
    g1.add_name('g1')

with csdl.namespace('Constraint 2'):
    g2 = y2 - 24.0
    g2.add_name('g2')

# Specifiy coupling
with csdl.namespace('Couple'):
    solver = csdl.nonlinear_solvers.Newton()
    solver.add_state(y2, residual, tolerance=1e-8)
    solver.run()

# Verify values with OpenMDAO documentation:
# https://openmdao.org/newdocs/versions/latest/basic_user_guide/reading_recording/basic_recording_example.html 
print('y1:  ',  y1.value)
print('y2:  ',  y2.value)
print('f:   ',  f.value)
print('g1:  ',  g1.value)
print('g2:  ',  g2.value)
print('z1:  ',  z1.value)
print('z2:  ',  z2.value)

# Visualize Graph
recorder.stop()
recorder.visualize_graph('sellar')

assert np.isclose(y1.value, 3.15999999)
assert np.isclose(y2.value, 3.75527776)
assert np.isclose(f.value, 3.18339394)
assert np.isclose(g1.value, 1.32573978e-08)
assert np.isclose(g2.value, -20.24472224)
assert np.isclose(z1.value, 1.97763888)
assert np.isclose(z2.value, 8.83056605e-15)