'''
ODE example:
'''

import csdl_alpha as csdl
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()


# Define the ODE dy/dt = y
def ode(y, t):
    return csdl.sin(y+t)

# Set the initial condition
y0 = csdl.Variable(value = 2*np.arctan((-3))+2*np.pi) #3.7850937623830774
t0 = csdl.Variable(value = 0.0)

# Set the time step size
dt = 0.01

# Set the total time
T = 1.0

# Initialize the state array
y = y0
t = t0

# Loop over the time dimension
for i in csdl.frange(0, int(T/dt)):
    # Apply the Euler method
    y = y + dt * ode(y, t)
    t = t+dt

# Final y value
final_y = y

# Derivative of final y with respect to initial condition
dyfinal_dy0 = csdl.derivative(final_y, y0)

# Second derivative of final y with respect to initial condition
d2yfinal_dy02 = csdl.derivative(dyfinal_dy0, y0)

# Verification:
# analytical y value:
analytical_final_y = 2*np.arctan((-2-T-1)/(T+1.0))+2*np.pi-T
print('analytical final y:    ', analytical_final_y)
print('computed final y:      ', final_y.value[0], end='\n\n')
assert np.isclose(final_y.value[0], analytical_final_y, rtol=1e-2)

# analytical dy/dy0 value:
print('computed dfinal/dy0:   ', None)
print('computed dfinal/dy0:   ', dyfinal_dy0.value[0,0])

# analytical dy/dy0 value:
print('computed d2final/dy02: ', None)
print('computed d2final/dy02: ', d2yfinal_dy02.value[0,0])

recorder.visualize_graph('ode', visualize_style='hierarchical')