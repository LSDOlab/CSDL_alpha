import csdl_alpha as csdl

# This example demonstrates how to define and solve an simple optimization problem using modopt
# minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

rec = csdl.Recorder()
rec.start()

# initialize design variables
x = csdl.Variable(name = 'x', value=1.)
y = csdl.Variable(name = 'y', value=1.)
x.set_as_design_variable(lower = 0.0, scaler=100.0)
y.set_as_design_variable(scaler=0.2)

z = x**4 + y**4

# compute objective
z.add_name('z')
z.set_as_objective(scaler=3.)

# compute constraints
constraint_1 = x + y
constraint_2 = x - y
constraint_1.add_name('constraint_1')
constraint_2.add_name('constraint_2')
constraint_1.set_as_constraint(lower=1., upper=1., scaler=20)
constraint_2.set_as_constraint(lower=1., scaler=12.)

rec.stop()

# Create a Simulator object from the Recorder object
sim = csdl.experimental.PySimulator(rec)

# Import CSDLAlphaProblem from modopt
from modopt import CSDLAlphaProblem
from modopt import SLSQP

# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLAlphaProblem(problem_name='quartic',simulator=sim)


# Setup your preferred optimizer (here, SLSQP) with the Problem object 
# Pass in the options for your chosen optimizer
optimizer = SLSQP(prob, ftol=1e-6, maxiter=20, outputs=['x'])

# Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0)

# Solve your optimization problem
optimizer.solve()

# Print results of optimization
optimizer.print_results()