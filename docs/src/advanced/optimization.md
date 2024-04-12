# Optimization

This page gives an overview on how to define an optimization problem in CSDL.

## Defining an Optimization Problem

To define an optimization problem, you must designate variables within your model as design variables, constraints, and objectives. These are done by calling the `set_as_design_variable()`, `set_as_constraint()`, and `set_as_objective()` methods on the variables, respectively. 

## Running an Optimization

This is handled by the backend, and has no inline equivalent.