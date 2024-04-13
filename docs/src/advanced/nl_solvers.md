# Nonlinear Solvers

Thus far, our examples have been focused on defining explicit mathematical relationships of the form $y = f(x)$. CSDL also provides an interface to define implicit relationships between variables of the form $f(x,y) = 0$ where $x$ represents parameters (inputs) and $y$ represents states (outputs). Iterative numerical methods (nonlinear solvers) are often used to solve for $y$ given $x$ by perturbing the values of $y$ to drive $r = f(x,y)$ to zero.

In this tutorial, we will show you how to specify this using CSDL's ImplicitVariable and nonlinear solver library.

We will use a simple nonlinear system of equations as our example:

${x_1}^3 = {x_2}(a-x_1)$

${x_1}^2+{x_2}^2 = a^2$

where $a$ is a parameter and $x_1$, $x_2$ are the states. We can rewrite this in the form $r = f(x, y)$:  

$r_1 = {x_2}(a-x_1)- {x_1}^3$

$r_2 = {x_1}^2+{x_2}^2 - a^2$

where $r_1$ and $r_2$ represents the residuals to drive to zero. In order to define this implicit relationship in CSDL, we start by initializing our state variables using CSDL's ImplicitVariable class and the parameter 'a' can  be initialized like any standard CSDL variable. We then compute the residuals explicitly from the state variables and parameters using CSDL operations.

```python
import csdl_alpha as csdl
recorder = csdl.Recorder(inline=True)
recorder.start()
x_1 = csdl.ImplicitVariable(name='x1', value=0.1) # States must be ImplicitVariable instances
x_2 = csdl.ImplicitVariable(name='x2', value=0.1)

a = csdl.Variable(name='a', value=1.0) # Inputs can be a standard CSDL Variable

# Compute residuals we want to drive to zero
residual_1 = (y**2.0)*(a - x) - x**3.0
residual_2 = x**2.0 + y**2.0 - a*a
```

At this point in the code, we have defined the states and residuals but haven't specified any coupling between them. We can do this by using CSDL's nonlinear solvers. In this example, we initialize a GaussSeidel solver and give the solver a name 'nlsolver_x1_x2'. Finally, we can assign states to residuals using the the nonlinear solver's add_state method.

```python
solver = csdl.nonlinear_solvers.GaussSeidel('nlsolver_x1_x2')
solver.add_state(x_1, residual_1) # Specify that the residual of the state x_1 is residual_1 
solver.add_state(x_2, residual_2) # Specify that the residual of the state x_2 is residual_2
```
Once the states and residual pairs have been assigned to a solver, call the run method to finalize the implicit operation. This step builds an implicit operation node in the graph by analyzing the computational graph built *thus far* (this is important for later). If ```inline = True``` was set when building the recorder, this is the point when the nonlinear solver is ran.
```Python
solver.run()

# The solved states
print(x_1.value)
print(x_2.value)

# The solved residuals
print(residual_1.value)
print(residual_2.value)
```

It can be assumed that any implicit variables (```x_1``` and ```x_2```) represent the solved state at any point of the code and can be used like any other variable.

## Nonlinear Solver Hierarchies

More complicated models often require multiple nonlinear solvers. The structure of the nonlinear solvers and the interfacing between each other  can make a significant impact on the performance of the model evaluation.
