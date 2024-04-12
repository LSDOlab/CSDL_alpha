# Custom Operation

CSDL provides a custom operation class that allows non-csdl models to be added to the graph. This is useful for adding custom operations that are not provided by CSDL, or for adding operations that are not implemented in CSDL.

A full example is shown below, after which we will break down each part of the script.

```python
import csdl_alpha as csdl

# custom paraboloid model
class Paraboloid(csdl.CustomExplicitOperation):
    def initialize(self):
        self.a = self.parameters.declare('a')
        self.b = self.parameters.declare('b')
        self.c = self.parameters.declare('c')
        self.return_g = self.parameters.declare('return_g', default=False)

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('x', inputs.x)
        self.declare_input('y', inputs.y)
        self.declare_input('z', inputs.z)

        # declare output variables
        f = self.create_output('f', inputs.x.shape)

        # declare any derivative parameters
        self.declare_derivative_parameters('f', 'z', dependent=False)

        # construct output of the model
        output = csdl.VariableGroup()
        output.f = f

        if self.return_g:
            g = self.create_output('g', inputs.x.shape)
            output.g = g

        return output
    
    def compute(self, input_vals, output_vals):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']

        output_vals['f'] = (x - self.a)**2 + x * y + (y + self.b)**2 - self.c

        if self.return_g:
            output_vals['g'] = output_vals['f']*z

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']

        derivatives['f', 'x'] = 2*x - self.a + y
        derivatives['f', 'y'] = 2*y + x + self.b

        if self.return_g:
            derivatives['g', 'x'] = z*derivatives['f', 'x']
            derivatives['g', 'y'] = z*derivatives['f', 'x']
            derivatives['g', 'z'] = outputs_vals['f']


recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

inputs.x = csdl.Variable(value=0.0, name='x')
inputs.y = csdl.Variable(value=0.0, name='y')
inputs.z = csdl.Variable(value=0.0, name='z')

paraboloid = Paraboloid(a=2, b=4, c=12, return_g=True)
outputs = paraboloid.evaluate(inputs)

f = outputs.f
g = outputs.g
 

recorder.stop()
```


## initialize

The initialize method is used to declare the parameters of the custom operation. The `initialize` method is called when the custom operation is created, and can be used to take any non-csdl arguments that modify the operation. Input parameters are checked for any criteria given when declared.

```python
def initialize(self):
    self.a = self.parameters.declare('a')
    self.b = self.parameters.declare('b')
    self.c = self.parameters.declare('c')
    self.return_g = self.parameters.declare('return_g', default=False)
```

## evaluate

The evaluate method defines the inputs and outputs of the operation, as well as any derivative parameters. The arguments to the evaluate method can be Variables, VariableGroups, or any other object. However, every `Variable` input to the model must be declared and assigned a string using the `declare_input()` method. Similarly, the `create_output()` method is used to create the output variables of the model, and assign them a string. These outputs can be optionally packaged into a `VariableGroup` object or any other object, and returned from the evaluate method.

The `declare_derivative_parameters()` method is used to declare any derivative parameters of the model. By default CSDL assumes derivatives will be provided between each output and each input, but this can be modified by setting the `dependent` argument to `True` or `False`.

```python
def evaluate(self, inputs: csdl.VariableGroup):
    # assign method inputs to input dictionary
    self.declare_input('x', inputs.x)
    self.declare_input('y', inputs.y)
    self.declare_input('z', inputs.z)

    # declare output variables
    f = self.create_output('f', inputs.x.shape)

    # declare any derivative parameters
    self.declare_derivative_parameters('f', 'z', dependent=False)

    # construct output of the model
    output = csdl.VariableGroup()
    output.f = f

    if self.return_g:
        g = self.create_output('g', inputs.x.shape)
        output.g = g

    return output
```


## compute

The compute method is used to calculate the output of the model. The `input_vals` dictionary contains the values of the input variables as numpy arrays, and the `output_vals` dictionary is used to store the output values. The `compute` method should assign the output values to the `output_vals` dictionary. The keys of these dictionaries correspond to the strings given in the `declare_input()` and `create_output()` methods.

```python
def compute(self, input_vals, output_vals):
    x = input_vals['x']
    y = input_vals['y']
    z = input_vals['z']

    output_vals['f'] = (x - self.a)**2 + x * y + (y + self.b)**2 - self.c

    if self.return_g:
        output_vals['g'] = output_vals['f']*z
```

## compute_derivatives

The `compute_derivatives` method is used to calculate the derivatives of the output variables with respect to the input variables. The `input_vals` dictionary contains the values of the input variables, the `output_vals` dictionary contains the values of the output variables, and the `derivatives` dictionary is used to store the derivative values. The keys of the `derivatives` dictionary are tuples of the form `(output_name, input_name)`, where `output_name` and `input_name` are the strings given in the `create_output()` and `declare_input()` methods.

```python
def compute_derivatives(self, input_vals, outputs_vals, derivatives):
    x = input_vals['x']
    y = input_vals['y']
    z = input_vals['z']

    derivatives['f', 'x'] = 2*x - self.a + y
    derivatives['f', 'y'] = 2*y + x + self.b

    if self.return_g:
        derivatives['g', 'x'] = z*derivatives['f', 'x']
        derivatives['g', 'y'] = z*derivatives['f', 'x']
        derivatives['g', 'z'] = outputs_vals['f']
```

## Using the custom operation

The custom operation can be used by first constructing the operation, then evaluating it. The custom operation is created by calling the constructor of the custom operation class and passing in any parameters as keyword arguments. It is then run by calling the `evaluate` method, passing in any inputs and receiving the outputs.

```python
recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

inputs.x = csdl.Variable(value=0.0, name='x')
inputs.y = csdl.Variable(value=0.0, name='y')
inputs.z = csdl.Variable(value=0.0, name='z')

paraboloid = Paraboloid(a=2, b=4, c=12, return_g=True)
outputs = paraboloid.evaluate(inputs)

f = outputs.f
g = outputs.g

recorder.stop()
```
