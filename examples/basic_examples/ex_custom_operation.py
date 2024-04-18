'''
Custom Explicit Operation:
'''
if __name__ == '__main__':

    import csdl_alpha as csdl

    # custom paraboloid model
    class Paraboloid(csdl.CustomExplicitOperation):
        def __init__(self, a, b, c, return_g=False):
            """
            Paraboloid function implemented as a custom explicit operation.

            Parameters
            ----------
            a : float or int
                The value of parameter 'a'.
            b : float or int
                The value of parameter 'b'.
            c : float or int
                The value of parameter 'c'.
            return_g : bool, optional
                Specifies whether to return the value of 'g', by default False.
            """
            super().__init__()
            
            # define any checks for the parameters
            csdl.check_parameter(a, 'a', types=(float, int))
            csdl.check_parameter(b, 'b', types=(float, int))
            csdl.check_parameter(c, 'c', types=(float, int))
            csdl.check_parameter(return_g, 'return_g', types=bool)
            
            # assign parameters to the class
            self.a = a
            self.b = b
            self.c = c
            self.return_g = return_g

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

    print(f.value) # should be 8
    print(g.value) # should be 0 

    recorder.stop()