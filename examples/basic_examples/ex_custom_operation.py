'''
Custom Explicit Operation:
'''
if __name__ == '__main__':

    import csdl_alpha as csdl

    # custom model
    class Paraboloid(csdl.CustomExplicitModel):
        def initialize(self):
            self.a = self.parameters.declare('a')
            self.b = self.parameters.declare('b')
            self.c = self.parameters.declare('c')
            self.return_g = self.parameters.declare('return_g', default=False)

        def evaluate(self, x, y, z):
            # assign method inputs to input dictionary
            self.inputs['x'] = x
            self.inputs['y'] = y
            self.inputs['z'] = z

            # TODO: consider self.set_input('x', x)

            # declare output variables
            f = self.declare_output('f', x.shape)

            # declare any derivative parameters
            self.declare_derivative_parameters('f', 'z', dependent=False)

            # construct output of the model
            output = csdl.VariableGroup()
            output['f'] = f

            if self.return_g:
                g = self.declare_output('g', x.shape)
                output['g'] = g

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


    recroder = csdl.Recorder(inline=True)
    recroder.start()

    x = csdl.Variable(value=0.0, name='x')
    y = csdl.Variable(value=0.0, name='y')
    z = csdl.Variable(value=0.0, name='z')

    paraboloid = Paraboloid(a=2, b=4, c=12, return_g=True)
    outputs = paraboloid.evaluate(x, y, z)

    f = outputs['f']
    g = outputs['g']

    print(f.value) # should be 0
    print(g.value) # should be 0 

    recroder.stop()