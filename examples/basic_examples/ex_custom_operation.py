'''
Custom Explicit Operation:
'''
if __name__ == '__main__':

    import csdl_alpha as csdl

    '''
    # custom model
    class Paraboloid(csdl.CusomExplicitModel):
        def initialize(self):
            self.a = self.parameters.declare('a')
            self.b = self.parameters.declare('b')
            self.c = self.parameters.declare('c')
            self.return_g = self.parameters.declare('return_g', value=False)

        def declare_options(self, x, y):
            outputs = self.outputs # use to specify information abount the outputs (eg, shape, dense/sparse, etc) (required for every output)
            derivatives = self.derivatives # use to specify information abount the derivatives (eg, shape, dense/sparse, etc) (optional?)
            
            self.assert_shape(x, (1,))
            self.assert_shape(y, (1,))

            outputs.declare('f', shape=(1,))

        def declare_parameters(self, x, y):

            if x.shape != y.shape:
                raise Exception('x and y must have the same shape')
            shape = x.shape

            f = self.create_output(shape=shape)
            # these are the defaults
            self.declare_derivative_parameters(
                f, x,
                dependent = True,
                rows=None, cols=None,
                val=None,
                method='exact', step=None, form=None, step_calc=None
                )
            
            if self.return_g:
                g = self.create_output(shape=shape)
                self.declare_derivative_parameters(
                    g, x,
                    dependent = True,
                    rows=None, cols=None,
                    val=None,
                    method='exact', step=None, form=None, step_calc=None
                    )
                
                return f, g
            else:
            
        return f
                

        def declare_parameters(self, x, y):

            if x.shape != y.shape:
                raise Exception('x and y must have the same shape')
            shape = x.shape

            f = self.declare_output('f', shape=shape)

            self.declare_derivative_parameters(
                f, x,
                dependent = True,
                rows=None, cols=None,
                val=None,
                method='exact', step=None, form=None, step_calc=None
                )
            
            if self.return_g:
                g = self.dinitializeeclare_output('g', shape=shape)
                self.declare_derivative_parameters(g, x, dependent=False)
                self.declare_derivative_parameters(g, y, dependent=False)
        
        def evaluate_derivatives(self, x, y):
            f = self.output_values['f']
            g = self.output_values['g']
            self.derivatives[f, x] = 2*x - self.a + y
            self.derivatives[g, y] = f

        def evaluate(self, x, y):
            # assign method inputs to input dictionary
            self.inputs['x'] = x
            self.inputs['y'] = y

            # declare output variables
            f = self.declare_output('f', x.shape)

            # declare any derivative parameters
            self.declare_derivative_parameters('f', 'x', dependent=True)

            # construct output of the model
            output = csdl.VariableGroup()
            output['f'] = f

            if self.return_g:
                g = self.declare_output('g', x.shape)
                output['g'] = g

            return output
        
        def compute(self, inputs, outputs):
            x = inputs['x']
            y = inputs['y']

            outputs['f'] = (x - self.a)**2 + x * y + (y + self.b)**2 - self.c

            if self.return_g:
                outputs['g'] = outputs['f']*y

        def compute_derivatives(self, inputs, outputs, derivatives):
            x = inputs['x']
            y = inputs['y']

            derivatives['f', 'x'] = 2*x - self.a + y
            derivatives['f', 'y'] = 2*y + x + self.b

            if self.return_g:
                derivatives['g', 'x'] = outputs['f']
                derivatives['g', 'y'] = outputs['f']*y




        def evaluate(self, x, y):
            f = (x - self.a)**2 + x * y + (y + self.b)**2 - self.c

            self.outputs['f'] = f
            if self.return_g:
                self.outputs['g'] = f*y

        def evaluate_derivative(self, x, y):

            f = self.evaluate(x, y)

            self.derivatives[f, x] = 2*x - self.a + y
            self.derivatives[f, y] = 2*y + x + self.b




        def create_outputs(self, x, y):
            
            # f_val = self.evaluate(x, y)
            
            f = csdl.Variable(shape=x.shape)

            return f





        def compute(self, inputs, outputs):
            x = inputs['x']
            y = inputs['y']

            outputs['f_xy'] = (x - self.a)**2 + x * y + (y + self.b)**2 - self.c

        def compute_derivatives(self, inputs, derivatives):
        
            derivatives['f_xy', 'x'] = 2*inputs['x'] - self.a + inputs['y']
            derivatives['f_xy', 'y'] = 2*inputs['y'] + inputs['x'] + self.b

        def arg_inputs(x, y):
            return {'x': x, 'y': y}

        def package_outputs(self, f):
            dict = {}
            dict['f_xy'] = outputs['f_xy']
            return dict
    '''



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