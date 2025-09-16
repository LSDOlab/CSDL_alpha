import csdl_alpha as csdl
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
 

# custom paraboloid model
class ParaboloidJVP(csdl.CustomExplicitOperation):
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

    def evaluate(self, x, y, z):
        # assign method _dict to input dictionary

        self.declare_input('x', x)
        self.declare_input('y', y)
        self.declare_input('z', z)

        # declare output variables
        f = self.create_output('f', x.shape)

        # declare any derivative parameters
        self.declare_derivative_parameters('f', 'z', dependent=False)

        # construct output of the model
        output = csdl.VariableGroup()
        output.f = f

        if self.return_g:
            g = self.create_output('g', x.shape)
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

        derivatives['f', 'x'] = 2*x - 2*self.a + y
        derivatives['f', 'y'] = 2*y + x + 2*self.b

        if self.return_g:
            derivatives['g', 'x'] = z*derivatives['f', 'x']
            derivatives['g', 'y'] = z*derivatives['f', 'y']
            derivatives['g', 'z'] = outputs_vals['f']

    def compute_jacvec_product(self, input_vals, outputs_vals, d_inputs, d_outputs, mode):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']

        if mode == 'rev':
            d_inputs['x'] = (2*x - 2*self.a + y)*d_outputs['f']
            d_inputs['y'] = (2*y + x + 2*self.b)*d_outputs['f']
            if self.return_g:
                d_inputs['x'] += z*(2*x - 2*self.a + y)*d_outputs['g']
                d_inputs['y'] += z*(2*y + x + 2*self.b)*d_outputs['g']
                d_inputs['z'] = outputs_vals['f']*d_outputs['g']

class TestCustom(csdl_tests.CSDLTest):
    def test_simple_deriv(self):
        self.prep(always_build_inline = True)
 
        import numpy as np
 
        x = csdl.Variable(value=1, name='x')
        y = csdl.Variable(value=2, name='y')
        z = csdl.Variable(value=3, name='z')
 
        paraboloid = ParaboloidJVP(a=2, b=4, c=12, return_g=True)
        outputs = paraboloid.evaluate(x, y, z)
 
        f = outputs.f
        g = outputs.g

        f.add_name('f')
        g.add_name('g')
        derivs = csdl.derivative([f, g], [x, y, z])

        for deriv in derivs:
            print(deriv, derivs[deriv].value)

        # f = (x - 2)**2 + x * y + (y + 4)**2 - 12
        # f = x**2 - 4*x + 4 + x*y + y**2 + 8*y + 16 - 12
        # f = x**2 - 4*x + 4 + x*y + y**2 + 8*y + 4
        # df_dx = 2*x - 4 + y
        # df_dy = x + 2*y + 8
        # df_dz = 0.0

        # g = f*z

        df_dx = derivs[f, x]
        df_dx_real = 2*x.value - 4 + y.value
        assert np.isclose(df_dx.value[0], df_dx_real)

        df_dy = derivs[f, y]
        df_dy_real = x.value + 2*y.value + 8
        assert np.isclose(df_dy.value[0], df_dy_real)

        df_dz = derivs[f, z]
        assert np.isclose(df_dz.value[0], 0.0)

        dg_dx = derivs[g, x]
        dg_dx_real = df_dx_real*z.value
        assert np.isclose(dg_dx.value[0], dg_dx_real)

        dg_dy = derivs[g, y]
        dg_dy_real = df_dy_real*z.value
        assert np.isclose(dg_dy.value[0], dg_dy_real)

        dg_dz = derivs[g, z]
        dg_dz_real = f.value
        assert np.isclose(dg_dz.value[0], dg_dz_real)

        self.run_tests(
            compare_values = [csdl_tests.TestingPair(f, f.value), csdl_tests.TestingPair(g, g.value)],
            verify_derivatives=True,
        )

    def test_derivs(self):
        self.prep(always_build_inline = True)
 
        import numpy as np
 
        class Custom2(csdl.CustomExplicitOperation):
            def __init__(self, a, b, c):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def evaluate(self, x, y, z):
                # assign method _dict to input dictionary

                self.declare_input('x', x)
                self.declare_input('y', y)
                self.declare_input('z', z)

                # declare output variables
                f = self.create_output('f', x.shape)
                g = self.create_output('g', x.shape)

                # declare any derivative parameters
                self.rows_cols = np.arange(x.size)
                self.declare_derivative_parameters('f', 'z', dependent=False)
                self.declare_derivative_parameters('f', 'y', rows = self.rows_cols, cols = self.rows_cols)
                self.declare_derivative_parameters('f', 'x', sparse = True)

                self.declare_derivative_parameters('g', 'z', rows = self.rows_cols, cols = self.rows_cols, val = 3.0*np.ones(x.size))

                return f, g
            
            def compute(self, input_vals, output_vals):
                x = input_vals['x']
                y = input_vals['y']
                z = input_vals['z']

                output_vals['f'] = self.a*y + x*2.0

                output_vals['g'] = 3.0*z + x*y

            # def compute_derivatives(self, input_vals, outputs_vals, derivatives):
            #     x = input_vals['x']
            #     y = input_vals['y']
            #     z = input_vals['z']
            #     import scipy.sparse as sp
            #     derivatives['f', 'x'] = sp.csc_matrix((2.0*np.ones(x.size), (np.arange(x.size), np.arange(x.size))), shape=(x.size, x.size))
            #     derivatives['f', 'y'] = self.a*np.ones(x.size)

            #     derivatives['g', 'x'] = np.diag(y.flatten())
            #     derivatives['g', 'y'] = np.diag(x.flatten())

            def compute_jacvec_product(self, input_vals, outputs_vals, d_inputs, d_outputs, mode):
                x = input_vals['x']
                y = input_vals['y']
                z = input_vals['z']

                if mode == 'rev':
                    d_inputs['x'] = 2.0*d_outputs['f']
                    d_inputs['y'] = self.a*d_outputs['f']
                    d_inputs['x'] += y*d_outputs['g']
                    d_inputs['y'] += x*d_outputs['g']
                    d_inputs['z'] = 3.0*d_outputs['g']

        x = csdl.Variable(value= np.arange(6).reshape(3,2), name='x')
        y = csdl.Variable(value= -np.arange(6).reshape(3,2), name='y')
        z = csdl.Variable(value= np.arange(6).reshape(3,2)+0.5, name='z')

        paraboloid = Custom2(a=3, b=4, c=12)
        f, g = paraboloid.evaluate(x, y, z)

        f.add_name('f')
        g.add_name('g')
        f_sum = csdl.sum(f)
        g_sum = csdl.sum(g)
        derivs = csdl.derivative([f_sum, g_sum], [x, y, z])

        # f = a*y + x*2.0
        # df_dx = 2.0
        # df_dy = a
        # df_dz = 0.0

        # g = 3.0*z + x*y
        # dg_dx = y
        # dg_dy = x
        # dg_dz = 3.0

        df_dx = derivs[f_sum, x]
        df_dx_real = 2.0*np.ones((1,x.size))
        np.testing.assert_array_equal(df_dx.value, df_dx_real)

        df_dy = derivs[f_sum, y]
        df_dy_real = np.ones((1,x.size))*3.0
        np.testing.assert_array_equal(df_dy.value, df_dy_real)

        df_dz = derivs[f_sum, z]
        np.testing.assert_array_equal(df_dz.value, np.zeros((1,x.size)))

        dg_dx = derivs[g_sum, x]
        dg_dx_real = y.value.flatten().reshape(1,-1)
        np.testing.assert_array_equal(dg_dx.value, dg_dx_real)

        dg_dy = derivs[g_sum, y]
        dg_dy_real = x.value.flatten().reshape(1,-1)
        np.testing.assert_array_equal(dg_dy.value, dg_dy_real)

        dg_dz = derivs[g_sum, z]
        dg_dz_real = 3.0*np.ones(x.size).reshape(1,-1)
        np.testing.assert_array_equal(dg_dz.value, dg_dz_real)

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(f, f.value),
                csdl_tests.TestingPair(g, g.value),
                csdl_tests.TestingPair(f_sum, f_sum.value),
                csdl_tests.TestingPair(g_sum, g_sum.value)
            ],
            verify_derivatives=True,
        )

if __name__ == '__main__':
    t = TestCustom()
    t.overwrite_backend = 'inline'
    t.test_simple_deriv()
    t.test_derivs()