import csdl_alpha as csdl
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
import numpy as np
 
from csdl_alpha.src.operations.custom.tests.test_custom import Paraboloid


class CustomParaboloidVJPVJP(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c
    
    def evaluate(self, inputs, d_outputs):
        # Inputs of first-order VJP
        self.declare_input('x', inputs['x'])
        self.declare_input('y', inputs['y'])
        self.declare_input('z', inputs['z'])
        self.declare_input('d_f', inputs['d_f'])
        self.declare_input('d_g', inputs['d_g'])

        # d_outputs of second-order VJP
        self.declare_input('d_d_x', d_outputs['d_x'])
        self.declare_input('d_d_y', d_outputs['d_y'])
        self.declare_input('d_d_z', d_outputs['d_z'])

        # declare output variables (derivatives)
        shape = inputs['x'].shape
        d_x = self.create_output('d_x', inputs['x'].shape)
        d_y = self.create_output('d_y', inputs['y'].shape)
        d_d_f = self.create_output('d_d_f', inputs['d_f'].shape)
        d_d_g = self.create_output('d_d_g', inputs['d_g'].shape)

        # define derivatives:
        d_inputs = {
            'x': d_x,
            'y': d_y,
            'z': None,
            'd_f': d_d_f,
            'd_g': d_d_g,
        }
        return d_inputs

    def compute(self, input_vals, outputs_vals):
        x, y, z = input_vals['x'], input_vals['y'], input_vals['z']
        d_f, d_g = input_vals['d_f'], input_vals['d_g']
        d_d_x, d_d_y, d_d_z = input_vals['d_d_x'], input_vals['d_d_y'], input_vals['d_d_z']

        # Second-order VJPS
        outputs_vals['d_x'] = 6.0*x*(y**2.0)*d_g*d_d_x + 6.0*(x**2.0)*y*d_g*d_d_y
        outputs_vals['d_y'] = 6.0*(x**2.0)*(y)*d_g*d_d_x + 2.0*(x**3.0)*d_g*d_d_y
        outputs_vals['d_d_f'] = 2.0*d_d_x + self.a*d_d_y
        outputs_vals['d_d_g'] = 3.0*(x**2.0)*(y**2.0)*d_d_x + 2.0*(x**3.0)*y*d_d_y + 3.0*d_d_z

class CustomParaboloidVJP(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c
    
    def evaluate(self, inputs, d_outputs):
        self.declare_input('x', inputs['x'])
        self.declare_input('y', inputs['y'])
        self.declare_input('z', inputs['z'])

        self.declare_input('d_f', d_outputs['f'])
        self.declare_input('d_g', d_outputs['g'])

        # declare output variables (derivatives)
        shape = inputs['x'].shape
        d_x = self.create_output('d_x', shape)
        d_y = self.create_output('d_y', shape)
        d_z = self.create_output('d_z', shape)

        # define derivative function
        self.declare_vjp_function(CustomParaboloidVJPVJP , a=self.a, b=self.b, c=self.c)

        # define derivatives:
        d_inputs = {
            'x': d_x,
            'y': d_y,
            'z': d_z,
        }
        return d_inputs

    def compute(self, input_vals, output_vals):
        x, y, z = input_vals['x'], input_vals['y'], input_vals['z']
        d_f, d_g = input_vals['d_f'], input_vals['d_g']

        output_vals['d_x'] = 2.0*d_f + 3.0*(x**2.0)*(y**2.0)*d_g
        output_vals['d_y'] = self.a*d_f + 2.0*(x**3.0)*y*d_g
        output_vals['d_z'] = 3.0*d_g

class CustomParaboloid(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c

    def evaluate(self, x, y, z):
        # assign method _dict to input dictionary
        self.declare_input('x', x)
        self.declare_input('y', y)
        self.declare_input('z', z)

        # declare output variables
        f,g = self.create_output('f', x.shape), self.create_output('g', x.shape)

        # define derivative function
        # self.declare_derivative_function(CustomParaboloidDerivative, a=self.a, b=self.b, c=self.c)
        self.declare_vjp_function(CustomParaboloidVJP, a=self.a, b=self.b, c=self.c)
        return f, g
    
    def compute(self, input_vals, output_vals):
        x,y,z = input_vals['x'], input_vals['y'], input_vals['z']

        output_vals['f'] = self.a*y + x*2.0
        output_vals['g'] = 3.0*z + (x**3.0)*(y**2.0)

class TestCustom(csdl_tests.CSDLTest):

    def test_derivs(self):
        self.prep(always_build_inline = True)
 
        import numpy as np
        x = csdl.Variable(value= np.arange(6).reshape(3,2), name='x')
        y = csdl.Variable(value= -np.arange(6).reshape(3,2), name='y')
        z = csdl.Variable(value= np.arange(6).reshape(3,2)+0.5, name='z')

        paraboloid = CustomParaboloid(a=3, b=4, c=12)
        f, g = paraboloid.evaluate(x, y, z)

        f.add_name('f')
        g.add_name('g')
        f_sum = csdl.sum(f)
        g_sum = csdl.sum(g)
        derivs = csdl.derivative([f_sum, g_sum], [x, y, z])

        df_dx = derivs[f_sum, x]
        df_dx_real = 2.0*np.ones((1,x.size))
        np.testing.assert_array_equal(df_dx.value, df_dx_real)

        df_dy = derivs[f_sum, y]
        df_dy_real = np.ones((1,x.size))*3.0
        np.testing.assert_array_equal(df_dy.value, df_dy_real)

        df_dz = derivs[f_sum, z]
        np.testing.assert_array_equal(df_dz.value, np.zeros((1,x.size)))

        dg_dx = derivs[g_sum, x]
        dg_dx_real = (y.value.flatten()**2.0 * 3.0*x.value.flatten()**2.0).reshape(1,-1)
        np.testing.assert_array_equal(dg_dx.value, dg_dx_real)

        dg_dy = derivs[g_sum, y]
        dg_dy_real = (x.value.flatten()**3.0 * 2.0*y.value.flatten()).reshape(1,-1)
        np.testing.assert_array_equal(dg_dy.value, dg_dy_real)

        dg_dz = derivs[g_sum, z]
        dg_dz_real = 3.0*np.ones(x.size).reshape(1,-1)
        np.testing.assert_array_equal(dg_dz.value, dg_dz_real)

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(f, f.value),
                csdl_tests.TestingPair(g, g.value),
                csdl_tests.TestingPair(f_sum, f_sum.value),
                csdl_tests.TestingPair(g_sum, g_sum.value),
                csdl_tests.TestingPair(df_dx, df_dx_real),
                csdl_tests.TestingPair(df_dy, df_dy_real),
                csdl_tests.TestingPair(df_dz, np.zeros((1,x.size))),
                csdl_tests.TestingPair(dg_dx, dg_dx_real),
                csdl_tests.TestingPair(dg_dy, dg_dy_real),
                csdl_tests.TestingPair(dg_dz, dg_dz_real),
            ],
            verify_derivatives=True,
        )

if __name__ == '__main__':
    t = TestCustom()
    t.overwrite_backend = 'inline'
    t.test_derivs()
    # t.test_2nd_derivs()