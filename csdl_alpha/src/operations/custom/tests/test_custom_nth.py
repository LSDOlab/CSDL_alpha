import csdl_alpha as csdl
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
import numpy as np
 
from csdl_alpha.src.operations.custom.tests.test_custom import Paraboloid

class CustomParaboloidDerivativeDerivative(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c
    
    def evaluate(self, inputs):
        self.declare_input('x', inputs['x'])
        self.declare_input('y', inputs['y'])
        self.declare_input('z', inputs['z'])

        # declare output variables (derivatives)
        size = inputs['x'].size
        g_x_x = self.create_output('g_x_x', (size*size, size))
        g_x_y = self.create_output('g_x_y', (size*size, size))
        g_y_x = self.create_output('g_y_x', (size*size, size))
        g_y_y = self.create_output('g_y_y', (size*size, size))

        # define derivatives:
        derivatives = {
            ('g_x', 'x'): g_x_x,
            ('g_x', 'y'): g_x_y,
            ('g_y', 'x'): g_y_x,
            ('g_y', 'y'): g_y_y,
        }
        return derivatives

    def compute(self, input_vals, outputs_vals):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']

        # Second-order derivatives
        ddg_dx2 = np.zeros((6, 6, 6))
        ddg_dxdy = np.zeros((6, 6, 6))

        ddg_dy2 = np.zeros((6, 6, 6))
        ddg_dydx = np.zeros((6, 6, 6)) 

        for i in range(6):
            x_i = x.flatten()[i]
            y_i = y.flatten()[i]
            
            # 'g' 'x'
            ddg_dx2[i, i, i] = 6 * x_i * y_i**2
            ddg_dxdy[i, i, i] = 6 * x_i**2 * y_i
            
            # 'g' 'y'
            ddg_dy2[i, i, i] = 2 * x_i**3
            ddg_dydx[i, i, i] = (3 * x_i**2) * (2 * y_i)
        
        # assign to derivatives
        outputs_vals['g_x_x'] = ddg_dx2.reshape(6*6, 6)
        outputs_vals['g_x_y'] = ddg_dxdy.reshape(6*6, 6)
        outputs_vals['g_y_x'] = ddg_dydx.reshape(6*6, 6)
        outputs_vals['g_y_y'] = ddg_dy2.reshape(6*6, 6)


class CustomParaboloidDerivative(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c
    
    def evaluate(self, inputs):
        self.declare_input('x', inputs['x'])
        self.declare_input('y', inputs['y'])
        self.declare_input('z', inputs['z'])

        # declare output variables (derivatives)
        size = inputs['x'].size
        f_x = self.create_output('f_x', (size, size))
        f_y = self.create_output('f_y', (size, size))
        g_x = self.create_output('g_x', (size, size))
        g_y = self.create_output('g_y', (size, size))
        g_z = self.create_output('g_z', (size, size))

        # define derivative function
        self.declare_derivative_function(CustomParaboloidDerivativeDerivative, a=self.a, b=self.b, c=self.c)

        # define derivatives:
        derivatives = {
            ('f', 'x'): f_x,
            ('f', 'y'): f_y,
            ('f', 'z'): None,
            ('g', 'x'): g_x,
            ('g', 'y'): g_y,
            ('g', 'z'): g_z,
        }
        return derivatives

    def compute(self, input_vals, output_vals):
        x, y, z = input_vals['x'], input_vals['y'], input_vals['z']

        # compute derivatives
        output_vals['f_x'] = np.diag(2.0*np.ones(x.size))
        output_vals['f_y'] = np.diag(self.a*np.ones(x.size))

        output_vals['g_x'] = np.diag(y.flatten()**2.0 * 3.0*x.flatten()**2.0)
        output_vals['g_y'] = np.diag(x.flatten()**3.0 * 2.0*y.flatten())
        output_vals['g_z'] = np.diag(3.0*np.ones(x.size))


class CustomParaboloid(csdl.experimental.CustomExplicitOperationBeta):
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

        # define derivative function
        self.declare_derivative_function(CustomParaboloidDerivative, a=self.a, b=self.b, c=self.c)

        return f, g
    
    def compute(self, input_vals, output_vals):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']

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