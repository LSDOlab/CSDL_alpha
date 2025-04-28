import csdl_alpha as csdl
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
import numpy as np
 
from csdl_alpha.src.operations.custom.tests.test_custom import Paraboloid
 
class CustomParaboloidDerivative(csdl.experimental.CustomExplicitOperationBeta):
    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c
    
    def evaluate(self,input_dict):
        self.declare_input('x', input_dict['x'])
        self.declare_input('y', input_dict['y'])
        self.declare_input('z', input_dict['z'])

        # declare output variables (derivatives)
        size = input_dict['x'].size
        f_x = self.create_output('f_x', (size, size))
        f_y = self.create_output('f_y', (size, size))
        g_x = self.create_output('g_x', (size, size))
        g_y = self.create_output('g_y', (size, size))
        g_z = self.create_output('g_z', (size, size))

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
        output_vals['g'] = 3.0*z + x*y


class Custom3(csdl.CustomExplicitOperation):
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
        return f, g
    
    def compute(self, input_vals, output_vals):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']

        output_vals['f'] = self.a*y + x*2.0

        output_vals['g'] = 3.0*z + (x**3.0)*(y**2.0)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']
        derivatives['f', 'x'] = np.diag(2.0*np.ones(x.size))
        derivatives['f', 'y'] = np.diag(self.a*np.ones(x.size))

        derivatives['g', 'x'] = np.diag(y.flatten()**2.0 * 3.0*x.flatten()**2.0)
        derivatives['g', 'y'] = np.diag(x.flatten()**3.0 * 2.0*y.flatten())
        derivatives['g', 'z'] = np.diag(3.0*np.ones(x.size))

    def compute_nth_derivatives(self, input_vals, outputs_vals, derivatives, order):
        x = input_vals['x']
        y = input_vals['y']
        z = input_vals['z']

        if order == 1:
            derivatives['f', 'x'] = np.diag(2.0*np.ones(x.size))
            derivatives['f', 'y'] = np.diag(self.a*np.ones(x.size))

            derivatives['g', 'x'] = np.diag(y.flatten()**2.0 * 3.0*x.flatten()**2.0)
            derivatives['g', 'y'] = np.diag(x.flatten()**3.0 * 2.0*y.flatten())
            derivatives['g', 'z'] = np.diag(3.0*np.ones(x.size))
        elif order == 2:
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
            derivatives['g', 'x', 'x'] = ddg_dx2.reshape(6*6, 6)
            derivatives['g', 'x', 'y'] = ddg_dxdy.reshape(6*6, 6)
            derivatives['g', 'y', 'x'] = ddg_dydx.reshape(6*6, 6)
            derivatives['g', 'y', 'y'] = ddg_dy2.reshape(6*6, 6)

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

    def test_2nd_derivs(self):
        self.prep(always_build_inline = True)
 
        import numpy as np
        x = csdl.Variable(value= np.arange(6).reshape(3,2), name='x')
        y = csdl.Variable(value= -np.arange(6).reshape(3,2), name='y')
        z = csdl.Variable(value= np.arange(6).reshape(3,2)+0.5, name='z')

        paraboloid = Custom3(a=3, b=4, c=12)
        f, g = paraboloid.evaluate(x, y, z)

        f.add_name('f')
        g.add_name('g')
        f_sum = csdl.sum(f)
        g_sum = csdl.sum(g)
        derivs = csdl.derivative([f_sum, g_sum], [x, y, z])

        df_dx = derivs[f_sum, x]
        df_dx_real = 2.0*np.ones((1,x.size))

        df_dy = derivs[f_sum, y]
        df_dy_real = np.ones((1,x.size))*3.0

        df_dz = derivs[f_sum, z]

        dg_dx = derivs[g_sum, x]
        dg_dx_real = y.value.flatten().reshape(1,-1)

        dg_dy = derivs[g_sum, y]
        dg_dy_real = x.value.flatten().reshape(1,-1)

        dg_dz = derivs[g_sum, z]
        dg_dz_real = 3.0*np.ones(x.size).reshape(1,-1)

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(f, f.value),
                csdl_tests.TestingPair(g, g.value),
                csdl_tests.TestingPair(f_sum, f_sum.value),
                csdl_tests.TestingPair(g_sum, g_sum.value),
                # csdl_tests.TestingPair(df_dx, df_dx_real),
            ],
            verify_derivatives=True,
        )

if __name__ == '__main__':
    t = TestCustom()
    t.overwrite_backend = 'inline'
    t.test_derivs()
    # t.test_2nd_derivs()