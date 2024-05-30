import csdl_alpha as csdl
import numpy as np
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest

class CustomImp(csdl.experimental.CustomImplicitOperation):

    def initialize(self):
        self.const = self.parameters.declare('a')

    def evaluate(self, a, b, c):
        self.declare_input('a', a)
        self.declare_input('b', b)
        self.declare_input('c', c)
        x = self.create_output('x', shape = b.shape)
        y = self.create_output('y', shape = b.shape)
        return x, y

    def solve_residual_equations(self, inputs, outputs):
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']
        cy = c/2.0
        ay = b+a

        # quadratic equation here:
        x_solved = (-(b) + np.sqrt(b**2 - 4*a*c))/(2*a)
        y_solved = (-(b) + np.sqrt(b**2 - 4*ay*cy))/(2*ay)

        # make sure residual == 0:
        residual_x = a*x_solved**2 + b*x_solved + c
        residual_y = ay*y_solved**2 + b*y_solved + cy
        print('residual_x:', residual_x)
        print('residual_y:', residual_y)
        assert residual_x < 1e-10
        assert residual_y < 1e-10

        # set the outputs
        self.x_solved = x_solved
        self.y_solved = y_solved
        outputs['x'] = x_solved
        outputs['y'] = y_solved

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']
        cy = c/2.0
        ay = b+a

        x_solved = outputs['x']
        y_solved = outputs['y']
        # for mode = rev
        # d_residuals --> d_inputs
        if mode == 'rev':
            d_inputs['a'][1,1] = d_residuals['x']*x_solved**2 + d_residuals['y']*y_solved**2
            d_inputs['b'] = d_residuals['x']*x_solved + d_residuals['y']*(y_solved+y_solved**2)
            d_inputs['c'] = d_residuals['x'] + d_residuals['y']/2.0

    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']
        cy = c/2.0
        ay = b+a

        x_solved = outputs['x']
        y_solved = outputs['y']

        # for mode = rev:
        # d_outputs --> d_residuals
        if mode == 'rev':
            d_residuals['x'] = 1.0/(2*a*x_solved+b)*d_outputs['x']
            d_residuals['y'] = 1.0/(2*ay*y_solved+b)*d_outputs['y']


class CustomImpVec(csdl.experimental.CustomImplicitOperation):

    def initialize(self):
        self.const = self.parameters.declare('a')

    def evaluate(self, a, b, c):
        self.declare_input('a', a)
        self.declare_input('b', b)
        self.declare_input('c', c)
        x = self.create_output('x', shape = b.shape)
        y = self.create_output('y', shape = b.shape)
        return x, y

    def solve_residual_equations(self, inputs, outputs):
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']
        cy = c/2.0
        ay = b+a

        # quadratic equation here:
        x_solved = (-(b) + np.sqrt(b**2 - 4*a*c))/(2*a)
        y_solved = (-(b) + np.sqrt(b**2 - 4*ay*cy))/(2*ay)

        # make sure residual == 0:
        residual_x = a*x_solved**2 + b*x_solved + c
        residual_y = ay*y_solved**2 + b*y_solved + cy
        print('residual_x:', residual_x)
        print('residual_y:', residual_y)
        assert np.all(residual_x < 1e-10)
        assert np.all(residual_y < 1e-10)

        # set the outputs
        self.x_solved = x_solved
        self.y_solved = y_solved
        outputs['x'] = x_solved
        outputs['y'] = y_solved

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']
        cy = c/2.0
        ay = b+a

        x_solved = outputs['x']
        y_solved = outputs['y']
        # for mode = rev
        # d_residuals --> d_inputs
        if mode == 'rev':
            d_inputs['a'][1,1] = np.sum(d_residuals['x']*x_solved**2 + d_residuals['y']*y_solved**2)
            d_inputs['b'] = d_residuals['x']*x_solved + d_residuals['y']*(y_solved+y_solved**2)
            d_inputs['c'] = d_residuals['x'] + d_residuals['y']/2.0

    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        a = inputs['a'][1, 1]
        b = inputs['b']
        c = inputs['c']
        cy = c/2.0
        ay = b+a

        x_solved = outputs['x']
        y_solved = outputs['y']

        # for mode = rev:
        # d_outputs --> d_residuals
        if mode == 'rev':
            d_residuals['x'] = 1.0/(2*a*x_solved+b)*d_outputs['x']
            d_residuals['y'] = 1.0/(2*ay*y_solved+b)*d_outputs['y']

class TestCustomImplicit(csdl_tests.CSDLTest):
    def test_custom_implicit(self):
        self.prep()
        a = csdl.Variable(value=np.ones((2,2)), name='a')
        b = csdl.Variable(value=10., name='b')
        c = csdl.Variable(value=3, name='c')

        custom_imp = CustomImp(a=2)
        x, y = custom_imp.evaluate(a, b, c)

        # check the derivative using real implict operation:
        x2 = csdl.ImplicitVariable(value=np.ones(b.shape), name='x2')
        y2 = csdl.ImplicitVariable(value=np.ones(b.shape), name='y2')
        a_11 = a[1, 1]
        cy = c/2.0
        ay = b+a_11
        residual_x = a_11*x2**2 + b*x2 + c
        residual_y = ay*y2**2 + b*y2 + cy

        solver = csdl.nonlinear_solvers.Newton()
        solver.add_state(x2, residual_x)
        solver.add_state(y2, residual_y)
        solver.run()

        print('x2: ', x2.value)
        print('x:  ', x.value)
        print('y2: ', y2.value)
        print('y:  ', y.value)

        np.testing.assert_almost_equal(x2.value, x.value)
        np.testing.assert_almost_equal(y2.value, y.value)

        real_derivs = csdl.derivative([x2, y2], [a, b, c])
        cust_derivs = csdl.derivative([x, y], [a, b, c])

        np.testing.assert_almost_equal(real_derivs[x2, a].value, cust_derivs[x, a].value)
        np.testing.assert_almost_equal(real_derivs[x2, b].value, cust_derivs[x, b].value)
        np.testing.assert_almost_equal(real_derivs[x2, c].value, cust_derivs[x, c].value)
        np.testing.assert_almost_equal(real_derivs[y2, a].value, cust_derivs[y, a].value)
        np.testing.assert_almost_equal(real_derivs[y2, b].value, cust_derivs[y, b].value)
        np.testing.assert_almost_equal(real_derivs[y2, c].value, cust_derivs[y, c].value)

    def test_custom_implicit_vec(self):
        self.prep()
        a = csdl.Variable(value=np.ones((2,2)), name='a')
        b = csdl.Variable(value=np.array([[10],[11]]), name='b')
        c = csdl.Variable(value=np.array([[3],[2]]), name='c')

        custom_imp = CustomImpVec(a=2)
        x, y = custom_imp.evaluate(a, b, c)

        # check the derivative using real implict operation:
        x2 = csdl.ImplicitVariable(value=np.ones(b.shape), name='x2')
        y2 = csdl.ImplicitVariable(value=np.ones(b.shape), name='y2')
        a_11 = a[1, 1]
        cy = c/2.0
        ay = b+a_11
        residual_x = a_11*x2**2 + b*x2 + c
        residual_y = ay*y2**2 + b*y2 + cy

        solver = csdl.nonlinear_solvers.Newton()
        solver.add_state(x2, residual_x)
        solver.add_state(y2, residual_y)
        solver.run()

        print('x2: ', x2.value)
        print('x:  ', x.value)
        print('y2: ', y2.value)
        print('y:  ', y.value)

        np.testing.assert_almost_equal(x2.value, x.value)
        np.testing.assert_almost_equal(y2.value, y.value)

        real_derivs = csdl.derivative([x2, y2], [a, b, c])
        cust_derivs = csdl.derivative([x, y], [a, b, c])

        np.testing.assert_almost_equal(real_derivs[x2, a].value, cust_derivs[x, a].value)
        np.testing.assert_almost_equal(real_derivs[x2, b].value, cust_derivs[x, b].value)
        np.testing.assert_almost_equal(real_derivs[x2, c].value, cust_derivs[x, c].value)
        np.testing.assert_almost_equal(real_derivs[y2, a].value, cust_derivs[y, a].value)
        np.testing.assert_almost_equal(real_derivs[y2, b].value, cust_derivs[y, b].value)
        np.testing.assert_almost_equal(real_derivs[y2, c].value, cust_derivs[y, c].value)

if __name__ == '__main__':
    t = TestCustomImplicit()
    t.test_custom_implicit()
    t.test_custom_implicit_vec()