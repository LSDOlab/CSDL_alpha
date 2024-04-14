import csdl_alpha as csdl
import csdl_alpha.utils.test_utils as csdl_tests
import pytest

# basic custom explicit
class BasicCustomExplicitOperation(csdl.CustomExplicitOperation):
    def initialize(self):
        pass

# custom paraboloid model
class Paraboloid(csdl.CustomExplicitOperation):
    def initialize(self):
        self.a = self.parameters.declare('a')
        self.b = self.parameters.declare('b')
        self.c = self.parameters.declare('c')
        self.return_g = self.parameters.declare('return_g', default=False)

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

        derivatives['f', 'x'] = 2*x - self.a + y
        derivatives['f', 'y'] = 2*y + x + self.b

        if self.return_g:
            derivatives['g', 'x'] = z*derivatives['f', 'x']
            derivatives['g', 'y'] = z*derivatives['f', 'x']
            derivatives['g', 'z'] = outputs_vals['f']

class TestCustom(csdl_tests.CSDLTest):
    def test_not_implemented(self):
        self.prep()

        # with pytest.raises(NotImplementedError) as e_info:
        #     csdl.CustomExplicitOperation()

        custom_model = BasicCustomExplicitOperation()

        methods = [
            ("evaluate", {}),
            ("compute", ({}, {})),
            ("compute_derivatives", ({}, {}, {})),
            ("evaluate_diagonal_jacobian", {}),
            ("evaluate_jvp", {}),
            ("evaluate_vjp", {}),
        ]

        for method, args in methods:
            with pytest.raises(NotImplementedError) as e_info:
                getattr(custom_model, method)(*args)

    # def test_derivative_parameters(self):
    #     self.prep()
    #     model = BasicCustomExplicitOperation()
    #     model.outputs = {'y1': csdl.Variable(value=1), 'y2': csdl.Variable(value=2)}
    #     model._dict = {'x1': csdl.Variable(value=1), 'x2': csdl.Variable(value=2)}

    #     with pytest.raises(TypeError) as e_info:
    #         model.declare_derivative_parameters(0, 'x1')
    #     with pytest.raises(TypeError) as e_info:
    #         model.declare_derivative_parameters('y1', 0)

    #     model.declare_derivative_parameters('*', 'x1')
    #     assert model.derivative_parameters.keys() == {('y1', 'x1'), ('y2', 'x1')}
    #     model.derivative_parameters = {}

    #     model.declare_derivative_parameters()

    def test_declare_derivative_parameters(self):
        self.prep()
        
        model = BasicCustomExplicitOperation()
        model.input_dict = {'x': csdl.Variable(value=1), 'y': csdl.Variable(value=2)}
        model.output_dict = {'f': csdl.Variable(value=3), 'g': csdl.Variable(value=4)}

        # Test case 1: Single derivative declaration
        model.declare_derivative_parameters('f', 'x')
        assert ('f', 'x') in model.derivative_parameters.keys()
        assert model.derivative_parameters[('f', 'x')]['dependent'] == True
        model.derivative_parameters = {}

        # Test case 2: Multiple derivative declarations
        model.declare_derivative_parameters(['f', 'g'], ['x', 'y'])
        assert ('f', 'x') in model.derivative_parameters.keys()
        assert ('f', 'y') in model.derivative_parameters.keys()
        assert ('g', 'x') in model.derivative_parameters.keys()
        assert ('g', 'y') in model.derivative_parameters.keys()
        model.derivative_parameters = {}

        # Test case 3: Wildcard derivative declarations
        model.declare_derivative_parameters('*', 'x')
        assert ('f', 'x') in model.derivative_parameters.keys()
        assert ('g', 'x') in model.derivative_parameters.keys()
        model.derivative_parameters = {}

        model.declare_derivative_parameters('f', '*')
        assert ('f', 'x') in model.derivative_parameters.keys()
        assert ('f', 'y') in model.derivative_parameters.keys()
        model.derivative_parameters = {}

        model.declare_derivative_parameters('*', '*')
        assert ('f', 'x') in model.derivative_parameters.keys()
        assert ('f', 'y') in model.derivative_parameters.keys()
        assert ('g', 'x') in model.derivative_parameters.keys()
        assert ('g', 'y') in model.derivative_parameters.keys()
        model.derivative_parameters = {}

        model.declare_derivative_parameters(['*'], ['*'])
        assert ('f', 'x') in model.derivative_parameters.keys()
        assert ('f', 'y') in model.derivative_parameters.keys()
        assert ('g', 'x') in model.derivative_parameters.keys()
        assert ('g', 'y') in model.derivative_parameters.keys()
        model.derivative_parameters = {}

        # Test case 4: Invalid input types
        with pytest.raises(TypeError):
            model.declare_derivative_parameters(0, 'x')
        with pytest.raises(TypeError):
            model.declare_derivative_parameters('f', 0)

        # Test case 5: Duplicate derivative declaration
        with pytest.raises(KeyError):
            model.declare_derivative_parameters('f', 'x')
            model.declare_derivative_parameters('f', 'x')


    def test_paraboloid(self):
        self.prep()

        import numpy as np

        x = csdl.Variable(value=1, name='x')
        y = csdl.Variable(value=2, name='y')
        z = csdl.Variable(value=3, name='z')

        paraboloid = Paraboloid(a=2, b=4, c=12, return_g=True)
        outputs = paraboloid.evaluate(x, y, z)

        f = outputs.f
        g = outputs.g

        self.run_tests(
            compare_values = [
                csdl_tests.TestingPair(f, np.array([27]), tag = 'f'),
                csdl_tests.TestingPair(g, np.array([81]), tag = 'g'),
            ],
        )
