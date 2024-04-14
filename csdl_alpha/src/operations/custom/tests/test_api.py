import csdl_alpha as csdl
import csdl_alpha.utils.test_utils as csdl_tests
import pytest

# basic custom explicit

class CustomOp(csdl.CustomExplicitOperation):
    def initialize(self):
        self.a = self.parameters.declare('a', types = (int, float))

    def evaluate(self, x1, x2, x3):
        self.declare_input('x1', x1)
        self.declare_input('x2', x2)
        self.declare_input('x3', x3)

        f = self.create_output('f', x1.shape)

        output = csdl.VariableGroup()
        output.f = f

        return output

class TestCustom(csdl_tests.CSDLTest):
    def test_evaluate_errors(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        x1 = csdl.Variable(value=np.array([1, 2, 3]))
        x2 = csdl.Variable(value=1.0)
        x3 = csdl.Variable(value=np.array([[1.0, 2.0], [3.0, 4.0]]))

        # check parameters
        with pytest.raises(TypeError):
            custom_op_test = CustomOp(a = 's')
    
        # check declare_input
        with pytest.raises(TypeError):
            custom_op_test = CustomOp(a = 1)
            outputs = custom_op_test.evaluate('s', x2, x3)

        # Cannot use same key for declare_input
        with pytest.raises(KeyError):
            class CustomOp2(csdl.CustomExplicitOperation):
                def evaluate(self,x1,x2):
                    self.declare_input('x1', x1)
                    self.declare_input('x1', x2)
                    return self.create_output('f', x1.shape)
                def compute(self, inputs, outputs):
                    pass
            custom_op_test2 = CustomOp2()
            outputs = custom_op_test2.evaluate(x1, x2)
            
        # Cannot use same key for create_output
        with pytest.raises(KeyError):
            class CustomOp2(csdl.CustomExplicitOperation):
                def evaluate(self,x1):
                    self.declare_input('x1', x1)
                    self.create_output('f', x1.shape)
                    return self.create_output('f', x1.shape)
                def compute(self, inputs, outputs):
                    pass
            custom_op_test2 = CustomOp2()
            outputs = custom_op_test2.evaluate(x1)

    def test_runtime_errors(self):
        

        # outputs must be populated with keys of outputs
        with pytest.raises(KeyError):
            self.prep()

            import csdl_alpha as csdl
            import numpy as np
            recorder = csdl.Recorder(inline=True)
            recorder.start()

            x1 = csdl.Variable(value=np.array([1, 2, 3]))
            x2 = csdl.Variable(value=1.0)

            class CustomOp(csdl.CustomExplicitOperation):
                def evaluate(self,x1, x2):
                    self.declare_input('x1', x1)
                    self.declare_input('x2', x2)
                    f1 = self.create_output('f1', x1.shape)
                    f2 = self.create_output('f2', x1.shape)
                    return f1, f2
                def compute(self, inputs, outputs):
                    pass
            op = CustomOp()
            f1, f2 = op.evaluate(x1, x2)

        # outputs must be populated with declared keys
        with pytest.raises(KeyError):
            self.prep()

            import csdl_alpha as csdl
            import numpy as np
            recorder = csdl.Recorder(inline=True)
            recorder.start()

            x1 = csdl.Variable(value=np.array([1, 2, 3]))
            x2 = csdl.Variable(value=1.0)

            class CustomOp(csdl.CustomExplicitOperation):
                def evaluate(self,x1, x2):
                    self.declare_input('x1', x1)
                    self.declare_input('x2', x2)
                    f1 = self.create_output('f1', x1.shape)
                    f2 = self.create_output('f2', x1.shape)
                    return f1, f2
                def compute(self, inputs, outputs):
                    outputs['f3'] = inputs['x1']
                    outputs['f1'] = inputs['x1']
                    outputs['f2'] = inputs['x2']

            op = CustomOp()
            f1, f2 = op.evaluate(x1, x2)

        # outputs must be populated with correct value types
        with pytest.raises(TypeError):
            self.prep()

            import csdl_alpha as csdl
            import numpy as np
            recorder = csdl.Recorder(inline=True)
            recorder.start()

            x1 = csdl.Variable(value=np.array([1, 2, 3]))
            x2 = csdl.Variable(value=1.0)

            class CustomOp(csdl.CustomExplicitOperation):
                def evaluate(self,x1, x2):
                    self.declare_input('x1', x1)
                    self.declare_input('x2', x2)
                    f1 = self.create_output('f1', x1.shape)
                    f2 = self.create_output('f2', x1.shape)
                    return f1, f2
                def compute(self, inputs, outputs):
                    outputs['f1'] = 'hello'
            op = CustomOp()
            f1, f2 = op.evaluate(x1, x2)

        # outputs must be populated with shapes
        with pytest.raises(ValueError):
            self.prep()

            import csdl_alpha as csdl
            import numpy as np
            recorder = csdl.Recorder(inline=True)
            recorder.start()

            x1 = csdl.Variable(value=np.array([1, 2, 3]))
            x2 = csdl.Variable(value=1.0)

            class CustomOp(csdl.CustomExplicitOperation):
                def evaluate(self,x1, x2):
                    self.declare_input('x1', x1)
                    self.declare_input('x2', x2)
                    f1 = self.create_output('f1', (3,2))
                    f2 = self.create_output('f2', x1.shape)
                    return f1, f2
                def compute(self, inputs, outputs):
                    outputs['f1'] = inputs['x1']
            op = CustomOp()
            f1, f2 = op.evaluate(x1, x2)

        # inputs cannot be modified
        with pytest.raises(RuntimeError):
            self.prep()

            import csdl_alpha as csdl
            import numpy as np
            recorder = csdl.Recorder(inline=True)
            recorder.start()

            x1 = csdl.Variable(value=np.array([1, 2, 3]))
            x2 = csdl.Variable(value=1.0)

            class CustomOp(csdl.CustomExplicitOperation):
                def evaluate(self,x1, x2):
                    self.declare_input('x1', x1)
                    self.declare_input('x2', x2)
                    f1 = self.create_output('f1', (3,2))
                    f2 = self.create_output('f2', x1.shape)
                    return f1, f2
                def compute(self, inputs, outputs):
                    inputs['x1'] = inputs['x1'].flatten()
                    
            op = CustomOp()
            f1, f2 = op.evaluate(x1, x2)

if __name__ == '__main__':
    test = TestCustom()
    test.test_evaluate_errors()
    test.test_runtime_errors()