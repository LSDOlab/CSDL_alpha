from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
import csdl_alpha.utils.testing_utils as csdl_tests
import numpy as np

class SubOperation(SubgraphOperation):
    def __init__(self, subgraph, inputs, outputs, name, jit=False):
        super().__init__(*inputs)
        self.name = name
        self.jit = jit
        self.outputs = outputs
        self.assign_subgraph(subgraph)
        for output in outputs:
            self.recorder._add_node(output)
        self._add_to_graph()

    def compute_inline(self, *args):
        for i, input in enumerate(self.inputs):
            input.value = args[i]
        self._subgraph.execute_inline()
        return tuple([output.value for output in self.outputs])
    
    def compute_jax(self, *args):
        from csdl_alpha.backends.jax import create_jax_function
        import jax
        jax_function = create_jax_function(self._subgraph, self.outputs, self.inputs)
        if self.jit:
            output = jax.pure_callback(jax.jit(jax_function),
                                       [jax.ShapeDtypeStruct(output.shape, np.float64) for output in self.outputs],
                                       *args)
            return tuple(output)
            # return tuple(jax.jit(jax_function)(*args))
        else:
            return tuple(jax_function(*args))
                    
    def evaluate_vjp(self, cotangents, *inputs_outputs):
        import csdl_alpha as csdl
        from csdl_alpha.src.operations.derivatives.reverse import vjp
        
        with subop(name=f'{self.name}_vjp', jit=self.jit) as sub:
            # Get the parent op to the current graph
            from csdl_alpha.src.graph.graph import _copy_to_current_graph
            _copy_to_current_graph(self._subgraph, {})
            # add inputs to inputs
            csdl.get_current_recorder().active_graph.inputs += self.inputs
            
            inputs = inputs_outputs[:self.num_inputs] # these could just be self.inputs? Same for outputs?
            outputs = inputs_outputs[self.num_inputs:]

            wrts = []
            for input_var in inputs:
                if cotangents.check(input_var):
                    wrts.append(input_var)

            # All the inputs we pass into the composed operation
            output_cotangents = []
            for of in outputs:
                cotangents.initialize(of)
                if cotangents[of] is None:
                    cotangents.accumulate(of, csdl.Variable(value = np.zeros(of.shape)))
                output_cotangents.append(cotangents[of])

            seeds = []
            for i, output_var in enumerate(self.outputs):
                seeds.append((output_var, output_cotangents[i]))

            wrts_composed = vjp(seeds, wrts, self._subgraph)
            
            outputs_composed = []
            for wrt_composed, wrt_cotangent in wrts_composed.items():
                if wrt_cotangent is None:
                    zeros = csdl.Variable(value = np.zeros(wrt_composed.shape))
                    outputs_composed.append(zeros)
                else:
                    outputs_composed.append(csdl.copyvar(wrt_cotangent))
                # print('IBE', wrt_composed.shape, wrt_cotangent.shape)

            for output in outputs_composed:
                sub.add_output(output)            

        i = 0
        for input_var in inputs:
            if cotangents.check(input_var):
                cotangents.accumulate(input_var, outputs_composed[i])
                i+=1       

class subop:
    def __init__(self, name:str='custom', add_all_outputs=False, jit=False):
        """Packages operations into a single subgraph operation

        Inputs to the subop are automatically added. Outputs must be added manually via the add_output method.
        If add_all_outputs is set to True, all output variables will be added as outputs, and the add_output method will be ignored.

        Parameters
        ----------
        name : str, optional
            The name of the SubOperation, by default 'custom'
        add_all_outputs : bool, optional
            Whether to add all output variabes as outputs, by default False.
            This will usually increase compile and run time in jax when using the jit option.
        jit : bool, optional
            Whether to jit the operation seperately in jax, by default False.
            This can reduce compile time for large operations, but may increase runtime.
        """
        import csdl_alpha as csdl
        self.name = name
        self.jit = jit
        self.add_all_outputs = add_all_outputs
        self.recorder = csdl.get_current_recorder()
        self.outputs = []

    def __enter__(self):
        self.recorder._enter_subgraph(add_missing_variables=True)
        return self
    
    def add_output(self, output: Variable):
            """Add an output variable to the operation.

            Parameters
            ----------
            output : Variable
                The output variable to be added.
            """
            self.outputs.append(output)

    def __exit__(self, exc_type, exc_val, exc_tb):
        subgraph = self.recorder.active_graph
        self.recorder._exit_subgraph()
        inputs = subgraph.inputs
        if self.add_all_outputs:
            self.outputs = []
            for node in subgraph.node_table:
                if isinstance(node, Operation):
                    for output in node.outputs:
                        self.outputs.append(output)
        self.recorder._add_node(SubOperation(subgraph, inputs, self.outputs, self.name, jit=self.jit))

class TestSubOp(csdl_tests.CSDLTest):
    @staticmethod
    def simple_model(x, y):
        import csdl_alpha as csdl
        return x*y + 3*x*y**2 + 5*x**2*y + 7*x**2*y**2
    
    @staticmethod
    def d_simple_model_dx(x, y):
        return y + 3*y**2 + 10*x*y + 14*x*y**2

    def test_simple(self):
        import csdl_alpha as csdl

        self.prep()
        x = csdl.Variable(name='x', value=2.0)
        y = csdl.Variable(name='y', value=3.0)
        with subop('simple') as sub:
            z = self.simple_model(x, y)
            sub.add_output(z)
        z_np = self.simple_model(x.value, y.value)
        compare_values = []
        compare_values += [csdl_tests.TestingPair(z, z_np, tag = 'simple')]

        self.run_tests(compare_values, verify_derivatives=True)

    def test_nested(self):
        import csdl_alpha as csdl

        self.prep()

        x = csdl.Variable(name='x', value=2.0)
        y = csdl.Variable(name='y', value=3.0)
        z = x + y
        with subop('simple') as sub:
            x2 = x*z
            y2 = y*z
            with subop('nested') as sub2:
                z2 = self.simple_model(x2, y2)
                sub2.add_output(z2)
            sub.add_output(y2)
            sub.add_output(z2)

        z_np = x.value + y.value
        x2_np = x.value*z_np
        y2_np = y.value*z_np
        z2_np = self.simple_model(x2_np, y2_np)
        compare_values = []
        compare_values += [csdl_tests.TestingPair(z, z_np, tag = 'z')]
        compare_values += [csdl_tests.TestingPair(y2, y2_np, tag = 'y2')]
        compare_values += [csdl_tests.TestingPair(z2, z2_np, tag = 'z2')]

        self.run_tests(compare_values, verify_derivatives=True)

    def test_simple_ddx(self):
        import csdl_alpha as csdl

        self.prep()
        x = csdl.Variable(name='x', value=2.0)
        y = csdl.Variable(name='y', value=3.0)
        with subop('simple') as sub:
            z = self.simple_model(x, y)
            sub.add_output(z)
        z_np = self.simple_model(x.value, y.value)
        compare_values = []
        compare_values += [csdl_tests.TestingPair(z, z_np, tag = 'simple')]

        dz = csdl.derivative(z, x)
        dz.add_name('dz_dx')
        dz_np = self.d_simple_model_dx(x.value, y.value).reshape((1,1))
        compare_values += [csdl_tests.TestingPair(dz, dz_np, tag = 'dz_dx')]

        self.run_tests(compare_values, verify_derivatives=True)




if __name__ == '__main__':
    test = TestSubOp()
    test.overwrite_backend = 'jax'
    test.test_simple()
    test.test_nested()
    test.test_simple_ddx()